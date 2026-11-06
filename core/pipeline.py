"""VideoPipeline: the public façade (§10.3). fit() runs stages 1->2->3->5->6 (stage 4,
GlobalReID, is deferred in v0 per SPEC §8 — each track id is its own canonical entity)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from core.cache import artifacts
from core.cache.keys import file_hash, stage_key
from core.detect.base import TrackedBox
from core.detect.detector import DetectConfig
from core.detect.ensemble import EnsembleConfig, track_keyframes_ensemble
from core.detect.open_vocab import OpenVocabConfig, track_keyframes_open_vocab
from core.detect.tracker import track_keyframes
from core.extractors.base import ExtractionDrop, extract_with_retry
from core.extractors.openrouter import OpenRouterBackend
from core.extractors.prompts import som_prompt
from core.extractors.som_renderer import render
from core.graph.base import GraphEntity, GraphStore
from core.graph.kuzu_store import KuzuStore
from core.preprocessing.filter import FilterConfig, DualPhaseFilter
from core.reduce.reducer import StateReducer
from core.report import Report, build_report
from core.retriever.compiler.kuzu import execute_ir
from core.retriever.nl import question_to_ir
from core.schema.types import Entity, SkillSchema, TemporalEdge, make_semantic_delta_model


@dataclass(frozen=True)
class ExtractConfig:
    model: str = "google/gemini-2.5-flash"
    max_retries: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    """detect_backend:
    - "ensemble" (default): YOLO for COCO-covered schema classes, OWLv2 for the rest. The only
      backend that covers a schema like WAREHOUSE, whose classes straddle both (see ensemble.py).
    - "open_vocab": OWLv2 alone. Loses person recall on surveillance footage.
    - "yolo": COCO alone. Cannot see door/forklift/pallet at all.
    """

    schema: SkillSchema
    filter: FilterConfig = field(default_factory=FilterConfig)
    detect_backend: Literal["ensemble", "open_vocab", "yolo"] = "ensemble"
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    open_vocab: OpenVocabConfig = field(default_factory=OpenVocabConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    cache_dir: Path = Path(".vgc_cache")
    max_drop_rate: float | None = None
    #: Keyframes a track must appear in before it counts as a real object. ByteTrack activates a
    #: track on its first detection, so without this every one-off low-confidence blob becomes a
    #: permanent Entity and the VLM is asked to relate it to things.
    min_track_hits: int = 2

    @property
    def detect_config(self) -> EnsembleConfig | OpenVocabConfig | DetectConfig:
        return {
            "ensemble": self.ensemble,
            "open_vocab": self.open_vocab,
            "yolo": self.detect,
        }[self.detect_backend]


@dataclass(frozen=True)
class QueryResult:
    answer: bool | int | list[dict]
    ir: object
    cypher: str
    rows: list[dict]
    provenance: list[int]
    warnings: list[str]


class Graph:
    def __init__(self, store: GraphStore, schema: SkillSchema, report: Report, vlm_model: str):
        self.store = store
        self.schema = schema
        self._report = report
        self.vlm_model = vlm_model

    def report(self) -> Report:
        return self._report

    def query(self, question: str) -> QueryResult:
        ir = question_to_ir(question, self.schema, model=self.vlm_model)
        trace = execute_ir(ir, self.store)
        return QueryResult(
            answer=trace.answer, ir=ir, cypher="\n".join(trace.cypher),
            rows=trace.rows, provenance=trace.provenance, warnings=trace.warnings,
        )


class VideoPipeline:
    def __init__(self, schema: SkillSchema, config: PipelineConfig | None = None):
        self.schema = schema
        self.config = config or PipelineConfig(schema=schema)
        self.delta_model = make_semantic_delta_model(schema)

    def fit(self, video_path: str | Path) -> Graph:
        video_path = Path(video_path)
        cache_dir = self.config.cache_dir
        video_hash = file_hash(video_path)

        keyframes = DualPhaseFilter(self.config.filter).select(video_path)
        filter_key = stage_key(video_hash, "filter", asdict(self.config.filter))
        # The schema is part of both stages' input — entity_classes are the detector's queries and
        # the relation vocabulary is the VLM's — so it has to be in their keys, or two schemas over
        # one video collide and silently serve each other's artifacts.
        detect_cfg = {
            "backend": self.config.detect_backend,
            **asdict(self.config.detect_config),
            "entity_classes": list(self.schema.entity_classes),
        }
        detect_key = stage_key(video_hash, "detect", detect_cfg, filter_key)
        extract_cfg = {
            **asdict(self.config.extract),
            "relations": sorted(self.schema.instant_relation_names),
        }
        extract_key = stage_key(video_hash, "extract", extract_cfg, detect_key)

        tracked = self._detect(keyframes, detect_key)
        events, drops, frames_attempted, attempts_by_class, operand_violations = self._extract(
            keyframes, tracked, extract_key
        )

        if self.config.max_drop_rate is not None and frames_attempted:
            drop_rate = len(drops) / frames_attempted
            if drop_rate > self.config.max_drop_rate:
                raise RuntimeError(f"drop rate {drop_rate:.1%} exceeds max_drop_rate={self.config.max_drop_rate:.1%}")

        states = StateReducer(self.schema).reduce(events)

        entities = _entities_from_tracks(tracked)
        # Keyed by the full stage chain, not just the video: KuzuStore.write MERGEs, so a graph
        # dir shared across configs accumulates events from runs whose settings were discarded.
        graph_dir = cache_dir / "graphs" / extract_key
        store = KuzuStore(graph_dir / "graph.kuzu")  # kuzu db is a file, graph_dir also holds the manifest
        store.init_schema(self.schema)
        store.write(entities, states, events)

        report = build_report(
            entities, states, events, drops, frames_attempted, attempts_by_class, operand_violations
        )
        _write_run_manifest(graph_dir, report, drops, video_path)
        return Graph(store, self.schema, report, self.config.extract.model)

    def _detect(self, keyframes, key: str) -> list[TrackedBox]:
        if artifacts.exists(self.config.cache_dir, key):
            cached = [TrackedBox(**row) for row in artifacts.load(self.config.cache_dir, key)]
            return _confirmed_tracks(cached, self.config.min_track_hits)
        if self.config.detect_backend == "ensemble":
            tracked = track_keyframes_ensemble(
                keyframes, self.config.ensemble, self.schema.entity_classes,
            )
        elif self.config.detect_backend == "open_vocab":
            tracked = track_keyframes_open_vocab(
                keyframes, self.config.open_vocab, self.schema.entity_classes,
            )
        else:
            tracked = track_keyframes(keyframes, self.config.detect)
        artifacts.save(self.config.cache_dir, key, [asdict(t) for t in tracked])
        return _confirmed_tracks(tracked, self.config.min_track_hits)

    def _extract(self, keyframes, tracked: list[TrackedBox], key: str):
        if artifacts.exists(self.config.cache_dir, key):
            cached = artifacts.load(self.config.cache_dir, key)
            events = [TemporalEdge(**e) for e in cached["events"]]
            drops = [ExtractionDrop(**d) for d in cached["drops"]]
            return (events, drops, cached["frames_attempted"], cached["attempts_by_class"],
                    cached.get("operand_violations", {}))

        by_frame: dict[int, list[TrackedBox]] = {}
        for t in tracked:
            by_frame.setdefault(t.frame_idx, []).append(t)
        images_by_frame = {kf.frame_idx: kf.image for kf in keyframes.frames}

        backend = OpenRouterBackend(model=self.config.extract.model)
        events: list[TemporalEdge] = []
        drops: list[ExtractionDrop] = []
        attempts_by_class: Counter[str] = Counter()
        operand_violations: Counter[str] = Counter()
        frames_attempted = 0

        for frame_idx, boxes in sorted(by_frame.items()):
            entities = [Entity(local_id=b.local_id, class_label=b.class_label, bbox=b.bbox) for b in boxes]
            if not entities:
                continue
            frames_attempted += 1
            for e in entities:
                attempts_by_class[e.class_label] += 1

            timestamp = boxes[0].timestamp
            annotated = render(images_by_frame[frame_idx], entities)
            prompt = som_prompt(entities, self.schema)
            delta, drop = extract_with_retry(
                backend, annotated, self.delta_model, prompt, timestamp,
                [e.class_label for e in entities], self.config.extract.max_retries,
            )
            if drop is not None:
                drops.append(drop)
                continue
            class_by_id = {e.local_id: e.class_label for e in entities}
            for rel in delta.relationships:
                subj, obj = class_by_id.get(rel.source_id), class_by_id.get(rel.target_id)
                if subj is None or obj is None or not self.schema.operands_conform(
                    rel.relation, subj, obj
                ):
                    operand_violations[rel.relation] += 1
                    continue
                # timestamp comes from the keyframe, never from the model (see VLMRelation)
                events.append(TemporalEdge(
                    source_id=rel.source_id, target_id=rel.target_id, relation=rel.relation,
                    timestamp=timestamp, src_frame=frame_idx,
                ))

        artifacts.save(self.config.cache_dir, key, {
            "events": [e.model_dump() for e in events],
            "drops": [d.model_dump() for d in drops],
            "frames_attempted": frames_attempted,
            "attempts_by_class": dict(attempts_by_class),
            "operand_violations": dict(operand_violations),
        })
        return events, drops, frames_attempted, dict(attempts_by_class), dict(operand_violations)


def _write_run_manifest(
    graph_dir: Path, report: Report, drops: list[ExtractionDrop], video_path: Path,
) -> None:
    """SPEC §4.3/§7.3: dropped frames must be inspectable, not buried in the stage cache.

    The per-class breakdown is the point — a uniform 3% drop is noise, while 60% of frames
    containing one class failing is a schema bug an aggregate number would hide (D15).
    """
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "run_manifest.json").write_text(json.dumps({
        "video": str(video_path),
        "frames_attempted": report.frames_attempted,
        "drop_count": report.drop_count,
        "drop_rate": report.drop_rate,
        "drop_rate_by_class": report.drop_rate_by_class,
        "reducer_flag_counts": report.reducer_flag_counts,
        "operand_violations": report.operand_violations,
        "entity_count": report.entity_count,
        "state_count": report.state_count,
        "event_count": report.event_count,
        "drops": [d.model_dump() for d in drops],
    }, indent=2))


def _confirmed_tracks(tracked: list[TrackedBox], min_hits: int) -> list[TrackedBox]:
    """Drop tracks seen in fewer than min_hits keyframes.

    Applied after the detect cache, not before it, so raising min_track_hits does not force a
    re-detection of footage already processed.
    """
    if min_hits <= 1:
        return tracked
    hits: Counter[str] = Counter(t.local_id for t in tracked)
    return [t for t in tracked if hits[t.local_id] >= min_hits]


def _entities_from_tracks(tracked: list[TrackedBox]) -> list[GraphEntity]:
    by_id: dict[str, list[TrackedBox]] = {}
    for t in tracked:
        by_id.setdefault(t.local_id, []).append(t)
    return [
        GraphEntity(
            id=local_id, class_label=boxes[0].class_label,
            first_seen=min(b.timestamp for b in boxes), last_seen=max(b.timestamp for b in boxes),
        )
        for local_id, boxes in by_id.items()
    ]
