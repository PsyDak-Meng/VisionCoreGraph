# VisionCoreGraph — Implementation Notes

Companion to `SPEC.md` (the design) and `README.md` (the quickstart). This document covers **what is
actually built**, the tradeoffs taken during implementation, and how to run it. Where the
implementation departs from `SPEC.md`, that is called out explicitly with the reason.

---

## 1. Architecture as built

```
video.mp4
   │
   ▼  STAGE 1  core/preprocessing/filter.py            (CPU, cheap)
 fixed-stride decode via PyAV  ──►  KeyframeSet
   │                                  SPEC's MV gate + ego-motion are NOT built (deferred, §8)
   ▼  STAGE 2  core/detect/ensemble.py                 (CPU, slow)
 YOLO(COCO classes) ∪ OWLv2(open-vocab classes) ──► per-class ByteTrack ──► TrackedBox[]
   │
   ▼  STAGE 3  core/extractors/                        (network, $$)
 Set-of-Mark overlay ──► OpenRouter ──► Pydantic validate ──► retry ≤N ──► VLMRelation | DROP
   │                                     timestamps stamped from the FRAME, never the model
   ▼  STAGE 4  (SKIPPED — GlobalReID deferred; each track id is its own canonical entity)
   │
   ▼  STAGE 5  core/reduce/reducer.py                  (CPU, trivial)
 sort by timestamp ──► replay opens_on/closes_on ──► State intervals
   │
   ▼  STAGE 6  core/graph/kuzu_store.py                (CPU, trivial)
 generate DDL ──► upsert Entity / State / Event ──► Kuzu db
   │
   ▼  RETRIEVAL  core/retriever/
 NL ──► IR tree (Pydantic) ──► validate ──► compile to Cypher ──► execute ──► QueryResult
```

Stages 2 and 3 are cached on disk (`core/cache/`), keyed by a chained content hash so that changing
an upstream config invalidates everything downstream of it.

### Module map

| Path | Role |
|---|---|
| `core/schema/types.py` | `SkillSchema`, relations, and the VLM-facing extraction contract |
| `core/schema/builtin/warehouse.py` | The one shipped schema |
| `core/preprocessing/filter.py` | Stage 1 — keyframe selection |
| `core/detect/ensemble.py` | Stage 2 — **default** detector (YOLO + OWLv2 + per-class ByteTrack) |
| `core/detect/tracker.py`, `open_vocab.py` | Single-detector backends, kept for comparison |
| `core/extractors/` | Stage 3 — SoM renderer, prompt, retry/drop policy, OpenRouter client |
| `core/reduce/reducer.py` | Stage 5 — events → interval states |
| `core/graph/` | Stage 6 — `GraphStore` protocol, Kuzu impl, DDL |
| `core/retriever/` | IR definition, Kuzu compiler, NL→IR layer |
| `core/cache/` | Content-addressed stage cache |
| `core/pipeline.py` | `VideoPipeline` / `Graph` façade |
| `eval/` | QA harness + metrics |

---

## 2. Key implementation decisions and tradeoffs

### 2.1 Detection is an ensemble, not one model

**Problem.** The WAREHOUSE schema declares `person, bag, forklift, door, pallet`. Measured on the 5
MEVA clips in `eval/qa/clips/`:

| class | YOLOv11n (COCO) | OWLv2 zero-shot |
|---|---|---|
| `person` | found in all 5 clips | **missed in all 5** |
| `door` | **not a COCO class at all** | found in all door clips |

Every WAREHOUSE relation needs a `person` subject *and* a `bag`/`door` object, so neither detector
alone can populate a single relation.

**Measured outcome.** Distinct tracked ids per clip, after the ensemble landed:

| clip | YOLO only | OWLv2 only | **ensemble** |
|---|---|---|---|
| pick_up_object | 6 person | **0 boxes** | door 4, person 3 |
| set_down_object | 2 person | **0 boxes** | door 1, person 1 |
| object_transfer | 14 person | 1 door, 0 person | door 9, person 13 |
| enter_through_structure | 2 person | 2 door, 0 person | door 2, person 1 |
| exit_through_structure | 2 person | 2 door, 0 person | door 2, person 1 |

Caveat on reading that table: the YOLO column was measured at `stride=5` and the other two at
`stride=15`, so the person counts are **not** a like-for-like comparison — fewer keyframes yield
fewer track fragments. What it does establish is directional and sufficient: persons are non-zero in
every clip again, and doors appear in every clip, which no single detector achieved.

**Decision.** Partition the schema's classes: anything with a COCO equivalent goes to YOLO, the
remainder becomes OWLv2 text queries. Partitioning (rather than running both everywhere and merging)
means the two detectors never propose competing boxes for the same class, so no cross-detector NMS
or dedup is required — that invariant is what keeps the merge simple.

**Tradeoff accepted.** Two models load per run and OWLv2 is slow on CPU (a full ViT pass per
keyframe). Mitigated by caching both models at module scope and by the stage cache. Set
`detect_backend="yolo"` for a fast path when a schema's classes happen to be COCO-covered.

### 2.2 One tracker per class

ByteTrack associates on IoU and is **class-agnostic**. A single shared tracker will hand a person's
id to the door box they are standing in front of — exactly the geometry of the enter/exit clips.
`_PerClassTracker` runs one `sv.ByteTrack` per class, and ids are namespaced (`person1`, `door2`) so
they stay globally unique, readable in Set-of-Mark overlays, and meaningful in the graph.

**Tradeoff.** An object that changes class between frames (detector flip-flop) becomes two entities.
Acceptable: the alternative — cross-class identity swaps — corrupts the graph silently, which is
worse than fragmenting it visibly.

### 2.3 Confidence scales are reconciled *before* tracking

This caused a real, measured regression. A trained COCO detector reports ~0.3–0.7 for a confident
box; zero-shot OWLv2 reports ~0.10–0.19 for the same confidence. `supervision.ByteTrack` defaults to
`track_activation_threshold=0.25`, so every OWLv2 detection was dropped before a track could form —
2 of 5 clips produced **zero** boxes.

Each detector is therefore filtered by its **own** threshold (`yolo_conf`, `owlv2_conf`), after which
the tracker's gate is deliberately low (`track_activation_threshold=0.03`) because the detections
reaching it are already vetted. All three live in `EnsembleConfig`, never as literals.

### 2.4 The VLM never supplies a timestamp

The model sees one frame and is never told what time it is. `VLMRelation` therefore has **no
timestamp field at all** — absent-from-the-schema is a stronger guarantee than ignoring the value.
`core/pipeline.py` stamps each `TemporalEdge` with the keyframe's real presentation time.

This mirrors SPEC D6's reasoning (the VLM must not emit `valid_from`/`valid_until` because it would
fabricate them) and extends it to event times, which the reducer uses for every interval boundary.

### 2.5 Reified interval nodes, and where provenance lives

States are nodes, not edge properties (SPEC D10). `State.derived_from` holds the ids of the events
that opened and closed the interval. Two consumers need different things from that, so:

- `DERIVED_FROM` edges keep the **event-level** audit trail in the graph.
- `State.src_frames` stores the resolved **frame indices**, because `QueryResult.provenance` is
  specified in frames (§6.4). Resolution happens once at write time in `KuzuStore.write`.

### 2.6 Out-of-algebra questions fail loudly

`validate_ir()` rejects, *before any query runs*: out-of-vocabulary relation names, `Overlaps`,
`PathBetween(max_hops != 1)`, `Count` over a combinator, and `Precedes` over non-`EventAt` operands.

The `PathBetween` case is the important one. It previously accepted `max_hops=3` and silently
executed a 1-hop query — a confident wrong answer with no signal, which SPEC §6.5 names as the worst
failure mode in this system. An empty result set is indistinguishable from a real "no", so the only
safe behaviour is a typed error.

### 2.7 Combinators execute in Python, not Cypher

`And`/`Or`/`Not` run their children and combine booleans in Python rather than compiling to one
compound query. For a single dialect this is far simpler and every child's Cypher and rows still
reach `QueryResult` for audit. Revisit if a second dialect or query-planner pushdown matters.

### 2.8 Cache artifacts are JSON, not Parquet

SPEC §7.3 specifies Parquet for the tabular stages. JSON keeps the dependency list smaller (no
pandas/pyarrow) and these artifacts are small at v0 scale. Revisit when artifact size bites.

---

## 3. Known limitations

| Limitation | Consequence | Where |
|---|---|---|
| **No GlobalReID** | Each track id is its own entity. A person who leaves and re-enters frame becomes two entities, so duration/history do not accumulate. | `cluster/*` empty |
| **Whole video decoded into RAM** | ~6MB per 1080p keyframe. Fine for short clips, a wall for long video. | `preprocessing/filter.py` |
| **Stage 1 is not cached** | A fully-cached re-run still re-decodes the video. | `pipeline.py` |
| **No MV filter / ego-motion** | Fixed-stride sampling is correct but wasteful. | `motion_vectors.py`, `ego_motion.py` empty |
| **`Overlaps`, multi-hop `PathBetween`** | Not implemented — raise `UnsupportedQueryError`. | `retriever/ir.py` |
| **OWLv2 is slow on CPU** | Raise `FilterConfig.stride` on longer clips. | `detect/ensemble.py` |
| **One hardcoded schema** | No `schema scaffold` CLI yet. | `schema/scaffold.py` empty |
| **Kuzu only** | Memgraph adapter and dialect not written. | `graph/memgraph_store.py` empty |
| **Extraction quality unmeasured** | The QA set is still placeholder ground truth; no accuracy number should be believed yet. | `eval/qa/qa_set.jsonl` |

---

## 4. How to run

### Install

```bash
conda create -n visioncoregraph python=3.11 -y
conda activate visioncoregraph
pip install -e ".[dev]"

cp .env.example .env          # then fill in OPENROUTER_API_KEY
export OPENROUTER_API_KEY=... # stages 3 and retrieval both need it
```

### Build a graph and query it

```bash
python scripts/run_demo.py eval/qa/clips/object_transfer.mp4 "Did anyone pick up a bag?"
```

Or from Python:

```python
from core import VideoPipeline
from core.schema.builtin.warehouse import WAREHOUSE

graph = VideoPipeline(schema=WAREHOUSE).fit("clip.mp4")
print(graph.report())                       # counts, drop rate, reducer flags

r = graph.query("Did person1 drop the bag before leaving?")
print(r.answer, r.ir, r.cypher, r.provenance)
```

`graph.report()` is the first thing to read: if `frames_attempted` is 0, the detector found nothing
and everything downstream is empty for that reason. A `run_manifest.json` is written next to the
graph under `.vgc_cache/graphs/<video_hash>/` with the per-class drop breakdown.

### Tuning knobs

```python
from core.pipeline import PipelineConfig
from core.preprocessing.filter import FilterConfig
from core.detect.ensemble import EnsembleConfig

config = PipelineConfig(
    schema=WAREHOUSE,
    filter=FilterConfig(stride=15),                 # ↑ for longer clips (fewer VLM calls, less $)
    ensemble=EnsembleConfig(owlv2_conf=0.08),       # ↓ if open-vocab classes are being missed
    detect_backend="ensemble",                      # "yolo" for a fast COCO-only path
    max_drop_rate=0.5,                              # abort the run if extraction is failing badly
)
```

### Tests

```bash
pytest tests/unit -v          # no network, no API key required
pytest tests/integration -v   # synthetic clip end-to-end; skips without OPENROUTER_API_KEY
```

### Evaluation

```bash
python -m eval.qa_harness     # reports ir_exact_match_rate + answer_accuracy
```

The QA clips are public MEVA surveillance footage, gitignored. Re-fetch with:

```bash
aws s3 cp --no-sign-request \
  s3://mevadata-public-01/examples/videos/ex036-object-transfer.mp4 \
  eval/qa/clips/object_transfer.mp4
```

### Cache behaviour

Artifacts live under `.vgc_cache/`. The key chains upstream stages, so:

| change | re-runs | cost |
|---|---|---|
| reducer rule / schema relation | stages 5–6 | free |
| VLM prompt or model | stages 3–6 | **full $$** |
| detector config or backend | stages 2–6 | **full $$** |
| filter stride | stages 1–6 | **full $$** |

Tune cheapest-first. Delete `.vgc_cache/` to force a clean run.

---

## 5. Optional upgrade: RelateAnything as a stage-3 backend

Not built. Recorded here because the fit is close enough that it should be evaluated before
stage 3 is tuned any further.

[RelateAnything](https://arxiv.org/abs/2609.12552) (53M params, Apache-2.0, weights on HF) takes
**an image, a set of boxes, and a predicate vocabulary supplied at inference as strings**, and
returns a score for every (box_i, box_j, predicate) triple. That is almost exactly the stage-3
contract this pipeline already enforces by prompt discipline.

```
current   boxes ─► SoM overlay ─► OpenRouter VLM ─► JSON ─► validate/retry/drop ─► VLMRelation
                                  network, $ per keyframe, no confidence score

proposed  boxes ─► RelateAnything(schema.relation_names) ─► scores ─► threshold ─► VLMRelation
                                  local, deterministic, scored
```

### Why it is worth evaluating

- **D4 becomes structural.** The model cannot emit entities, coordinates, or timestamps — it only
  scores pairs of boxes the detector already owns. Today that guarantee rests on the prompt plus
  `VLMRelation` having no timestamp field; there it would be a property of the architecture.
- **Relations gain a confidence score**, which the pipeline currently has none of. This matters more
  than the cost saving: drop/keep becomes a tunable threshold measurable against the QA set,
  instead of the all-or-nothing retry/drop policy in `extractors/base.py`.
- **Deterministic and local.** Same input, same output — satisfies the repo's determinism and
  idempotency rules, and removes the one stage that costs money per keyframe.
- **Open vocabulary at inference.** The predicate list is `schema.relation_names` verbatim, so a new
  `SkillSchema` needs no retraining — the same property that makes the detector ensemble work.

### Why it is not a drop-in

1. **Interface mismatch.** `VLMBackend.complete() -> str` is text-shaped; RelateAnything returns a
   score matrix. It needs a sibling path producing `SemanticDelta` directly, not another
   `VLMBackend` implementation. `extract_with_retry` would not apply — the model cannot produce
   structurally invalid output, so there is nothing to retry.
2. **Spatial predicates are weak.** The authors report `above`, `left of`, `next to` performing at
   chance. `ENTERS`/`EXITS`/`INSIDE` are spatial in WAREHOUSE and are the likeliest to degrade.
3. **Single-frame and stative.** `PICKS_UP` vs `DROPS` is a *change* between frames; one frame
   cannot distinguish them. The current VLM path has the same defect, so this is not a regression —
   but it is not the fix either.
4. **Domain shift.** The authors state in-domain measurement overstates transfer ~5x. Fixed-camera
   surveillance is out of domain for RA-4M.
5. A third model to load, after YOLO and OWLv2. 20 ms/frame is quoted on an A40; expect an order of
   magnitude more on CPU — still well under a network round trip.

### The inversion this suggests

Point 3 is the interesting one. RelateAnything predicts **stative** relations well and instants
badly, while `StateReducer` currently runs *instants → intervals*. Scoring `POSSESSES`/`INSIDE`
directly per keyframe and deriving `PICKS_UP`/`DROPS` from the **state change between consecutive
keyframes** inverts that: transitions become a consequence of the state track rather than an
independent per-frame guess, and `_resolve_ties()` has far less to reconcile.

That is a change to the reducer's direction, not a model swap, and it is the part worth thinking
about before either is built.

### Cheapest decisive experiment

Run RelateAnything over the existing ensemble boxes on the 5 MEVA clips, dump the score matrix for
the WAREHOUSE predicates, and check whether `POSSESSES` separates from background. One new file, no
pipeline changes. It is blocked on the same thing as everything else in §3: `eval/qa/qa_set.jsonl`
still holds placeholder ground truth, so "separates" currently has to be judged by eye.
