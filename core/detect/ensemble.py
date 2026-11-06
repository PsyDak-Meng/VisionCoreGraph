"""Ensemble detection: YOLO for COCO-covered classes, OWLv2 for the rest, one tracker over both.

Why an ensemble rather than either detector alone — measured on the 5 MEVA clips in eval/qa/clips:

| class  | YOLO/COCO                   | OWLv2 zero-shot            |
|--------|-----------------------------|----------------------------|
| person | found in all 5 clips        | missed in all 5            |
| door   | not a COCO class            | found in all door clips    |
| bag    | synonyms peak 0.17, < gate  | 0.37 on object_transfer    |

Neither covers the WAREHOUSE schema on its own, so each class is routed to whichever detector can
actually see it (D4's runner-up, pulled forward). Classes are partitioned, never duplicated, so the
two detectors cannot propose competing boxes for the same class and no cross-detector NMS is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
import torch
from PIL import Image
from ultralytics import YOLO

from core.detect.base import TrackedBox
from core.preprocessing.filter import KeyframeSet

#: Schema class -> COCO names that stand in for it. Anything absent here is matched against COCO
#: by its own name, and falls through to OWLv2 when COCO has no equivalent.
#:
#: `bag` deliberately has no entry. COCO nominally covers it via backpack/handbag/suitcase, and
#: because the partition is exclusive that routing was enough to hide the class from OWLv2
#: entirely. Measured on object_transfer.mp4: YOLO's best bag-synonym score is 0.173, under the
#: 0.25 yolo_conf gate, so nothing survived; OWLv2 scores the same object 0.373. Listing a synonym
#: here claims COCO can see the class, not merely that it has a name for it.
DEFAULT_COCO_SYNONYMS: dict[str, tuple[str, ...]] = {
    "vehicle": ("car", "truck", "bus"),
}

#: Classes COCO has a name for but cannot actually see in this footage, forced to OWLv2.
#: The synonyms table above can only add COCO coverage; this is how a class opts out of it.
#: Best scores measured over the 5 MEVA clips (scripts are in the session log):
#:   backpack  YOLO 0.11  OWLv2 0.43
#:   bicycle   YOLO 0.25  OWLv2 0.68
#: Both are surveillance-distance objects, which is the regime trained COCO detectors lose.
DEFAULT_FORCE_OPEN_VOCAB: tuple[str, ...] = ("backpack", "bicycle")


@dataclass(frozen=True)
class EnsembleConfig:
    yolo_model: str = "yolo11n.pt"
    owlv2_model: str = "google/owlv2-base-patch16-ensemble"
    # Separate thresholds because the score scales are not comparable: a trained COCO detector
    # reports ~0.3-0.7 for a confident box, zero-shot OWLv2 reports ~0.10-0.19 for the same
    # confidence. One shared threshold silently deletes every open-vocab detection.
    yolo_conf: float = 0.25
    owlv2_conf: float = 0.08
    #: Detections are already vetted by the per-detector thresholds above, so the tracker's own
    #: gate stays low. Leaving supervision's 0.25 default here is what made OWLv2 boxes vanish.
    track_activation_threshold: float = 0.03
    minimum_matching_threshold: float = 0.8
    track_buffer: int = 30
    coco_synonyms: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_COCO_SYNONYMS)
    )
    force_open_vocab: tuple[str, ...] = DEFAULT_FORCE_OPEN_VOCAB


_YOLO_CACHE: dict[str, YOLO] = {}
_OWLV2_CACHE: dict[str, tuple] = {}


def _load_yolo(name: str) -> YOLO:
    if name not in _YOLO_CACHE:
        _YOLO_CACHE[name] = YOLO(name)
    return _YOLO_CACHE[name]


def _load_owlv2(name: str):
    """Cached: the QA harness runs several clips per process and the weights are ~600MB."""
    if name not in _OWLV2_CACHE:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        processor = Owlv2Processor.from_pretrained(name)
        model = Owlv2ForObjectDetection.from_pretrained(name)
        model.eval()
        _OWLV2_CACHE[name] = (processor, model)
    return _OWLV2_CACHE[name]


def partition_classes(
    entity_classes: tuple[str, ...], yolo_names: set[str], synonyms: dict[str, tuple[str, ...]],
    force_open: tuple[str, ...] = DEFAULT_FORCE_OPEN_VOCAB,
) -> tuple[dict[str, set[str]], list[str]]:
    """Split schema classes into COCO-detectable and open-vocab-only.

    Returns (schema_class -> COCO names that map to it, [classes needing OWLv2]).
    """
    coco_covered: dict[str, set[str]] = {}
    open_vocab: list[str] = []
    for cls in entity_classes:
        present = set() if cls in force_open else {
            c for c in synonyms.get(cls, (cls,)) if c in yolo_names
        }
        if present:
            coco_covered[cls] = present
        else:
            open_vocab.append(cls)
    return coco_covered, open_vocab


def _detect_yolo(image: Image.Image, model: YOLO, coco_to_schema: dict[str, str], conf: float):
    result = model.predict(source=np.array(image), conf=conf, verbose=False)[0]
    boxes, labels, scores = [], [], []
    if result.boxes is not None:
        for xyxy, cls, score in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist(),
        ):
            schema_class = coco_to_schema.get(result.names[int(cls)])
            if schema_class is not None:
                boxes.append(xyxy)
                labels.append(schema_class)
                scores.append(float(score))
    return boxes, labels, scores


def _detect_owlv2(image: Image.Image, processor, model, queries: list[str], threshold: float):
    inputs = processor(text=[queries], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=threshold,
    )[0]
    boxes = [b.tolist() for b in result["boxes"]]
    labels = [queries[int(i)] for i in result["labels"]]
    scores = [float(s) for s in result["scores"]]
    return boxes, labels, scores


class _PerClassTracker:
    """One ByteTrack per class.

    ByteTrack associates on IoU and is class-agnostic, so a single shared tracker will happily
    hand a person's id to the door box they are standing in front of — which is precisely the
    geometry of the enter/exit clips. Tracking each class separately makes that impossible.
    Track ids are namespaced by class (`person1`, `door2`) so they stay globally unique and
    stay readable in Set-of-Mark overlays and in the graph.
    """

    def __init__(self, config: EnsembleConfig):
        self._config = config
        self._trackers: dict[str, sv.ByteTrack] = {}

    def _tracker(self, class_label: str) -> sv.ByteTrack:
        if class_label not in self._trackers:
            self._trackers[class_label] = sv.ByteTrack(
                track_activation_threshold=self._config.track_activation_threshold,
                minimum_matching_threshold=self._config.minimum_matching_threshold,
                lost_track_buffer=self._config.track_buffer,
            )
        return self._trackers[class_label]

    def update(self, class_label: str, boxes: list, scores: list) -> list[tuple[list[float], float, int]]:
        detections = (
            sv.Detections(
                xyxy=np.array(boxes, dtype=float),
                confidence=np.array(scores, dtype=float),
                class_id=np.zeros(len(boxes), dtype=int),
            )
            if boxes
            else sv.Detections.empty()
        )
        tracked = self._tracker(class_label).update_with_detections(detections)
        return list(zip(tracked.xyxy.tolist(), tracked.confidence.tolist(), tracked.tracker_id.tolist()))


def track_keyframes_ensemble(
    keyframes: KeyframeSet, config: EnsembleConfig, entity_classes: tuple[str, ...],
) -> list[TrackedBox]:
    yolo = _load_yolo(config.yolo_model)
    coco_covered, open_vocab = partition_classes(
        entity_classes, set(yolo.names.values()), config.coco_synonyms, config.force_open_vocab,
    )
    coco_to_schema = {coco: schema for schema, cocos in coco_covered.items() for coco in cocos}

    owlv2 = _load_owlv2(config.owlv2_model) if open_vocab else None
    tracker = _PerClassTracker(config)
    out: list[TrackedBox] = []

    for kf in keyframes.frames:
        boxes: list[list[float]] = []
        labels: list[str] = []
        scores: list[float] = []

        if coco_to_schema:
            b, lb, s = _detect_yolo(kf.image, yolo, coco_to_schema, config.yolo_conf)
            boxes.extend(b)
            labels.extend(lb)
            scores.extend(s)

        if open_vocab and owlv2 is not None:
            b, lb, s = _detect_owlv2(kf.image, owlv2[0], owlv2[1], open_vocab, config.owlv2_conf)
            boxes.extend(b)
            labels.extend(lb)
            scores.extend(s)

        by_class: dict[str, tuple[list, list]] = {}
        for box, label, score in zip(boxes, labels, scores):
            bucket = by_class.setdefault(label, ([], []))
            bucket[0].append(box)
            bucket[1].append(score)

        for class_label in set(list(coco_covered) + open_vocab):
            cls_boxes, cls_scores = by_class.get(class_label, ([], []))
            for xyxy, conf, track_id in tracker.update(class_label, cls_boxes, cls_scores):
                out.append(TrackedBox(
                    frame_idx=kf.frame_idx, timestamp=kf.timestamp,
                    local_id=f"{class_label}{int(track_id)}", class_label=class_label,
                    bbox=[int(v) for v in xyxy], conf=float(conf),
                ))
    return out
