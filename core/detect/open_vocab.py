"""Open-vocab detection: OWLv2 text-prompted boxes + a detector-agnostic ByteTrack (D4 runner-up,
pulled forward). Detected classes are exactly the active schema's entity_classes — no hand-maintained
COCO label_set/class_map to keep in sync, and classes the schema needs (e.g. bag, door) that stock
COCO detectors can't see at all are addressable here since the query is the class name itself.

Confidence scores from zero-shot OWLv2 run much lower than a trained closed-vocab detector's — a
score of 0.1-0.2 here is a normal confident detection, not noise (verified empirically, not COCO's
usual 0.25 default).
"""

from __future__ import annotations

from dataclasses import dataclass

import supervision as sv
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from core.detect.base import TrackedBox
from core.preprocessing.filter import KeyframeSet


@dataclass(frozen=True)
class OpenVocabConfig:
    model: str = "google/owlv2-base-patch16-ensemble"
    score_threshold: float = 0.1
    track_buffer: int = 30


def track_keyframes_open_vocab(
    keyframes: KeyframeSet, config: OpenVocabConfig, entity_classes: tuple[str, ...],
) -> list[TrackedBox]:
    processor = Owlv2Processor.from_pretrained(config.model)
    model = Owlv2ForObjectDetection.from_pretrained(config.model)
    model.eval()
    tracker = sv.ByteTrack(lost_track_buffer=config.track_buffer)

    boxes: list[TrackedBox] = []
    for kf in keyframes.frames:
        inputs = processor(text=[list(entity_classes)], images=kf.image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([kf.image.size[::-1]])
        result = processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=config.score_threshold,
        )[0]

        detections = (
            sv.Detections(
                xyxy=result["boxes"].numpy(),
                confidence=result["scores"].numpy(),
                class_id=result["labels"].numpy(),
            )
            if len(result["boxes"])
            else sv.Detections.empty()
        )
        tracked = tracker.update_with_detections(detections)

        for xyxy, conf, class_id, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id,
        ):
            boxes.append(TrackedBox(
                frame_idx=kf.frame_idx, timestamp=kf.timestamp, local_id=f"t{int(track_id)}",
                class_label=entity_classes[int(class_id)], bbox=[int(v) for v in xyxy], conf=float(conf),
            ))
    return boxes
