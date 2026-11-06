"""Closed-vocab detector: ByteTrack via ultralytics' built-in tracker over YOLO/COCO boxes.
local_id is the persisted track id (§4.2). See open_vocab.py for the open-vocab alternative."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from core.detect.base import TrackedBox
from core.detect.detector import DetectConfig
from core.preprocessing.filter import KeyframeSet


def track_keyframes(keyframes: KeyframeSet, config: DetectConfig) -> list[TrackedBox]:
    model = YOLO(config.detector)
    boxes: list[TrackedBox] = []
    for kf in keyframes.frames:
        result = model.track(
            source=np.array(kf.image), tracker="bytetrack.yaml", persist=True,
            conf=config.conf_threshold, verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            continue
        for xyxy, track_id, cls, conf in zip(
            result.boxes.xyxy.tolist(), result.boxes.id.tolist(),
            result.boxes.cls.tolist(), result.boxes.conf.tolist(),
        ):
            label = result.names[int(cls)]
            if config.label_set and label not in config.label_set:
                continue
            boxes.append(TrackedBox(
                frame_idx=kf.frame_idx, timestamp=kf.timestamp, local_id=f"t{int(track_id)}",
                class_label=config.class_map.get(label, label), bbox=[int(v) for v in xyxy], conf=float(conf),
            ))
    return boxes
