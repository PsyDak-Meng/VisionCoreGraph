"""Shared detect/track output type (SPEC §4.2, D4).

Two detector implementations exist: closed-vocab YOLO+ByteTrack (tracker.py) and
open-vocab OWLv2+ByteTrack (open_vocab.py, pulling D4's runner-up forward). Both
produce TrackedBox rows so the rest of the pipeline is detector-agnostic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedBox:
    frame_idx: int
    timestamp: float
    local_id: str
    class_label: str
    bbox: list[int]
    conf: float
