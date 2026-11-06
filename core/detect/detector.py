"""DetectConfig (§4.2). Detector and tracker own boxes and local_id; the VLM never does (D4).

label_set/class_map default to what stock COCO-trained YOLO can actually see, remapped onto the
WAREHOUSE schema's vocabulary (person, bag). forklift/door/pallet have no COCO equivalent and are
undetected by the default weights — a real instance of D4's "fixed label_set is required" cost, not a
bug. A custom-trained or open-vocab detector is needed to cover the rest of WAREHOUSE's entity classes.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectConfig:
    detector: str = "yolo11n.pt"
    label_set: tuple[str, ...] = ("person", "backpack", "handbag", "suitcase")
    class_map: dict[str, str] = field(default_factory=lambda: {
        "backpack": "bag", "handbag": "bag", "suitcase": "bag",
    })
    conf_threshold: float = 0.25
    track_buffer: int = 30
