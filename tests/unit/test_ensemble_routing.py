"""B2 regression: schema classes must route to a detector that can actually see them.

partition_classes is pure (COCO names are passed in), so this needs no model weights.
"""

from core.detect.ensemble import DEFAULT_COCO_SYNONYMS, EnsembleConfig, partition_classes
from core.schema.builtin.warehouse import WAREHOUSE

# The subset of COCO that matters here; the real set comes from YOLO(...).names at runtime.
COCO = {"person", "backpack", "handbag", "suitcase", "car", "truck", "bus", "chair"}


def test_warehouse_classes_split_across_both_detectors():
    covered, open_vocab = partition_classes(WAREHOUSE.entity_classes, COCO, DEFAULT_COCO_SYNONYMS)
    assert covered["person"] == {"person"}
    # bag has COCO synonyms but no entry in DEFAULT_COCO_SYNONYMS: naming a class is not seeing it.
    # YOLO peaks at 0.17 on the bag in object_transfer.mp4, under its 0.25 gate, while OWLv2 gets
    # 0.37 — so routing bag to COCO hid it from the only detector that finds it.
    assert "bag" not in covered
    # none of these exist in COCO at any confidence — they are why the ensemble exists
    assert set(open_vocab) == {"forklift", "door", "pallet", "bag"}


def test_every_schema_class_is_routed_exactly_once():
    """No class may be dropped, and none may go to both detectors — non-overlap is what
    makes the merge safe without cross-detector NMS."""
    covered, open_vocab = partition_classes(WAREHOUSE.entity_classes, COCO, DEFAULT_COCO_SYNONYMS)
    routed = list(covered) + open_vocab
    assert sorted(routed) == sorted(WAREHOUSE.entity_classes)
    assert len(routed) == len(set(routed))


def test_class_matches_coco_by_its_own_name_without_a_synonym():
    covered, open_vocab = partition_classes(("chair",), COCO, DEFAULT_COCO_SYNONYMS)
    assert covered == {"chair": {"chair"}}
    assert open_vocab == []


def test_empty_coco_sends_everything_to_open_vocab():
    covered, open_vocab = partition_classes(WAREHOUSE.entity_classes, set(), DEFAULT_COCO_SYNONYMS)
    assert covered == {}
    assert set(open_vocab) == set(WAREHOUSE.entity_classes)


def test_tracker_gate_is_below_open_vocab_detection_floor():
    """The regression that broke detection: supervision's 0.25 default silently deleted every
    OWLv2 box (they score 0.10-0.19). The tracker gate must sit below the detector's own."""
    config = EnsembleConfig()
    assert config.track_activation_threshold < config.owlv2_conf


def test_force_open_vocab_overrides_coco_coverage():
    """COCO names backpack and bicycle but cannot see either at surveillance distance
    (YOLO 0.11/0.25 vs OWLv2 0.43/0.68), so both must reach the open-vocab detector."""
    covered, open_vocab = partition_classes(
        ("person", "backpack", "bicycle"), COCO | {"bicycle"}, DEFAULT_COCO_SYNONYMS,
    )
    assert covered == {"person": {"person"}}
    assert set(open_vocab) == {"backpack", "bicycle"}


def test_meva_schema_routes_to_a_detector_that_can_see_each_class():
    from core.schema.builtin.meva import MEVA

    covered, open_vocab = partition_classes(MEVA.entity_classes, COCO, DEFAULT_COCO_SYNONYMS)
    assert covered["person"] == {"person"}
    assert covered["vehicle"] == {"car", "truck", "bus"}
    assert set(open_vocab) == {"bag", "backpack", "door", "trash can"}
    assert sorted(list(covered) + open_vocab) == sorted(MEVA.entity_classes)
