"""Operand conformance (§6.5) and the track confirmation gate."""

from core.detect.base import TrackedBox
from core.pipeline import _confirmed_tracks
from core.schema.builtin.warehouse import WAREHOUSE


def test_declared_operands_conform():
    assert WAREHOUSE.operands_conform("HANDS_OFF", "person", "bag")
    assert WAREHOUSE.operands_conform("INSIDE", "person", "door")


def test_wrong_object_class_rejected():
    # the exact edge the VLM emitted on object_transfer.mp4 before this check existed
    assert not WAREHOUSE.operands_conform("HANDS_OFF", "person", "person")
    assert not WAREHOUSE.operands_conform("INSIDE", "person", "person")


def test_unknown_relation_rejected():
    assert not WAREHOUSE.operands_conform("TELEPORTS", "person", "door")


def _box(local_id: str, frame_idx: int) -> TrackedBox:
    return TrackedBox(
        frame_idx=frame_idx, timestamp=frame_idx / 25, local_id=local_id,
        class_label="door", bbox=[0, 0, 10, 10], conf=0.2,
    )


def test_single_frame_track_dropped():
    tracked = [_box("door5", 0), _box("door5", 15), _box("door1", 0)]
    kept = {t.local_id for t in _confirmed_tracks(tracked, min_hits=2)}
    assert kept == {"door5"}


def test_gate_disabled_keeps_everything():
    tracked = [_box("door1", 0)]
    assert _confirmed_tracks(tracked, min_hits=1) == tracked


def test_vlm_vocabulary_excludes_durative_relations():
    """The VLM reports transitions; the reducer derives the intervals they open and close."""
    assert WAREHOUSE.instant_relation_names == {
        "PICKS_UP", "DROPS", "HANDS_OFF", "ENTERS", "EXITS",
    }
    assert "POSSESSES" not in WAREHOUSE.instant_relation_names
    assert "INSIDE" not in WAREHOUSE.instant_relation_names
    # queries may still ask about them
    assert "POSSESSES" in WAREHOUSE.relation_names
