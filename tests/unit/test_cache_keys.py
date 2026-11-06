from pathlib import Path

from core.cache.keys import stage_key


def test_same_inputs_same_key():
    k1 = stage_key("videohash", "filter", {"stride": 15})
    k2 = stage_key("videohash", "filter", {"stride": 15})
    assert k1 == k2


def test_different_config_different_key():
    k1 = stage_key("videohash", "filter", {"stride": 15})
    k2 = stage_key("videohash", "filter", {"stride": 30})
    assert k1 != k2


def test_upstream_change_cascades():
    upstream_a = stage_key("videohash", "filter", {"stride": 15})
    upstream_b = stage_key("videohash", "filter", {"stride": 30})
    downstream_a = stage_key("videohash", "detect", {"model": "yolo11n"}, upstream_a)
    downstream_b = stage_key("videohash", "detect", {"model": "yolo11n"}, upstream_b)
    assert downstream_a != downstream_b


def test_two_schemas_over_one_video_do_not_share_stage_keys():
    """entity_classes drive the detector's queries and the verbs drive the VLM prompt, so a
    key that omits the schema hands a MEVA run WAREHOUSE's detections."""
    from core.pipeline import PipelineConfig
    from core.schema.builtin.meva import MEVA
    from core.schema.builtin.warehouse import WAREHOUSE
    from scripts.build_viewer import _keys

    video = Path("eval/qa/clips/object_transfer.mp4")
    w = _keys(video, PipelineConfig(schema=WAREHOUSE))
    m = _keys(video, PipelineConfig(schema=MEVA))
    assert w[0] == m[0]        # stage 1 does not depend on the schema
    assert w[1] != m[1]        # detect does
    assert w[2] != m[2]        # extract does
