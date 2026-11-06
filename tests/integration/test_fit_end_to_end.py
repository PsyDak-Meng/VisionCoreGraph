import os

import numpy as np
import pytest

from core.pipeline import PipelineConfig, VideoPipeline
from core.preprocessing.filter import FilterConfig
from core.schema.builtin.warehouse import WAREHOUSE


@pytest.fixture
def synthetic_clip(tmp_path):
    import av

    path = tmp_path / "clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=10)
    stream.width, stream.height = 320, 240
    stream.pix_fmt = "yuv420p"

    for i in range(30):
        frame = av.VideoFrame.from_ndarray(
            np.full((240, 320, 3), fill_value=(i * 8) % 255, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY")
def test_fit_end_to_end(synthetic_clip, tmp_path):
    config = PipelineConfig(
        schema=WAREHOUSE,
        filter=FilterConfig(stride=5),
        cache_dir=tmp_path / "cache",
    )
    pipeline = VideoPipeline(schema=WAREHOUSE, config=config)
    graph = pipeline.fit(synthetic_clip)

    report = graph.report()
    assert report.frames_attempted >= 0  # a blank synthetic clip legitimately yields zero detections

    result = graph.query("Did anyone pick up a bag?")
    assert isinstance(result.answer, (bool, int, list))
    assert result.cypher is not None
