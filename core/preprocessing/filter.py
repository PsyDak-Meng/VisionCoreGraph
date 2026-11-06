"""DualPhaseFilter (§4.1). v0 implements only fixed-stride sampling per SPEC §8;
the MV coarse gate and ego-motion compensation are deferred (motion_vectors.py, ego_motion.py stay empty)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


@dataclass(frozen=True)
class FilterConfig:
    stride: int = 15  # keep every Nth decoded frame


@dataclass(frozen=True)
class KeyFrame:
    frame_idx: int
    timestamp: float
    image: Image.Image


@dataclass(frozen=True)
class KeyframeSet:
    frames: list[KeyFrame]
    fps: float


class DualPhaseFilter:
    def __init__(self, config: FilterConfig):
        self.config = config

    def select(self, video_path: str | Path) -> KeyframeSet:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = []
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx % self.config.stride != 0:
                continue
            frames.append(KeyFrame(frame_idx=frame_idx, timestamp=frame_idx / fps, image=frame.to_image()))
        container.close()
        return KeyframeSet(frames=frames, fps=fps)
