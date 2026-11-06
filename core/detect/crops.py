"""Crop utility, retained for the deferred Re-ID stage (cluster/*, not wired in v0)."""

from PIL import Image


def crop(image: Image.Image, bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    return image.crop((x0, y0, x1, y1))
