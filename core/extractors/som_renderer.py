"""Set-of-Mark overlay: numbered boxes drawn on the frame (D12)."""

from __future__ import annotations

import colorsys

from PIL import Image, ImageDraw

from core.schema.types import Entity


def _color_for_id(local_id: str) -> tuple[int, int, int]:
    """Deterministic per-id color so the same entity looks the same across frames."""
    hue = (hash(local_id) % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def render(frame: Image.Image, entities: list[Entity]) -> Image.Image:
    annotated = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for e in entities:
        color = _color_for_id(e.local_id)
        x0, y0, x1, y1 = e.bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label_y = y0 - 14 if y0 - 14 > 0 else y1 + 2
        draw.text((x0, label_y), e.local_id, fill=color)
    return annotated
