"""VLMBackend protocol and the bounded-retry/drop extraction policy (SPEC §4.3, D15)."""

from __future__ import annotations

from typing import Protocol

from PIL import Image
from pydantic import BaseModel, ValidationError


class VLMBackend(Protocol):
    def complete(self, image: Image.Image, schema: type[BaseModel], prompt: str) -> str:
        """Raw text response for one frame; caller validates against `schema`."""
        ...


class ExtractionDrop(BaseModel):
    timestamp: float
    raw_response: str
    error: str
    entity_classes_present: list[str]


def extract_with_retry(
    backend: VLMBackend,
    image: Image.Image,
    delta_model: type[BaseModel],
    prompt: str,
    timestamp: float,
    entity_classes_present: list[str],
    max_retries: int = 2,
) -> tuple[BaseModel | None, ExtractionDrop | None]:
    last_error = ""
    raw = ""
    current_prompt = prompt
    for _ in range(max_retries + 1):
        raw = backend.complete(image, delta_model, current_prompt)
        try:
            return delta_model.model_validate_json(raw), None
        except ValidationError as e:
            last_error = str(e)
            current_prompt = f"{prompt}\n\nPrevious attempt was invalid: {last_error}\nFix and retry."
    return None, ExtractionDrop(
        timestamp=timestamp, raw_response=raw, error=last_error,
        entity_classes_present=entity_classes_present,
    )
