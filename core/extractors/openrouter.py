"""OpenRouterBackend: single VLM gateway, schema-as-tool-input, best-effort conformance (D14)."""

from __future__ import annotations

import base64
import io
import os

import requests
from PIL import Image
from pydantic import BaseModel

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _response_format(schema: type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
    }


def _post(payload: dict, api_key: str) -> str:
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class OpenRouterBackend:
    def __init__(self, model: str = "google/gemini-2.5-flash", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]

    def complete(self, image: Image.Image, schema: type[BaseModel], prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_uri(image)}},
                ],
            }],
            "response_format": _response_format(schema),
        }
        return _post(payload, self.api_key)


def chat_json(prompt: str, schema: type[BaseModel], model: str = "google/gemini-2.5-flash",
              api_key: str | None = None) -> str:
    """Text-only structured completion, used by the NL->IR layer (§6.1) — same gateway as extraction (D14)."""
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": _response_format(schema),
    }
    return _post(payload, key)
