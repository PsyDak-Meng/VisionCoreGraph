"""GraphStore protocol (SPEC §5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from core.reduce.policies import State
from core.schema.types import SkillSchema, TemporalEdge

Dialect = Literal["kuzu", "memgraph"]


@dataclass(frozen=True)
class GraphEntity:
    id: str
    class_label: str
    first_seen: float
    last_seen: float


class GraphStore(Protocol):
    def init_schema(self, schema: SkillSchema) -> None: ...

    def write(self, entities: list[GraphEntity], states: list[State], events: list[TemporalEdge]) -> None: ...

    def execute(self, cypher: str, params: dict) -> list[dict]: ...

    @property
    def dialect(self) -> Dialect: ...
