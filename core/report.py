"""graph.report() — counts and drop rates, overall and per entity class (D15)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from core.extractors.base import ExtractionDrop
from core.graph.base import GraphEntity
from core.reduce.policies import State
from core.schema.types import TemporalEdge


@dataclass(frozen=True)
class Report:
    entity_count: int
    state_count: int
    event_count: int
    drop_count: int
    frames_attempted: int
    drop_rate: float
    drop_rate_by_class: dict[str, float]
    reducer_flag_counts: dict[str, int]
    #: Events the VLM returned whose operand classes contradict the schema, by relation. Counted
    #: rather than folded into drop_count: a drop is a frame that produced no parseable output,
    #: this is a frame that parsed cleanly and still said something impossible.
    operand_violations: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"entities={self.entity_count} states={self.state_count} events={self.event_count}",
            f"drop rate: {self.drop_rate:.1%} ({self.drop_count}/{self.frames_attempted})",
        ]
        for cls, rate in sorted(self.drop_rate_by_class.items()):
            lines.append(f"  {cls}: {rate:.1%}")
        if self.reducer_flag_counts:
            lines.append(f"reducer flags: {self.reducer_flag_counts}")
        if self.operand_violations:
            total = sum(self.operand_violations.values())
            lines.append(f"schema violations: {total} rejected {self.operand_violations}")
        return "\n".join(lines)


def build_report(
    entities: list[GraphEntity],
    states: list[State],
    events: list[TemporalEdge],
    drops: list[ExtractionDrop],
    frames_attempted: int,
    attempts_by_class: dict[str, int],
    operand_violations: dict[str, int] | None = None,
) -> Report:
    drops_by_class: Counter[str] = Counter()
    for d in drops:
        for cls in d.entity_classes_present:
            drops_by_class[cls] += 1

    drop_rate_by_class = {
        cls: drops_by_class.get(cls, 0) / count for cls, count in attempts_by_class.items() if count
    }

    reducer_flag_counts: Counter[str] = Counter()
    for s in states:
        for f in s.flags:
            reducer_flag_counts[f] += 1

    return Report(
        entity_count=len(entities),
        state_count=len(states),
        event_count=len(events),
        drop_count=len(drops),
        frames_attempted=frames_attempted,
        drop_rate=(len(drops) / frames_attempted) if frames_attempted else 0.0,
        drop_rate_by_class=drop_rate_by_class,
        reducer_flag_counts=dict(reducer_flag_counts),
        operand_violations=dict(operand_violations or {}),
    )
