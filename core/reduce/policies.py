"""Edge-case policies for StateReducer, per SPEC §4.6's table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    sid: str
    type: str
    subject: str
    object: str
    valid_from: float
    valid_until: float | None
    flags: tuple[str, ...] = ()
    derived_from: tuple[str, ...] = ()


def state_id(type_: str, subject: str, object_: str, valid_from: float) -> str:
    return f"{type_}:{subject}:{object_}:{valid_from}"
