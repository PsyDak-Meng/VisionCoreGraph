"""Temporal query IR: 8 operators + 3 combinators (SPEC §6.2)."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

from core.schema.types import SkillSchema


class HoldsAt(BaseModel):
    op: Literal["holds_at"] = "holds_at"
    relation: str
    subject: str | None = None
    object: str | None = None
    t: float


class HoldsDuring(BaseModel):
    op: Literal["holds_during"] = "holds_during"
    relation: str
    subject: str | None = None
    object: str | None = None
    t0: float
    t1: float
    mode: Literal["all", "any"] = "any"


class EventAt(BaseModel):
    op: Literal["event_at"] = "event_at"
    relation: str
    subject: str | None = None
    object: str | None = None
    t0: float | None = None
    t1: float | None = None


class Precedes(BaseModel):
    op: Literal["precedes"] = "precedes"
    a: EventAt
    b: EventAt


class Overlaps(BaseModel):
    op: Literal["overlaps"] = "overlaps"
    a: EventAt
    b: EventAt


class PathBetween(BaseModel):
    op: Literal["path_between"] = "path_between"
    source: str
    target: str
    max_hops: int = 3
    at: float | None = None


#: Operators that stand alone as a complete query. Count is defined over these, and Precedes
#: over EventAt, because that is the whole of what the Kuzu compiler can lower (§6.5). Typing
#: the operands this narrowly — rather than accepting any IRNode and rejecting later — is also
#: what keeps the emitted JSON schema free of the required-field reference loops that
#: structured-output providers refuse.
LeafNode = Annotated[
    Union[HoldsAt, HoldsDuring, EventAt, Precedes, PathBetween],
    Field(discriminator="op"),
]


class Count(BaseModel):
    op: Literal["count"] = "count"
    inner: LeafNode


class And(BaseModel):
    op: Literal["and"] = "and"
    children: list["IRNode"]


class Or(BaseModel):
    op: Literal["or"] = "or"
    children: list["IRNode"]


class Not(BaseModel):
    op: Literal["not"] = "not"
    #: A one-element list, not a bare field. Structured-output providers accept a reference
    #: loop only through a potentially-zero-length array, so the arity is enforced by a
    #: validator rather than by min_length/max_length, which would emit minItems into the schema.
    children: list["IRNode"]

    @field_validator("children")
    @classmethod
    def _exactly_one_child(cls, v: list) -> list:
        if len(v) != 1:
            raise ValueError(f"not takes exactly one child, got {len(v)}")
        return v

    @property
    def child(self) -> "IRNode":
        return self.children[0]


IRNode = Annotated[
    Union[HoldsAt, HoldsDuring, EventAt, Precedes, Overlaps, Count, PathBetween, And, Or, Not],
    Field(discriminator="op"),
]

for _cls in (And, Or, Not):
    _cls.model_rebuild()


class SchemaValidationError(ValueError):
    pass


class UnsupportedQueryError(NotImplementedError):
    """The question is outside v0's implemented algebra. Raised before anything runs (§6.5)."""


def _check_supported(node: IRNode) -> None:
    if isinstance(node, Overlaps):
        raise UnsupportedQueryError(
            "Overlaps is not implemented in v0. Rephrase using holds_during, or extend the compiler."
        )

    if isinstance(node, PathBetween) and node.max_hops != 1:
        # Silently running a 1-hop query for a multi-hop question would be a confident wrong
        # answer, which §6.5 calls out as the worst failure mode in this system.
        raise UnsupportedQueryError(
            f"path_between supports only max_hops=1 in v0 (got {node.max_hops}); "
            "multi-hop traversal is not implemented."
        )


def validate_ir(node: IRNode, schema: SkillSchema) -> None:
    """Reject out-of-vocabulary names and out-of-algebra questions before any query runs (§6.1, §6.5).

    A loud error here is the whole point: an unvalidated tree either crashes mid-execution or
    returns an empty result set that reads like a real 'no'.
    """
    _check_supported(node)

    relation = getattr(node, "relation", None)
    if relation is not None and relation not in schema.relation_names:
        raise SchemaValidationError(f"relation {relation!r} not in schema {schema.name!r} vocabulary")

    for attr in ("a", "b", "inner"):
        child = getattr(node, attr, None)
        if child is not None:
            validate_ir(child, schema)
    for child in getattr(node, "children", []):
        validate_ir(child, schema)
