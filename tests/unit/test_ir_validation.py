"""A2 regression: out-of-algebra questions fail loudly at validation, never at execution
and never as a plausible wrong answer (SPEC §6.5)."""

import pytest
from pydantic import ValidationError

from core.retriever.ir import (
    And, Count, EventAt, HoldsAt, IRNode, Overlaps, PathBetween, Precedes,
    SchemaValidationError, UnsupportedQueryError, validate_ir,
)
from core.schema.builtin.warehouse import WAREHOUSE


def test_overlaps_rejected_at_validation():
    ir = Overlaps(a=EventAt(relation="DROPS"), b=EventAt(relation="EXITS"))
    with pytest.raises(UnsupportedQueryError, match="Overlaps"):
        validate_ir(ir, WAREHOUSE)


def test_path_between_multi_hop_rejected_not_silently_downgraded():
    """The dangerous case: max_hops=3 previously ran a 1-hop query and returned a confident
    wrong answer with no signal at all."""
    with pytest.raises(UnsupportedQueryError, match="max_hops"):
        validate_ir(PathBetween(source="t1", target="t2", max_hops=3), WAREHOUSE)


def test_path_between_single_hop_allowed():
    validate_ir(PathBetween(source="t1", target="t2", max_hops=1), WAREHOUSE)


def test_count_over_combinator_is_unrepresentable():
    """A tree the NL layer can legitimately emit; it used to crash inside the compiler.
    Count.inner is now typed to the leaf operators, so this fails at construction —
    earlier than validation, and equally loud."""
    with pytest.raises(ValidationError):
        Count(inner=And(children=[EventAt(relation="DROPS"), EventAt(relation="EXITS")]))


def test_count_over_leaf_allowed():
    validate_ir(Count(inner=EventAt(relation="PICKS_UP")), WAREHOUSE)


def test_precedes_over_non_event_operands_is_unrepresentable():
    with pytest.raises(ValidationError):
        Precedes(a=HoldsAt(relation="POSSESSES", t=1.0), b=EventAt(relation="EXITS"))


def test_unsupported_check_recurses_into_children():
    """An unsupported node nested inside a combinator must still be caught."""
    ir = And(children=[EventAt(relation="DROPS"), PathBetween(source="a", target="b", max_hops=2)])
    with pytest.raises(UnsupportedQueryError):
        validate_ir(ir, WAREHOUSE)


def test_vocabulary_check_still_enforced():
    with pytest.raises(SchemaValidationError, match="TELEPORTS"):
        validate_ir(EventAt(relation="TELEPORTS"), WAREHOUSE)


def test_ir_json_schema_has_no_required_field_ref_loop():
    """Structured-output providers reject a reference loop made only of required fields.
    Recursion is allowed only through And/Or/Not's arrays — see the LeafNode comment in ir.py."""
    from pydantic import BaseModel

    class _Req(BaseModel):
        query: IRNode

    defs = _Req.model_json_schema()["$defs"]
    for name, body in defs.items():
        for field, spec in body.get("properties", {}).items():
            if field in ("op", "children"):
                continue
            assert "oneOf" not in spec or all(
                "/$defs/And" not in o.get("$ref", "") for o in spec["oneOf"]
            ), f"{name}.{field} recurses into a combinator through a required field"
