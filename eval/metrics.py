"""IR tree exact-match and end-to-end answer accuracy (SPEC §9.2)."""

from pydantic import TypeAdapter

from core.retriever.ir import IRNode

_ir_adapter = TypeAdapter(IRNode)


def parse_ir(data: dict):
    return _ir_adapter.validate_python(data)


def ir_exact_match(predicted, expected: dict) -> bool:
    """Structural equality: predicted IR tree == the annotated expected tree."""
    return predicted.model_dump() == parse_ir(expected).model_dump()


def answer_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["actual_answer"] == r["expected_answer"])
    return correct / len(results)
