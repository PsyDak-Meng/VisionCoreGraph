"""NL question -> IR tree, the one non-deterministic step in retrieval (§6.1)."""

from pydantic import BaseModel

from core.extractors.openrouter import chat_json
from core.retriever.ir import IRNode, validate_ir
from core.schema.types import SkillSchema


class _IRRequest(BaseModel):
    query: IRNode


def question_to_ir(question: str, schema: SkillSchema, model: str = "google/gemini-2.5-flash") -> IRNode:
    prompt = (
        "Convert this question into the query IR.\n"
        f"Entity classes: {', '.join(schema.entity_classes)}\n"
        f"Relations: {', '.join(sorted(schema.relation_names))}\n"
        f"Question: {question}"
    )
    raw = chat_json(prompt, _IRRequest, model=model)
    parsed = _IRRequest.model_validate_json(raw)
    validate_ir(parsed.query, schema)  # invalid type name -> loud error, never an empty result
    return parsed.query
