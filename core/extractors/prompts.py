"""Set-of-Mark prompt template, parameterized by schema vocabulary (§4.3)."""

from core.schema.types import Entity, SkillSchema


def som_prompt(entities: list[Entity], schema: SkillSchema) -> str:
    """Relations only: no ids, no coordinates, no timing. The caller owns all three (D4, D12)."""
    entity_list = "; ".join(f"{e.local_id} is {e.class_label}" for e in entities)
    relations = ", ".join(sorted(schema.instant_relation_names))
    return (
        f"Numbered boxes mark entities in this single frame: {entity_list}.\n"
        f"Report only relationships between these numbered ids, using exactly one of: {relations}.\n"
        "These are momentary actions: report one only if it is happening in this frame, not "
        "an ongoing situation that merely holds.\n"
        "Report only what is visible in this frame. Do not invent new entities or ids, "
        "do not report coordinates, and do not report times or durations."
    )
