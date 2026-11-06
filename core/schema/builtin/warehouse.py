"""The one hardcoded v0 schema (SPEC §4.5)."""

from core.schema.types import DurativeRelation, InstantRelation, SkillSchema

WAREHOUSE = SkillSchema(
    name="warehouse",
    entity_classes=("person", "bag", "forklift", "door", "pallet"),
    instant_relations=(
        InstantRelation("PICKS_UP", subject="person", object=("bag", "pallet")),
        InstantRelation("DROPS", subject="person", object=("bag", "pallet")),
        InstantRelation("HANDS_OFF", subject="person", object=("bag", "pallet")),
        InstantRelation("ENTERS", subject="person", object="door"),
        InstantRelation("EXITS", subject="person", object="door"),
    ),
    durative_relations=(
        DurativeRelation(
            "POSSESSES",
            subject="person",
            object=("bag", "pallet"),
            opens_on=("PICKS_UP",),
            closes_on=("DROPS", "HANDS_OFF"),
            cardinality="one_subject_per_object",
        ),
        DurativeRelation(
            "INSIDE",
            subject="person",
            object="door",
            opens_on=("ENTERS",),
            closes_on=("EXITS",),
            cardinality="unconstrained",
        ),
    ),
)
