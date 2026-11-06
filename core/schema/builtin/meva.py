"""MEVA surveillance schema, fitted to the clips in eval/qa/clips (SPEC §4.5).

WAREHOUSE was written before there was footage to test against, and two of its five classes —
forklift and pallet — do not occur in MEVA at all. Every class below was chosen from measured
detectability across all five clips rather than from what a warehouse ontology ought to contain:

| class     | best score                      | detector |
|-----------|---------------------------------|----------|
| person    | YOLO 0.93                       | COCO     |
| door      | OWLv2 0.66 (enter/exit clips)   | open     |
| vehicle   | YOLO 0.67 (car) / 0.53 (truck)  | COCO     |
| backpack  | OWLv2 0.43, YOLO only 0.11      | open     |
| bag       | OWLv2 0.37 (object_transfer)    | open     |
| trash can | OWLv2 0.64                      | open     |

A score proves the detector fires, not that it is right: `cup` 0.37, `bottle` 0.39 and
`cell phone` 0.28 also fired on this footage, on objects a few pixels wide at surveillance
distance. Those are confident false positives, and they are exactly why a class belongs here
only after someone has looked at the frames. The classes above are the ones that survived that
check; see the probe output in the session log for the full table.
"""

from core.schema.types import DurativeRelation, InstantRelation, SkillSchema

MEVA = SkillSchema(
    name="meva",
    entity_classes=("person", "bag", "backpack", "door", "vehicle", "trash can"),
    instant_relations=(
        InstantRelation("PICKS_UP", subject="person", object=("bag", "backpack")),
        InstantRelation("DROPS", subject="person", object=("bag", "backpack")),
        InstantRelation("HANDS_OFF", subject="person", object=("bag", "backpack")),
        InstantRelation("DISCARDS", subject="person", object=("bag", "backpack")),
        InstantRelation("ENTERS", subject="person", object="door"),
        InstantRelation("EXITS", subject="person", object="door"),
        InstantRelation("BOARDS", subject="person", object="vehicle"),
        InstantRelation("ALIGHTS", subject="person", object="vehicle"),
    ),
    durative_relations=(
        DurativeRelation(
            "POSSESSES",
            subject="person",
            object=("bag", "backpack"),
            opens_on=("PICKS_UP",),
            closes_on=("DROPS", "HANDS_OFF", "DISCARDS"),
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
        DurativeRelation(
            "IN_VEHICLE",
            subject="person",
            object="vehicle",
            opens_on=("BOARDS",),
            closes_on=("ALIGHTS",),
            cardinality="unconstrained",
        ),
    ),
)
