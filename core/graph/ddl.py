"""Kuzu DDL, fixed per SPEC §5.3 — the reified-interval-node model (D10)."""

from core.schema.types import SkillSchema

DDL_STATEMENTS = (
    "CREATE NODE TABLE Entity (id STRING, class STRING, first_seen DOUBLE, "
    "last_seen DOUBLE, PRIMARY KEY(id))",
    "CREATE NODE TABLE State (sid STRING, type STRING, valid_from DOUBLE, "
    "valid_until DOUBLE, confidence DOUBLE, src_frames STRING, flags STRING, PRIMARY KEY(sid))",
    "CREATE NODE TABLE Event (eid STRING, type STRING, timestamp DOUBLE, "
    "src_frame INT64, PRIMARY KEY(eid))",
    "CREATE REL TABLE SUBJECT_OF (FROM Entity TO State)",
    "CREATE REL TABLE OBJECT_OF (FROM State TO Entity)",
    "CREATE REL TABLE EVENT_SUBJ (FROM Entity TO Event)",
    "CREATE REL TABLE EVENT_OBJ (FROM Event TO Entity)",
    "CREATE REL TABLE DERIVED_FROM (FROM State TO Event)",
)


def generate_ddl(schema: SkillSchema) -> tuple[str, ...]:
    """Table structure is fixed (D10); the schema only determines valid values, enforced at write time."""
    return DDL_STATEMENTS
