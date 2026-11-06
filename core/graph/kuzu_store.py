"""KuzuStore: embedded default GraphStore (D13)."""

from __future__ import annotations

from pathlib import Path

import kuzu

from core.graph.base import Dialect, GraphEntity
from core.graph.ddl import generate_ddl
from core.reduce.policies import State
from core.schema.types import SkillSchema, TemporalEdge, event_id


class KuzuStore:
    def __init__(self, db_path: str | Path):
        # kuzu creates the db dir itself, but not its parents
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)

    @property
    def dialect(self) -> Dialect:
        return "kuzu"

    def init_schema(self, schema: SkillSchema) -> None:
        for stmt in generate_ddl(schema):
            try:
                self._conn.execute(stmt)
            except RuntimeError:
                pass  # table already exists — init_schema is idempotent

    def write(self, entities: list[GraphEntity], states: list[State], events: list[TemporalEdge]) -> None:
        # State.derived_from holds event ids; provenance (§6.4) wants source frames, so resolve
        # here. The DERIVED_FROM edges below keep the event-level audit trail intact.
        frame_by_event = {event_id(ev): ev.src_frame for ev in events}

        for e in entities:
            self._conn.execute(
                "MERGE (n:Entity {id: $id}) "
                "SET n.class = $class, n.first_seen = $first_seen, n.last_seen = $last_seen",
                {"id": e.id, "class": e.class_label, "first_seen": e.first_seen, "last_seen": e.last_seen},
            )

        for ev in events:
            eid = event_id(ev)
            self._conn.execute(
                "MERGE (n:Event {eid: $eid}) "
                "SET n.type = $type, n.timestamp = $timestamp, n.src_frame = $src_frame",
                {"eid": eid, "type": ev.relation, "timestamp": ev.timestamp, "src_frame": ev.src_frame},
            )
            self._conn.execute(
                "MATCH (s:Entity {id: $sid}), (e:Event {eid: $eid}) MERGE (s)-[:EVENT_SUBJ]->(e)",
                {"sid": ev.source_id, "eid": eid},
            )
            self._conn.execute(
                "MATCH (e:Event {eid: $eid}), (o:Entity {id: $oid}) MERGE (e)-[:EVENT_OBJ]->(o)",
                {"eid": eid, "oid": ev.target_id},
            )

        for st in states:
            self._conn.execute(
                "MERGE (n:State {sid: $sid}) SET n.type = $type, n.valid_from = $valid_from, "
                "n.valid_until = $valid_until, n.confidence = $confidence, "
                "n.src_frames = $src_frames, n.flags = $flags",
                {
                    "sid": st.sid, "type": st.type, "valid_from": st.valid_from,
                    "valid_until": st.valid_until, "confidence": 1.0,
                    "src_frames": ",".join(
                        str(frame_by_event[eid]) for eid in st.derived_from if eid in frame_by_event
                    ),
                    "flags": ",".join(st.flags),
                },
            )
            self._conn.execute(
                "MATCH (s:Entity {id: $sid}), (st:State {sid: $stid}) MERGE (s)-[:SUBJECT_OF]->(st)",
                {"sid": st.subject, "stid": st.sid},
            )
            self._conn.execute(
                "MATCH (st:State {sid: $stid}), (o:Entity {id: $oid}) MERGE (st)-[:OBJECT_OF]->(o)",
                {"stid": st.sid, "oid": st.object},
            )
            for eid in st.derived_from:
                self._conn.execute(
                    "MATCH (st:State {sid: $stid}), (e:Event {eid: $eid}) MERGE (st)-[:DERIVED_FROM]->(e)",
                    {"stid": st.sid, "eid": eid},
                )

    def execute(self, cypher: str, params: dict) -> list[dict]:
        result = self._conn.execute(cypher, params)
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(result.get_column_names(), row)))
        return rows
