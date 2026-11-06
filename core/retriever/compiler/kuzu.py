"""IR -> Cypher lowering for the Kuzu dialect (SPEC §6.3).

Leaf nodes compile to one parametrized Cypher query each. Combinators
(And/Or/Not) execute their children and combine booleans in Python — for a
single-dialect v0 this is simpler than building compound Cypher, and every
child query's cypher/rows still end up in the final QueryResult for audit.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.graph.base import GraphStore
from core.retriever.ir import (
    And, Count, EventAt, HoldsAt, HoldsDuring, IRNode, Not, Or, PathBetween, Precedes,
    UnsupportedQueryError,
)


@dataclass
class CompiledLeaf:
    cypher: str
    params: dict


def compile_leaf(node: IRNode) -> CompiledLeaf:
    if isinstance(node, HoldsAt):
        return CompiledLeaf(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(st:State)-[:OBJECT_OF]->(o:Entity) "
            "WHERE st.type = $relation "
            "AND ($subject IS NULL OR s.id = $subject) "
            "AND ($object IS NULL OR o.id = $object) "
            "AND st.valid_from <= $t "
            "AND (st.valid_until IS NULL OR st.valid_until > $t) "
            "RETURN s.id AS subject, o.id AS object, st.valid_from AS valid_from, "
            "st.valid_until AS valid_until, st.sid AS sid, st.src_frames AS src_frames",
            {"relation": node.relation, "subject": node.subject, "object": node.object, "t": node.t},
        )

    if isinstance(node, HoldsDuring):
        # half-open intervals: valid_until is exclusive, matching the HoldsAt asymmetry (§6.3)
        overlap = (
            "st.valid_from <= $t0 AND (st.valid_until IS NULL OR st.valid_until >= $t1)"
            if node.mode == "all"
            else "st.valid_from < $t1 AND (st.valid_until IS NULL OR st.valid_until > $t0)"
        )
        return CompiledLeaf(
            "MATCH (s:Entity)-[:SUBJECT_OF]->(st:State)-[:OBJECT_OF]->(o:Entity) "
            "WHERE st.type = $relation "
            "AND ($subject IS NULL OR s.id = $subject) "
            "AND ($object IS NULL OR o.id = $object) "
            f"AND {overlap} "
            "RETURN s.id AS subject, o.id AS object, st.valid_from AS valid_from, "
            "st.valid_until AS valid_until, st.sid AS sid, st.src_frames AS src_frames",
            {"relation": node.relation, "subject": node.subject, "object": node.object,
             "t0": node.t0, "t1": node.t1},
        )

    if isinstance(node, EventAt):
        return CompiledLeaf(
            "MATCH (s:Entity)-[:EVENT_SUBJ]->(e:Event)-[:EVENT_OBJ]->(o:Entity) "
            "WHERE e.type = $relation "
            "AND ($subject IS NULL OR s.id = $subject) "
            "AND ($object IS NULL OR o.id = $object) "
            "AND ($t0 IS NULL OR e.timestamp >= $t0) "
            "AND ($t1 IS NULL OR e.timestamp <= $t1) "
            "RETURN s.id AS subject, o.id AS object, e.timestamp AS timestamp, e.src_frame AS src_frame",
            {"relation": node.relation, "subject": node.subject, "object": node.object,
             "t0": node.t0, "t1": node.t1},
        )

    if isinstance(node, Precedes):
        if not isinstance(node.a, EventAt) or not isinstance(node.b, EventAt):
            raise UnsupportedQueryError("precedes is defined over event_at operands in v0 (§6.5)")
        return CompiledLeaf(
            "MATCH (s1:Entity)-[:EVENT_SUBJ]->(e1:Event) "
            "WHERE e1.type = $rel_a AND ($subj_a IS NULL OR s1.id = $subj_a) "
            "MATCH (s2:Entity)-[:EVENT_SUBJ]->(e2:Event) "
            "WHERE e2.type = $rel_b AND ($subj_b IS NULL OR s2.id = $subj_b) "
            "AND e2.timestamp > e1.timestamp "
            "RETURN e1.timestamp AS a_at, e2.timestamp AS b_at, "
            "s1.id AS subject_a, s2.id AS subject_b, "
            "e1.src_frame AS a_frame, e2.src_frame AS b_frame "
            "ORDER BY a_at LIMIT 1",
            {"rel_a": node.a.relation, "subj_a": node.a.subject,
             "rel_b": node.b.relation, "subj_b": node.b.subject},
        )

    if isinstance(node, PathBetween):
        # One hop through a single State. max_hops != 1 is rejected in validate_ir rather than
        # silently under-answered here; re-checked so a direct compiler call can't bypass it.
        if node.max_hops != 1:
            raise UnsupportedQueryError(f"path_between supports only max_hops=1 in v0 (got {node.max_hops})")
        return CompiledLeaf(
            "MATCH (a:Entity {id: $source})-[:SUBJECT_OF]->(st:State)-[:OBJECT_OF]->(b:Entity {id: $target}) "
            "RETURN st.sid AS sid, st.type AS type, st.valid_from AS valid_from, "
            "st.valid_until AS valid_until, st.src_frames AS src_frames",
            {"source": node.source, "target": node.target},
        )

    raise UnsupportedQueryError(f"{type(node).__name__} is not an implemented leaf operator in v0")


@dataclass
class ExecutionTrace:
    answer: bool | int
    cypher: list[str]
    rows: list[dict]
    provenance: list[int]
    warnings: list[str]


def execute_ir(node: IRNode, store: GraphStore) -> ExecutionTrace:
    if isinstance(node, And):
        traces = [execute_ir(c, store) for c in node.children]
        return _combine(traces, all(t.answer for t in traces))

    if isinstance(node, Or):
        traces = [execute_ir(c, store) for c in node.children]
        return _combine(traces, any(t.answer for t in traces))

    if isinstance(node, Not):
        inner = execute_ir(node.child, store)
        return ExecutionTrace(not inner.answer, inner.cypher, inner.rows, inner.provenance, inner.warnings)

    if isinstance(node, Count):
        inner_leaf = compile_leaf(node.inner)
        rows = store.execute(inner_leaf.cypher, inner_leaf.params)
        return ExecutionTrace(len(rows), [inner_leaf.cypher], rows, _provenance(rows), [])

    leaf = compile_leaf(node)
    rows = store.execute(leaf.cypher, leaf.params)
    return ExecutionTrace(len(rows) > 0, [leaf.cypher], rows, _provenance(rows), [])


def _combine(traces: list[ExecutionTrace], answer: bool) -> ExecutionTrace:
    cypher, rows, provenance, warnings = [], [], [], []
    for t in traces:
        cypher.extend(t.cypher)
        rows.extend(t.rows)
        provenance.extend(t.provenance)
        warnings.extend(t.warnings)
    return ExecutionTrace(answer, cypher, rows, provenance, warnings)


#: Single-frame columns (event queries) and the comma-joined column State nodes carry.
_FRAME_KEYS = ("src_frame", "a_frame", "b_frame")


def _provenance(rows: list[dict]) -> list[int]:
    """Source frame indices backing an answer (§6.4) — never a bare boolean without evidence."""
    frames: list[int] = []
    for row in rows:
        for key in _FRAME_KEYS:
            if row.get(key) is not None:
                frames.append(int(row[key]))
        # State rows carry every contributing frame, joined at write time
        for part in (row.get("src_frames") or "").split(","):
            if part.strip():
                frames.append(int(part))
    return sorted(dict.fromkeys(frames))
