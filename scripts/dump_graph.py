"""Print a built graph as a flat timeline: entities, events in order, state intervals.

Usage: python scripts/dump_graph.py <.vgc_cache/graphs/<hash>/graph.kuzu>
The counts in graph.report() say how much was extracted; this says what it actually says.
"""

import sys
from core.graph.kuzu_store import KuzuStore

s = KuzuStore(sys.argv[1])

print("=== ENTITIES ===")
for r in s.execute("MATCH (n:Entity) RETURN n.id, n.class, n.first_seen, n.last_seen ORDER BY n.first_seen, n.id", {}):
    print(f"  {r['n.id']:<24} {r['n.class']:<10} seen {r['n.first_seen']:.2f}..{r['n.last_seen']:.2f}")

print("\n=== EVENTS (chronological) ===")
for r in s.execute(
    "MATCH (s:Entity)-[:EVENT_SUBJ]->(e:Event)-[:EVENT_OBJ]->(o:Entity) "
    "RETURN e.timestamp, s.id, e.type, o.id, e.src_frame ORDER BY e.timestamp", {}):
    print(f"  t={r['e.timestamp']:>6.2f} f{r['e.src_frame']:<4} {r['s.id']} --{r['e.type']}--> {r['o.id']}")

print("\n=== STATES (intervals) ===")
for r in s.execute(
    "MATCH (s:Entity)-[:SUBJECT_OF]->(st:State)-[:OBJECT_OF]->(o:Entity) "
    "RETURN st.valid_from, st.valid_until, s.id, st.type, o.id, st.flags, st.src_frames "
    "ORDER BY st.valid_from", {}):
    until = r['st.valid_until']
    span = f"{r['st.valid_from']:>6.2f}..{until:<6.2f}" if until is not None else f"{r['st.valid_from']:>6.2f}..open  "
    flags = f"  [{r['st.flags']}]" if r['st.flags'] else ""
    print(f"  {span} {r['s.id']} --{r['st.type']}--> {r['o.id']}{flags}  frames={r['st.src_frames']}")
