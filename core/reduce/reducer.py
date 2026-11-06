"""StateReducer: instantaneous events -> interval-valid states (SPEC §4.6).

Pure function of (events, schema). Every branch below corresponds to
one row of the §4.6 edge-case policy table.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from core.reduce.policies import State, state_id
from core.schema.types import SkillSchema, TemporalEdge, event_id

Key = tuple[str, str, str]  # (durative_relation_name, subject, object)
# open_intervals value: (valid_from, opening_event_id, flags accumulated so far)
OpenValue = tuple[float, str, list[str]]


class StateReducer:
    def __init__(self, schema: SkillSchema):
        self.schema = schema

    def _resolve_ties(self, ordered: list[TemporalEdge]):
        """Group (event, durative relation) pairs by (key, timestamp); when more than one applies
        to the same key at the identical instant, only the deterministic tie-break winner (first
        in sort order) is applied, and it's flagged (§4.6: "deterministic tie-break ... and flag")."""
        pairs = [(event, rel) for event in ordered for rel in self.schema.durative_for(event.relation)]
        groups: dict[tuple[Key, float], list] = defaultdict(list)
        for event, rel in pairs:
            key = (rel.name, event.source_id, event.target_id)
            groups[(key, event.timestamp)].append((event, rel))

        resolved = []
        for event, rel in pairs:
            key = (rel.name, event.source_id, event.target_id)
            group = groups[(key, event.timestamp)]
            if group[0] != (event, rel):
                continue  # loser of a same-instant contradiction: not applied
            flags = ["contradictory_at_identical_timestamp"] if len(group) > 1 else []
            resolved.append((event, rel, flags))
        return resolved

    def reduce(self, events: Iterable[TemporalEdge]) -> list[State]:
        """Intervals still open at the end of the stream get valid_until=None, which means
        'still true at the observation horizon' — explicitly not 'unknown' (§4.6). No horizon
        argument is needed: None already carries that meaning, and a numeric horizon would
        force a false distinction between 'ended exactly at EOF' and 'still running'.
        """
        ordered = sorted(events, key=lambda e: (e.timestamp, e.relation, e.source_id))

        states: list[State] = []
        open_intervals: dict[Key, OpenValue] = {}
        open_by_object: dict[tuple[str, str], str] = {}  # (rel_name, object) -> subject, one_subject_per_object only

        for event, rel, flags in self._resolve_ties(ordered):
            key = (rel.name, event.source_id, event.target_id)
            eid = event_id(event)

            if event.relation in rel.opens_on:
                if rel.cardinality == "one_subject_per_object":
                    obj_key = (rel.name, event.target_id)
                    prev_subject = open_by_object.get(obj_key)
                    if prev_subject is not None and prev_subject != event.source_id:
                        prev_key = (rel.name, prev_subject, event.target_id)
                        prev_from, prev_eid, prev_flags = open_intervals.pop(prev_key)
                        states.append(State(
                            sid=state_id(rel.name, prev_subject, event.target_id, prev_from),
                            type=rel.name, subject=prev_subject, object=event.target_id,
                            valid_from=prev_from, valid_until=event.timestamp,
                            flags=tuple(prev_flags), derived_from=(prev_eid, eid),
                        ))
                    open_by_object[obj_key] = event.source_id

                if key not in open_intervals:
                    # first open; a repeated opens_on while already open extends silently (no-op)
                    open_intervals[key] = (event.timestamp, eid, flags)
                elif flags:
                    open_intervals[key][2].extend(flags)

            if event.relation in rel.closes_on:
                if key in open_intervals:
                    valid_from, open_eid, open_flags = open_intervals.pop(key)
                    open_flags = open_flags + flags
                    if rel.cardinality == "one_subject_per_object":
                        obj_key = (rel.name, event.target_id)
                        if open_by_object.get(obj_key) == event.source_id:
                            del open_by_object[obj_key]
                    states.append(State(
                        sid=state_id(rel.name, event.source_id, event.target_id, valid_from),
                        type=rel.name, subject=event.source_id, object=event.target_id,
                        valid_from=valid_from, valid_until=event.timestamp,
                        flags=tuple(open_flags), derived_from=(open_eid, eid),
                    ))
                else:
                    # closes_on with no open interval: zero-length state, flagged, not dropped
                    states.append(State(
                        sid=state_id(rel.name, event.source_id, event.target_id, event.timestamp),
                        type=rel.name, subject=event.source_id, object=event.target_id,
                        valid_from=event.timestamp, valid_until=event.timestamp,
                        flags=tuple(flags + ["closed_without_open"]), derived_from=(eid,),
                    ))

        # still open at EOF -> valid_until=None means "still true", not "unknown"
        for (rel_name, subject, obj), (valid_from, open_eid, flags) in open_intervals.items():
            states.append(State(
                sid=state_id(rel_name, subject, obj, valid_from),
                type=rel_name, subject=subject, object=obj,
                valid_from=valid_from, valid_until=None,
                flags=tuple(flags), derived_from=(open_eid,),
            ))

        return states
