from core.reduce.reducer import StateReducer
from core.schema.builtin.warehouse import WAREHOUSE
from core.schema.types import TemporalEdge


def edge(rel, src, dst, t):
    return TemporalEdge(source_id=src, target_id=dst, relation=rel, timestamp=t)


def test_basic_timeline_from_spec_diagram():
    # SPEC §4.6: P1 picks up B1, enters D1, drops B1, exits D1; P2 picks up B1
    events = [
        edge("PICKS_UP", "P1", "B1", 3.1),
        edge("ENTERS", "P1", "D1", 8.0),
        edge("DROPS", "P1", "B1", 12.4),
        edge("EXITS", "P1", "D1", 19.2),
        edge("PICKS_UP", "P2", "B1", 25.0),
    ]
    states = StateReducer(WAREHOUSE).reduce(events)
    by_type = {(s.type, s.subject, s.object): s for s in states}

    possesses_p1 = by_type[("POSSESSES", "P1", "B1")]
    assert possesses_p1.valid_from == 3.1 and possesses_p1.valid_until == 12.4

    inside_p1 = by_type[("INSIDE", "P1", "D1")]
    assert inside_p1.valid_from == 8.0 and inside_p1.valid_until == 19.2

    possesses_p2 = by_type[("POSSESSES", "P2", "B1")]
    assert possesses_p2.valid_from == 25.0 and possesses_p2.valid_until is None
    assert not possesses_p2.flags


def test_closes_on_with_no_open_interval_is_flagged_zero_length():
    events = [edge("DROPS", "P1", "B1", 5.0)]
    states = StateReducer(WAREHOUSE).reduce(events)
    assert len(states) == 1
    assert states[0].valid_from == states[0].valid_until == 5.0
    assert "closed_without_open" in states[0].flags


def test_second_open_while_already_open_extends_silently():
    events = [
        edge("PICKS_UP", "P1", "B1", 1.0),
        edge("PICKS_UP", "P1", "B1", 2.0),  # repeated observation of ongoing state
        edge("DROPS", "P1", "B1", 5.0),
    ]
    states = StateReducer(WAREHOUSE).reduce(events)
    assert len(states) == 1
    assert states[0].valid_from == 1.0 and states[0].valid_until == 5.0


def test_one_subject_per_object_violation_auto_closes_prior_holder():
    events = [
        edge("PICKS_UP", "P1", "B1", 1.0),
        edge("PICKS_UP", "P2", "B1", 4.0),  # handoff without an explicit DROPS/HANDS_OFF
    ]
    states = StateReducer(WAREHOUSE).reduce(events)
    by_subject = {s.subject: s for s in states if s.type == "POSSESSES"}
    assert by_subject["P1"].valid_from == 1.0 and by_subject["P1"].valid_until == 4.0
    assert by_subject["P2"].valid_from == 4.0 and by_subject["P2"].valid_until is None


def test_unconstrained_cardinality_keeps_both_overlapping_intervals():
    events = [
        edge("ENTERS", "P1", "D1", 1.0),
        edge("ENTERS", "P2", "D1", 2.0),
        edge("EXITS", "P1", "D1", 3.0),
        edge("EXITS", "P2", "D1", 4.0),
    ]
    states = StateReducer(WAREHOUSE).reduce(events)
    inside = [s for s in states if s.type == "INSIDE"]
    assert len(inside) == 2


def test_contradictory_events_at_identical_timestamp_are_flagged():
    events = [
        edge("PICKS_UP", "P1", "B1", 1.0),
        edge("DROPS", "P1", "B1", 1.0),
    ]
    states = StateReducer(WAREHOUSE).reduce(events)
    assert len(states) == 1
    assert "contradictory_at_identical_timestamp" in states[0].flags


def test_derived_from_links_state_back_to_its_events():
    events = [
        edge("PICKS_UP", "P1", "B1", 1.0),
        edge("DROPS", "P1", "B1", 5.0),
    ]
    states = StateReducer(WAREHOUSE).reduce(events)
    assert len(states[0].derived_from) == 2
