from core.graph.base import GraphEntity
from core.graph.kuzu_store import KuzuStore
from core.reduce.policies import State
from core.retriever.compiler.kuzu import execute_ir
from core.retriever.ir import EventAt, HoldsAt, Precedes
from core.schema.builtin.warehouse import WAREHOUSE
from core.schema.types import TemporalEdge, event_id


def make_store(tmp_path):
    store = KuzuStore(tmp_path / "graph.kuzu")
    store.init_schema(WAREHOUSE)

    entities = [
        GraphEntity(id="P1", class_label="person", first_seen=0.0, last_seen=20.0),
        GraphEntity(id="B1", class_label="bag", first_seen=0.0, last_seen=20.0),
        GraphEntity(id="D1", class_label="door", first_seen=0.0, last_seen=20.0),
    ]
    events = [
        TemporalEdge(source_id="P1", target_id="B1", relation="PICKS_UP", timestamp=3.1, src_frame=1),
        TemporalEdge(source_id="P1", target_id="B1", relation="DROPS", timestamp=12.4, src_frame=2),
        TemporalEdge(source_id="P1", target_id="D1", relation="EXITS", timestamp=19.2, src_frame=3),
    ]
    states = [
        State(
            sid="s1", type="POSSESSES", subject="P1", object="B1",
            valid_from=3.1, valid_until=12.4,
            # the two events that opened and closed this interval, in frames 1 and 2
            derived_from=(event_id(events[0]), event_id(events[1])),
        ),
    ]
    store.write(entities, states, events)
    return store


def test_holds_at_true_inside_interval(tmp_path):
    store = make_store(tmp_path)
    trace = execute_ir(HoldsAt(relation="POSSESSES", subject="P1", object="B1", t=5.0), store)
    assert trace.answer is True


def test_holds_at_false_outside_interval(tmp_path):
    store = make_store(tmp_path)
    trace = execute_ir(HoldsAt(relation="POSSESSES", subject="P1", object="B1", t=15.0), store)
    assert trace.answer is False


def test_holds_at_boundary_is_half_open(tmp_path):
    # valid_until is exclusive: an instant equal to valid_until is outside the state (§6.3)
    store = make_store(tmp_path)
    trace = execute_ir(HoldsAt(relation="POSSESSES", subject="P1", object="B1", t=12.4), store)
    assert trace.answer is False


def test_precedes_true_when_a_happens_before_b(tmp_path):
    store = make_store(tmp_path)
    ir = Precedes(a=EventAt(relation="DROPS", subject="P1"), b=EventAt(relation="EXITS", subject="P1"))
    trace = execute_ir(ir, store)
    assert trace.answer is True
    assert trace.provenance  # src_frame values present for audit


def test_precedes_false_when_order_is_reversed(tmp_path):
    store = make_store(tmp_path)
    ir = Precedes(a=EventAt(relation="EXITS", subject="P1"), b=EventAt(relation="DROPS", subject="P1"))
    trace = execute_ir(ir, store)
    assert trace.answer is False


def test_holds_at_returns_provenance(tmp_path):
    """A3: state queries used to always return empty provenance, leaving half of §6.4's
    'never a bare boolean' contract unmet."""
    store = make_store(tmp_path)
    trace = execute_ir(HoldsAt(relation="POSSESSES", subject="P1", object="B1", t=5.0), store)
    assert trace.answer is True
    assert trace.provenance == [1, 2]  # the opening and closing event frames
