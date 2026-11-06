import json

from core.retriever.ir import validate_ir
from core.schema.builtin.warehouse import WAREHOUSE
from eval.metrics import parse_ir
from eval.qa_harness import QA_SET_PATH, load_qa_set, run


def test_missing_clips_are_skipped_not_errored():
    report = run()
    assert report["n_total"] == len(load_qa_set())
    assert report["n_skipped_missing_video"] == report["n_total"]  # no clips checked in yet
    assert report["results"] == []


def test_placeholder_qa_set_expected_ir_is_well_formed():
    """Every checked-in placeholder must parse and validate against WAREHOUSE, even before a
    real clip exists — this is what keeps the QA-pair shape honest as clips get added."""
    for pair in load_qa_set():
        ir = parse_ir(pair["expected_ir"])
        validate_ir(ir, WAREHOUSE)  # raises SchemaValidationError on an out-of-vocabulary relation


def test_qa_set_is_valid_jsonl():
    with open(QA_SET_PATH) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) >= 3
    for line in lines:
        json.loads(line)  # raises on malformed JSON
