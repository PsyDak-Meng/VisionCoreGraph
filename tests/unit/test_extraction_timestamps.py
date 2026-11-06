"""A1 regression: event timestamps must come from the keyframe, never from the VLM.

The reducer anchors every interval boundary to these values, so a model-supplied timestamp
would silently corrupt the whole graph (SPEC D6's argument, applied to event times).
"""

import pytest
from pydantic import ValidationError

from core.schema.builtin.warehouse import WAREHOUSE
from core.schema.types import make_semantic_delta_model


def test_vlm_model_has_no_timestamp_field():
    """The field is absent from the schema entirely — the model cannot emit one."""
    model = make_semantic_delta_model(WAREHOUSE)
    relation_fields = model.model_fields["relationships"].annotation.__args__[0].model_fields
    assert "timestamp" not in relation_fields
    assert "timestamp" not in model.model_fields


def test_model_supplied_timestamp_is_ignored_not_trusted():
    """Even if a model volunteers a timestamp, it must not survive into the parsed delta."""
    model = make_semantic_delta_model(WAREHOUSE)
    raw = """{
        "entities": [{"local_id": "t1", "class_label": "person", "bbox": [0,0,10,10]}],
        "relationships": [{"source_id": "t1", "target_id": "t2",
                           "relation": "PICKS_UP", "timestamp": 99999.0}]
    }"""
    delta = model.model_validate_json(raw)
    assert not hasattr(delta.relationships[0], "timestamp")


def test_out_of_vocabulary_relation_still_rejected():
    """The vocabulary constraint that keeps the ontology closed must survive the refactor."""
    model = make_semantic_delta_model(WAREHOUSE)
    raw = """{
        "entities": [],
        "relationships": [{"source_id": "t1", "target_id": "t2", "relation": "TELEPORTS"}]
    }"""
    with pytest.raises(ValidationError):
        model.model_validate_json(raw)
