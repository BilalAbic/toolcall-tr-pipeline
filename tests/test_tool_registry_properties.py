from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import pytest
from hypothesis import given
from hypothesis import strategies as st

from toolcall_tr.hashing import JsonValue, to_json_value
from toolcall_tr.models import RawToolDefinition, ToolFunction
from toolcall_tr.tool_registry import ToolNormalizationError, normalize_tool


def definition(schema: Mapping[str, object], description: str = "docs") -> RawToolDefinition:
    converted = to_json_value(schema)
    assert isinstance(converted, dict)
    return RawToolDefinition(
        type="function",
        function=ToolFunction(
            name="example",
            description=description,
            parameters=converted,
            strict=None,
        ),
    )


@given(st.permutations(["a", "b", "c"]))
def test_required_order_does_not_change_tool_id(order: list[str]) -> None:
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in ["a", "b", "c"]},
        "required": order,
    }
    baseline = deepcopy(schema)
    baseline["required"] = ["a", "b", "c"]
    assert (
        normalize_tool(definition(schema)).canonical.tool_id
        == normalize_tool(definition(baseline)).canonical.tool_id
    )


@given(st.permutations(["red", "green", "blue"]))
def test_enum_order_does_not_change_tool_id(order: list[str]) -> None:
    schema = {"type": "string", "enum": order}
    assert (
        normalize_tool(definition(schema)).canonical.tool_id
        == normalize_tool(
            definition({"type": "string", "enum": ["red", "green", "blue"]})
        ).canonical.tool_id
    )


def test_description_changes_documentation_not_structure() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "old"}},
    }
    changed = deepcopy(schema)
    properties = changed["properties"]
    assert isinstance(properties, dict)
    city = properties["city"]
    assert isinstance(city, dict)
    city["description"] = "new"
    left = normalize_tool(definition(schema, "old docs")).canonical
    right = normalize_tool(definition(changed, "new docs")).canonical
    assert left.tool_id == right.tool_id
    assert left.documentation_hash != right.documentation_hash


@pytest.mark.parametrize(
    "changed",
    [
        {"type": "object", "properties": {"city": {"type": "number"}}},
        {"type": "object", "properties": {"town": {"type": "string"}}},
        {"type": "object", "properties": {"city": {"type": "string", "default": "X"}}},
    ],
)
def test_validation_semantics_change_tool_id(changed: dict[str, JsonValue]) -> None:
    base: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert (
        normalize_tool(definition(base)).canonical.tool_id
        != normalize_tool(definition(changed)).canonical.tool_id
    )


def test_order_semantic_prefix_items_change_tool_id() -> None:
    left = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}]}
    right = {"type": "array", "prefixItems": [{"type": "number"}, {"type": "string"}]}
    assert (
        normalize_tool(definition(left)).canonical.tool_id
        != normalize_tool(definition(right)).canonical.tool_id
    )


@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf"])
def test_schema_combinator_order_does_not_change_tool_id(keyword: str) -> None:
    left = {
        keyword: [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10},
        ]
    }
    right = {keyword: list(reversed(left[keyword]))}
    assert (
        normalize_tool(definition(left)).canonical.tool_id
        == normalize_tool(definition(right)).canonical.tool_id
    )


def test_unknown_keyword_and_remote_ref_fail_closed() -> None:
    with pytest.raises(ToolNormalizationError, match="Unsupported") as unknown:
        normalize_tool(definition({"type": "object", "mystery": True}))
    assert unknown.value.code == "SCHEMA_UNSUPPORTED_KEYWORD"
    with pytest.raises(ToolNormalizationError, match="Remote"):
        normalize_tool(definition({"$ref": "https://example.com/schema.json"}))


def test_duplicate_set_value_fails_closed() -> None:
    with pytest.raises(ToolNormalizationError) as error:
        normalize_tool(definition({"type": "object", "required": ["x", "x"]}))
    assert error.value.code == "SCHEMA_DUPLICATE_SET_VALUE"
