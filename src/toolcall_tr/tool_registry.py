"""Versioned JSON Schema normalizer and semantic tool registry."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import cast

from jsonschema import Draft202012Validator

from toolcall_tr.constants import NORMALIZER_VERSION
from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_jcs, stable_id
from toolcall_tr.models import CanonicalTool, RawToolDefinition

DOCUMENTATION_KEYS = frozenset({"description", "title", "$comment", "examples"})
SET_LIKE_ARRAY_KEYS = frozenset({"required", "enum", "type"})
SCHEMA_MAP_KEYS = frozenset({"properties", "patternProperties", "dependentSchemas", "$defs"})
SET_LIKE_SCHEMA_ARRAY_KEYS = frozenset({"allOf", "anyOf", "oneOf"})
ORDERED_SCHEMA_ARRAY_KEYS = frozenset({"prefixItems"})
SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
SUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$ref",
        "$schema",
        "$vocabulary",
        "additionalItems",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "default",
        "dependentRequired",
        "dependentSchemas",
        "deprecated",
        "description",
        "else",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "if",
        "items",
        "maxContains",
        "maximum",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minimum",
        "minItems",
        "minLength",
        "minProperties",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "readOnly",
        "required",
        "then",
        "title",
        "type",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
        "writeOnly",
    }
)


class ToolNormalizationError(ValueError):
    def __init__(self, code: str, message: str, pointer: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.pointer = pointer or "/"


def _pointer(parent: str, token: str) -> str:
    escaped = token.replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def _sort_unique(values: list[JsonValue], pointer: str) -> list[JsonValue]:
    keyed = sorted((canonical_bytes(value), value) for value in values)
    if any(left[0] == right[0] for left, right in itertools.pairwise(keyed)):
        raise ToolNormalizationError(
            "SCHEMA_DUPLICATE_SET_VALUE", "Set-like schema list contains duplicate values", pointer
        )
    return [value for _, value in keyed]


def _normalize_schema(value: JsonValue, pointer: str = "") -> JsonValue:
    if isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        raise ToolNormalizationError(
            "TOOL_DEFINITION_INVALID", "Every schema node must be an object or boolean", pointer
        )

    unknown = set(value) - SUPPORTED_SCHEMA_KEYS
    if unknown:
        key = sorted(unknown)[0]
        raise ToolNormalizationError(
            "SCHEMA_UNSUPPORTED_KEYWORD",
            f"Unsupported JSON Schema keyword: {key}",
            _pointer(pointer, key),
        )
    ref = value.get("$ref")
    dynamic_ref = value.get("$dynamicRef")
    for keyword, candidate in (("$ref", ref), ("$dynamicRef", dynamic_ref)):
        if isinstance(candidate, str) and not candidate.startswith("#"):
            raise ToolNormalizationError(
                "SCHEMA_UNSUPPORTED_KEYWORD",
                f"Remote {keyword} is forbidden; bundle it into the snapshot",
                _pointer(pointer, keyword),
            )

    result: dict[str, JsonValue] = {}
    for key in sorted(value):
        item = value[key]
        child_pointer = _pointer(pointer, key)
        if key in DOCUMENTATION_KEYS:
            continue
        if key in SCHEMA_MAP_KEYS:
            if not isinstance(item, dict):
                raise ToolNormalizationError(
                    "TOOL_DEFINITION_INVALID", f"{key} must be an object", child_pointer
                )
            result[key] = {
                map_key: _normalize_schema(map_value, _pointer(child_pointer, map_key))
                for map_key, map_value in sorted(item.items())
            }
        elif key in SET_LIKE_SCHEMA_ARRAY_KEYS | ORDERED_SCHEMA_ARRAY_KEYS:
            if not isinstance(item, list):
                raise ToolNormalizationError(
                    "TOOL_DEFINITION_INVALID", f"{key} must be an array", child_pointer
                )
            normalized_items = [
                _normalize_schema(schema, _pointer(child_pointer, str(index)))
                for index, schema in enumerate(item)
            ]
            result[key] = (
                sorted(normalized_items, key=canonical_bytes)
                if key in SET_LIKE_SCHEMA_ARRAY_KEYS
                else normalized_items
            )
        elif key in SCHEMA_VALUE_KEYS:
            if not isinstance(item, dict | bool):
                raise ToolNormalizationError(
                    "TOOL_DEFINITION_INVALID", f"{key} must be a schema", child_pointer
                )
            result[key] = _normalize_schema(item, child_pointer)
        elif key == "dependentRequired":
            if not isinstance(item, dict):
                raise ToolNormalizationError(
                    "TOOL_DEFINITION_INVALID", "dependentRequired must be an object", child_pointer
                )
            normalized_dependent: dict[str, JsonValue] = {}
            for property_name, dependencies in sorted(item.items()):
                if not isinstance(dependencies, list) or not all(
                    isinstance(dependency, str) for dependency in dependencies
                ):
                    raise ToolNormalizationError(
                        "TOOL_DEFINITION_INVALID",
                        "dependentRequired values must be string arrays",
                        _pointer(child_pointer, property_name),
                    )
                normalized_dependent[property_name] = _sort_unique(
                    cast(list[JsonValue], dependencies), _pointer(child_pointer, property_name)
                )
            result[key] = normalized_dependent
        elif key in SET_LIKE_ARRAY_KEYS:
            if key == "type" and isinstance(item, str):
                result[key] = item
            elif isinstance(item, list):
                result[key] = _sort_unique(item, child_pointer)
            else:
                raise ToolNormalizationError(
                    "TOOL_DEFINITION_INVALID", f"{key} has an invalid value", child_pointer
                )
        else:
            result[key] = item
    return result


@dataclass(frozen=True, slots=True)
class NormalizedTool:
    canonical: CanonicalTool
    structural_schema: JsonValue


def normalize_tool(raw: RawToolDefinition) -> NormalizedTool:
    parameters = raw.function.parameters
    structural = _normalize_schema(parameters)
    try:
        Draft202012Validator.check_schema(parameters)
    except Exception as exc:
        raise ToolNormalizationError("TOOL_DEFINITION_INVALID", str(exc), "/parameters") from exc
    semantic_hash = sha256_jcs(structural)
    tool_id = stable_id("tool", {"name": raw.function.name, "schema": structural})
    documentation_hash = sha256_jcs(
        {
            "name": raw.function.name,
            "description": raw.function.description,
            "parameters": parameters,
        }
    )
    canonical = CanonicalTool(
        tool_id=tool_id,
        raw_schema_hash=sha256_jcs(parameters),
        semantic_schema_hash=semantic_hash,
        documentation_hash=documentation_hash,
        normalizer_version=NORMALIZER_VERSION,
        type="function",
        function=raw.function,
    )
    return NormalizedTool(canonical=canonical, structural_schema=structural)


class ToolRegistry:
    """In-memory deterministic view; JSONL remains the persisted truth."""

    def __init__(self, tools: list[CanonicalTool]) -> None:
        self.tools = tuple(sorted(tools, key=lambda tool: (tool.tool_id, tool.documentation_hash)))

    @classmethod
    def build(cls, definitions: list[RawToolDefinition]) -> ToolRegistry:
        unique: dict[tuple[str, str], CanonicalTool] = {}
        for definition in definitions:
            tool = normalize_tool(definition).canonical
            unique[(tool.tool_id, tool.documentation_hash)] = tool
        return cls(list(unique.values()))

    def by_name(self, name: str) -> list[CanonicalTool]:
        return [tool for tool in self.tools if tool.function.name == name]

    def as_records(self) -> list[dict[str, JsonValue]]:
        return [
            cast(dict[str, JsonValue], tool.model_dump(mode="json", exclude_none=False))
            for tool in self.tools
        ]
