"""Fail-closed field policy, leaf extraction, and host-side segment merge.

This module deliberately has no provider, prompt, or network dependency.  It
turns only policy-approved natural-language leaves into translation segments;
the canonical JSON tree itself always remains host-owned.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from toolcall_tr.hashing import JsonValue, canonical_bytes, sha256_jcs, stable_id
from toolcall_tr.models import (
    CanonicalEpisode,
    EpisodeId,
    NonEmptyStr,
    RawToolDefinition,
    Role,
    StrictModel,
)
from toolcall_tr.tool_registry import normalize_tool

_JSON_POINTER = r"^/(?:[^/~]|~[01])+(?:/[^/]*)*$"
_URL_OR_SCHEME = re.compile(r"(?:^[a-z][a-z0-9+.-]*:|://|^www\.)", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"^[a-z]:[\\/]", re.IGNORECASE)
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_LONG_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{24,}$")


class FieldPolicyError(ValueError):
    """Raised when a text leaf has no safe, explicit policy outcome."""


class FieldAction(StrEnum):
    COPY_EXACT = "copy_exact"
    TRANSLATE = "translate"
    TRANSLATE_IF_NATURAL_LANGUAGE = "translate_if_natural_language"
    OMIT_FROM_MODEL_INPUT = "omit_from_model_input"
    MANUAL_POLICY_REQUIRED = "manual_policy_required"


class ArgumentPathPolicy(StrictModel):
    """An explicit policy for one leaf of a named function's arguments."""

    tool_name: NonEmptyStr
    argument_pointer: Annotated[str, Field(pattern=_JSON_POINTER)]
    action: Literal[
        FieldAction.COPY_EXACT,
        FieldAction.TRANSLATE,
        FieldAction.MANUAL_POLICY_REQUIRED,
    ]


class FieldPolicy(StrictModel):
    """Versioned, closed field-policy contract for one extraction pass.

    ``translate_if_natural_language`` is intentionally not resolved here:
    this deterministic layer has no semantic classifier, so an encountered
    value must be promoted to an explicit ``translate`` or ``copy_exact``
    policy before it can leave the host.
    """

    schema_version: Literal["field-policy-0.1.0"] = "field-policy-0.1.0"
    policy_version: NonEmptyStr
    user_content_action: FieldAction = FieldAction.TRANSLATE
    assistant_content_action: FieldAction = FieldAction.TRANSLATE
    system_content_action: FieldAction = FieldAction.COPY_EXACT
    developer_content_action: FieldAction = FieldAction.COPY_EXACT
    tool_content_action: FieldAction = FieldAction.COPY_EXACT
    tool_description_action: FieldAction = FieldAction.TRANSLATE_IF_NATURAL_LANGUAGE
    parameter_description_action: FieldAction = FieldAction.TRANSLATE_IF_NATURAL_LANGUAGE
    argument_policies: list[ArgumentPathPolicy]

    @model_validator(mode="after")
    def reject_duplicate_argument_policies(self) -> FieldPolicy:
        keys = [(item.tool_name, item.argument_pointer) for item in self.argument_policies]
        if len(set(keys)) != len(keys):
            raise ValueError("argument policies must be unique by tool name and pointer")
        return self

    def argument_action(self, tool_name: str, argument_pointer: str) -> FieldAction | None:
        for item in self.argument_policies:
            if item.tool_name == tool_name and item.argument_pointer == argument_pointer:
                return item.action
        return None


class Segment(StrictModel):
    """One policy-approved, source-owned textual leaf."""

    schema_version: Literal["translation-segment-0.1.0"] = "translation-segment-0.1.0"
    segment_id: Annotated[str, Field(pattern=r"^seg_[0-9a-f]{64}$")]
    episode_id: EpisodeId
    json_pointer: Annotated[str, Field(pattern=_JSON_POINTER)]
    source_text: str
    source_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    field_policy_version: NonEmptyStr


class SegmentExtraction(StrictModel):
    """Deterministic extract request; all omitted fields remain host-owned."""

    schema_version: Literal["segment-extraction-0.1.0"] = "segment-extraction-0.1.0"
    episode_id: EpisodeId
    input_variant_id: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    field_policy_version: NonEmptyStr
    segments: list[Segment]

    @model_validator(mode="after")
    def validate_segment_identity(self) -> SegmentExtraction:
        ids = [segment.segment_id for segment in self.segments]
        pointers = [segment.json_pointer for segment in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("segment IDs must be unique")
        if len(set(pointers)) != len(pointers):
            raise ValueError("segment pointers must be unique")
        if any(segment.episode_id != self.episode_id for segment in self.segments):
            raise ValueError("all segments must belong to the extraction episode")
        if any(
            segment.field_policy_version != self.field_policy_version
            for segment in self.segments
        ):
            raise ValueError("all segments must use the extraction field policy version")
        return self


class SegmentTranslation(StrictModel):
    """A bounded response for exactly one extracted segment."""

    schema_version: Literal["segment-translation-0.1.0"] = "segment-translation-0.1.0"
    segment_id: Annotated[str, Field(pattern=r"^seg_[0-9a-f]{64}$")]
    target_text: str


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _leaf_values(value: JsonValue, pointer: str = "") -> list[tuple[str, JsonValue]]:
    if isinstance(value, dict):
        return [
            leaf
            for key in sorted(value)
            for leaf in _leaf_values(value[key], f"{pointer}/{_escape_pointer_token(key)}")
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _leaf_values(item, f"{pointer}/{index}")
        ]
    return [(pointer or "/", value)]


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise FieldPolicyError(f"JSON Pointer must be absolute: {pointer}")
    return [_decode_pointer_token(token) for token in pointer[1:].split("/")]


def _set_pointer(document: JsonValue, pointer: str, replacement: str) -> None:
    tokens = _pointer_tokens(pointer)
    if not tokens:
        raise FieldPolicyError("the canonical document root may not be translated")
    current = document
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise FieldPolicyError(f"segment pointer no longer resolves: {pointer}")
            current = current[token]
        elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise FieldPolicyError(f"segment pointer no longer resolves: {pointer}")
    last = tokens[-1]
    if isinstance(current, dict) and last in current:
        current[last] = replacement
        return
    if isinstance(current, list) and last.isdecimal() and int(last) < len(current):
        current[int(last)] = replacement
        return
    raise FieldPolicyError(f"segment pointer no longer resolves: {pointer}")


def _argument_pointer_from_canonical(tokens: list[str]) -> tuple[str, str] | None:
    """Return ``(tool_name, argument_pointer)`` for a call-argument leaf."""
    if len(tokens) < 7 or tokens[:1] != ["conversation"]:
        return None
    if not tokens[1].isdecimal() or tokens[2] != "tool_calls" or not tokens[3].isdecimal():
        return None
    if tokens[4:6] != ["function", "arguments"]:
        return None
    # Function name is a sibling, so caller supplies it after resolving the call.
    return "", "/" + "/".join(_escape_pointer_token(token) for token in tokens[6:])


def _argument_context(
    document: dict[str, JsonValue], tokens: list[str]
) -> tuple[str, str, JsonValue] | None:
    location = _argument_pointer_from_canonical(tokens)
    if location is None:
        return None
    conversation = document.get("conversation")
    message_index = int(tokens[1])
    call_index = int(tokens[3])
    if not isinstance(conversation, list) or message_index >= len(conversation):
        raise FieldPolicyError("canonical conversation shape changed during extraction")
    message = conversation[message_index]
    if not isinstance(message, dict):
        raise FieldPolicyError("canonical conversation message is not an object")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or call_index >= len(calls):
        raise FieldPolicyError("canonical tool call shape changed during extraction")
    call = calls[call_index]
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        raise FieldPolicyError("canonical tool call function is not an object")
    function = cast(dict[str, JsonValue], call["function"])
    name = function.get("name")
    if not isinstance(name, str):
        raise FieldPolicyError("canonical tool call name is not a string")
    return name, location[1], function.get("arguments")


def _schema_for_argument_pointer(schema: JsonValue, pointer: str) -> JsonValue:
    current = schema
    for token in _pointer_tokens(pointer):
        if not isinstance(current, dict):
            raise FieldPolicyError(f"argument schema is not safely navigable at {pointer}")
        properties = current.get("properties")
        if isinstance(properties, dict) and token in properties:
            current = properties[token]
            continue
        items = current.get("items")
        if isinstance(items, dict) and token.isdecimal():
            current = items
            continue
        raise FieldPolicyError(f"argument schema has no explicit leaf policy at {pointer}")
    return current


def _argument_translation_is_safe(value: str, schema: JsonValue, pointer: str) -> None:
    if not isinstance(schema, dict):
        raise FieldPolicyError(f"argument schema must be an object for translation: {pointer}")
    if schema.get("type") != "string":
        raise FieldPolicyError(
            f"argument translation requires an explicit string schema: {pointer}"
        )
    forbidden_keywords = {
        "enum",
        "const",
        "format",
        "pattern",
        "contentEncoding",
        "contentMediaType",
    }
    present = sorted(keyword for keyword in forbidden_keywords if keyword in schema)
    if present:
        raise FieldPolicyError(
            f"argument value is technical because its schema declares {present[0]}: {pointer}"
        )
    if (
        _URL_OR_SCHEME.search(value)
        or _WINDOWS_PATH.match(value)
        or value.startswith(("/", "\\"))
    ):
        raise FieldPolicyError(
            f"argument value looks like a URL or path and cannot be translated: {pointer}"
        )
    if _UUID.fullmatch(value) or _LONG_IDENTIFIER.fullmatch(value) or value.isdecimal():
        raise FieldPolicyError(
            f"argument value looks like an identifier and cannot be translated: {pointer}"
        )


def _message_content_action(role: str, policy: FieldPolicy) -> FieldAction:
    actions = {
        Role.USER.value: policy.user_content_action,
        Role.ASSISTANT.value: policy.assistant_content_action,
        Role.SYSTEM.value: policy.system_content_action,
        Role.DEVELOPER.value: policy.developer_content_action,
        Role.TOOL.value: policy.tool_content_action,
    }
    try:
        return actions[role]
    except KeyError as exc:
        raise FieldPolicyError(f"uncovered message role: {role}") from exc


def _classify_text_leaf(
    document: dict[str, JsonValue],
    pointer: str,
    policy: FieldPolicy,
) -> FieldAction:
    tokens = _pointer_tokens(pointer)
    if not tokens:
        raise FieldPolicyError("the canonical document root cannot be textual")

    argument = _argument_context(document, tokens)
    if argument is not None:
        tool_name, argument_pointer, _ = argument
        action = policy.argument_action(tool_name, argument_pointer)
        if action is None:
            raise FieldPolicyError(
                f"uncovered textual argument path for {tool_name}{argument_pointer}: {pointer}"
            )
        return action

    if len(tokens) == 3 and tokens[0] == "conversation" and tokens[1].isdecimal():
        field = tokens[2]
        conversation = document.get("conversation")
        if not isinstance(conversation, list) or int(tokens[1]) >= len(conversation):
            raise FieldPolicyError(f"conversation pointer does not resolve: {pointer}")
        message = conversation[int(tokens[1])]
        if not isinstance(message, dict):
            raise FieldPolicyError(f"conversation item is not an object: {pointer}")
        if field == "content":
            role = message.get("role")
            if not isinstance(role, str):
                raise FieldPolicyError(f"message role is not a string: {pointer}")
            return _message_content_action(role, policy)
        if field in {"reasoning_content", "thinking", "name", "tool_call_id", "role"}:
            return FieldAction.COPY_EXACT

    if (
        len(tokens) == 4
        and tokens[0] == "tools"
        and tokens[1].isdecimal()
        and tokens[2:] == ["function", "description"]
    ):
        return policy.tool_description_action

    if (
        len(tokens) >= 5
        and tokens[0] == "tools"
        and tokens[1].isdecimal()
        and tokens[2:4] == ["function", "parameters"]
    ):
        if tokens[-1] == "description":
            return policy.parameter_description_action
        if "examples" in tokens[4:] or tokens[-1] in {"title", "$comment"}:
            return FieldAction.MANUAL_POLICY_REQUIRED

    # All remaining string leaves belong to the frozen canonical contract:
    # identifiers, hashes, schema syntax/defaults/enums, provenance, and quality
    # metadata are copied byte-for-byte by the host.
    known_roots = {
        "schema_version",
        "episode_id",
        "source_episode_fingerprint",
        "variant_id",
        "parent_variant_id",
        "conversation",
        "tools",
        "provenance",
        "annotations",
        "quality",
    }
    if tokens[0] not in known_roots:
        raise FieldPolicyError(f"uncovered textual path: {pointer}")
    return FieldAction.COPY_EXACT


def _resolve_called_tool_schema(episode: CanonicalEpisode, tool_name: str) -> JsonValue:
    matches = [
        tool.function.parameters for tool in episode.tools if tool.function.name == tool_name
    ]
    if len(matches) != 1:
        raise FieldPolicyError(f"argument policy cannot resolve exactly one tool: {tool_name}")
    return matches[0]


def _validate_translation_action(
    episode: CanonicalEpisode,
    document: dict[str, JsonValue],
    pointer: str,
    source_text: str,
) -> None:
    argument = _argument_context(document, _pointer_tokens(pointer))
    if argument is None:
        return
    tool_name, argument_pointer, _ = argument
    schema = _schema_for_argument_pointer(
        _resolve_called_tool_schema(episode, tool_name), argument_pointer
    )
    _argument_translation_is_safe(source_text, schema, pointer)


def _segment_for(
    episode: CanonicalEpisode, pointer: str, source_text: str, policy: FieldPolicy
) -> Segment:
    source_sha256 = sha256_jcs(source_text)
    return Segment(
        segment_id=stable_id(
            "seg",
            {
                "episode_id": episode.episode_id,
                "json_pointer": pointer,
                "source_sha256": source_sha256,
                "field_policy_version": policy.policy_version,
            },
        ),
        episode_id=episode.episode_id,
        json_pointer=pointer,
        source_text=source_text,
        source_sha256=source_sha256,
        field_policy_version=policy.policy_version,
    )


def extract_leaf_segments(episode: CanonicalEpisode, policy: FieldPolicy) -> SegmentExtraction:
    """Extract every and only explicitly translatable string leaf.

    Encountering an unresolved policy is an error, rather than a partial
    extraction.  This prevents an output from being labelled translated when a
    textual field or a function argument silently escaped review.
    """
    document = cast(dict[str, JsonValue], episode.model_dump(mode="json", exclude_none=False))
    segments: list[Segment] = []
    for pointer, value in _leaf_values(document):
        if not isinstance(value, str):
            continue
        action = _classify_text_leaf(document, pointer, policy)
        if action is FieldAction.TRANSLATE:
            _validate_translation_action(episode, document, pointer, value)
            segments.append(_segment_for(episode, pointer, value, policy))
        elif action in {
            FieldAction.TRANSLATE_IF_NATURAL_LANGUAGE,
            FieldAction.MANUAL_POLICY_REQUIRED,
        }:
            raise FieldPolicyError(f"unresolved field policy {action.value}: {pointer}")
        elif action not in {FieldAction.COPY_EXACT, FieldAction.OMIT_FROM_MODEL_INPUT}:
            raise FieldPolicyError(f"uncovered field policy action: {action.value}")
    return SegmentExtraction(
        episode_id=episode.episode_id,
        input_variant_id=episode.variant_id,
        field_policy_version=policy.policy_version,
        segments=segments,
    )


def _renormalize_tools(document: dict[str, JsonValue]) -> None:
    tools = document.get("tools")
    if not isinstance(tools, list):
        raise FieldPolicyError("canonical tools is not a list")
    normalized: list[JsonValue] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise FieldPolicyError("canonical tool is not an object")
        raw = RawToolDefinition.model_validate(
            {"type": tool.get("type"), "function": tool.get("function")}, strict=True
        )
        normalized.append(normalize_tool(raw).canonical.model_dump(mode="json", exclude_none=False))
    document["tools"] = normalized


def _variant_id(document: dict[str, JsonValue]) -> str:
    episode_id = document.get("episode_id")
    conversation = document.get("conversation")
    tools = document.get("tools")
    annotations = document.get("annotations")
    if (
        not isinstance(episode_id, str)
        or not isinstance(conversation, list)
        or not isinstance(tools, list)
    ):
        raise FieldPolicyError("canonical identity fields have invalid shapes")
    if not isinstance(annotations, dict):
        raise FieldPolicyError("canonical annotations has an invalid shape")
    return sha256_jcs(
        {
            "episode_id": episode_id,
            "conversation": conversation,
            "tools": tools,
            "annotations": annotations,
        }
    )


def _assert_preservation(
    source: CanonicalEpisode, merged: CanonicalEpisode, segments: Iterable[Segment]
) -> None:
    source_json = cast(JsonValue, source.model_dump(mode="json", exclude_none=False))
    merged_json = cast(JsonValue, merged.model_dump(mode="json", exclude_none=False))
    source_leaves = dict(_leaf_values(source_json))
    merged_leaves = dict(_leaf_values(merged_json))
    if set(source_leaves) != set(merged_leaves):
        raise FieldPolicyError("host merge changed the canonical tree shape")
    mutable = {segment.json_pointer for segment in segments}
    derived = {"/variant_id", "/parent_variant_id"}
    for index in range(len(source.tools)):
        derived.update(
            {
                f"/tools/{index}/raw_schema_hash",
                f"/tools/{index}/documentation_hash",
            }
        )
    for pointer, source_value in source_leaves.items():
        if pointer in mutable or pointer in derived:
            continue
        if source_value != merged_leaves[pointer]:
            raise FieldPolicyError(f"technical field changed during host merge: {pointer}")
    for source_tool, merged_tool in zip(source.tools, merged.tools, strict=True):
        if (
            source_tool.tool_id != merged_tool.tool_id
            or source_tool.semantic_schema_hash != merged_tool.semantic_schema_hash
            or source_tool.normalizer_version != merged_tool.normalizer_version
        ):
            raise FieldPolicyError("host merge changed a structural tool identity")


def merge_translated_segments(
    episode: CanonicalEpisode,
    policy: FieldPolicy,
    extraction: SegmentExtraction,
    translations: Iterable[SegmentTranslation],
) -> CanonicalEpisode:
    """Apply a complete translation response and rebuild only derived hashes.

    No caller-supplied JSON tree is accepted.  The merge begins from the
    canonical input, writes only approved leaf pointers, then re-normalizes
    translated documentation and recomputes the variant identity locally.
    """
    expected = extract_leaf_segments(episode, policy)
    if extraction != expected:
        raise FieldPolicyError(
            "segment extraction does not exactly match the canonical input and policy"
        )
    by_id: dict[str, SegmentTranslation] = {}
    for translation in translations:
        if translation.segment_id in by_id:
            raise FieldPolicyError(f"duplicate translation segment ID: {translation.segment_id}")
        by_id[translation.segment_id] = translation
    expected_ids = {segment.segment_id for segment in expected.segments}
    if set(by_id) != expected_ids:
        raise FieldPolicyError("translation response must cover exactly the extracted segment IDs")

    document = cast(dict[str, JsonValue], episode.model_dump(mode="json", exclude_none=False))
    for segment in expected.segments:
        _set_pointer(document, segment.json_pointer, by_id[segment.segment_id].target_text)
    _renormalize_tools(document)
    new_variant_id = _variant_id(document)
    if new_variant_id == episode.variant_id:
        # A fully identical response is not a new child variant.
        return episode
    document["parent_variant_id"] = episode.variant_id
    document["variant_id"] = new_variant_id
    merged = CanonicalEpisode.model_validate_json(canonical_bytes(document), strict=True)
    _assert_preservation(episode, merged, expected.segments)
    return merged


def load_field_policy(path: Path) -> FieldPolicy:
    """Load a versioned policy through a JSON boundary to retain strict enums."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return FieldPolicy.model_validate_json(canonical_bytes(payload), strict=True)
