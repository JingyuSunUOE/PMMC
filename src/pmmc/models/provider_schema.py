"""Provider-only schema projections with mandatory canonical postvalidation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast

from jsonschema import FormatChecker
from jsonschema.validators import validator_for

from pmmc.models.gateways import _canonical_json_hash, project_vllm_response_format
from pmmc.models.provider_config import ProviderKind

_ANNOTATIONS = {"default", "examples", "title", "$schema"}
_UNSUPPORTED = {
    "uniqueItems",
    "discriminator",
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "minProperties",
    "maxProperties",
}
_CLAUDE_CONSTRAINTS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "pattern",
    "format",
}
_MAPS = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
_LISTS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SINGLES = {"items", "additionalProperties", "not", "if", "then", "else", "contains"}


def _schema_children(schema: dict[str, Any]):  # type: ignore[no-untyped-def]
    for key, value in schema.items():
        if key in _MAPS and isinstance(value, dict):
            yield from (child for child in value.values() if isinstance(child, dict))
        elif key in _LISTS and isinstance(value, list):
            yield from (child for child in value if isinstance(child, dict))
        elif key in _SINGLES and isinstance(value, dict):
            yield value


def canonical_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    envelope = response_format.get("json_schema", {})
    schema = envelope.get("schema") if isinstance(envelope, dict) else None
    if response_format.get("type") != "json_schema" or not isinstance(schema, dict):
        raise ValueError("Step 8 structured output requires a canonical json_schema")

    def check_refs(node: dict[str, Any]) -> None:
        if "$id" in node or any(key in node for key in ("$dynamicRef", "$recursiveRef")):
            raise ValueError("schema IDs and dynamic references are not supported")
        if "$ref" in node and not str(node["$ref"]).startswith("#"):
            raise ValueError("external schema references are not supported")
        for child in _schema_children(node):
            check_refs(child)

    check_refs(schema)
    validator_for(schema).check_schema(schema)
    return schema


def _root_object_union(schema: dict[str, Any]) -> dict[str, Any]:
    """Relax the answer's cross-field union, retaining its per-field alternatives."""
    branches = schema.get("anyOf", schema.get("oneOf"))
    if branches is None:
        return schema
    if not all(isinstance(b, dict) and b.get("type") == "object" for b in branches):
        raise ValueError("native strict JSON requires an object or object union root")
    if schema.get("properties"):
        raise ValueError("root object unions with sibling properties are not supported")
    properties: dict[str, Any] = {}
    for branch in branches:
        for name, child in branch.get("properties", {}).items():
            choices = properties.setdefault(name, {"anyOf": []})["anyOf"]
            if child not in choices:
                choices.append(child)
    required = set(branches[0].get("required", []))
    for branch in branches[1:]:
        required.intersection_update(branch.get("required", []))
    return {
        **{k: v for k, v in schema.items() if k not in {"anyOf", "oneOf"}},
        "type": "object",
        "properties": properties,
        "required": sorted(required),
        "additionalProperties": False,
    }


def project_provider_response_format(
    response_format: dict[str, Any],
    kind: ProviderKind,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = canonical_schema(response_format)
    removed: dict[str, int] = {}

    def remove(key: str) -> None:
        removed[key] = removed.get(key, 0) + 1

    def project(node: dict[str, Any], refs: tuple[str, ...] = ()) -> dict[str, Any]:
        node = deepcopy(node)
        if kind == "anthropic" and "$ref" in node:
            ref = node["$ref"]
            if ref in refs:
                raise ValueError("Anthropic strict JSON does not support recursive schemas")
            target: Any = source
            for token in ref.removeprefix("#/").split("/"):
                target = target[token.replace("~1", "/").replace("~0", "~")]
            return project(
                {**target, **{k: v for k, v in node.items() if k != "$ref"}}, (*refs, ref)
            )
        result: dict[str, Any] = {}
        for key, value in node.items():
            if (
                key in _ANNOTATIONS
                or key in _UNSUPPORTED
                or (kind == "anthropic" and key in _CLAUDE_CONSTRAINTS)
            ):
                remove(key)
                continue
            if key in {"allOf", "patternProperties", "prefixItems", "contains"}:
                raise ValueError(f"unsupported strict JSON schema construct: {key}")
            if key in {"properties", "$defs", "definitions"}:
                result[key] = {name: project(child, refs) for name, child in value.items()}
            elif key in {"anyOf", "oneOf"}:
                if key == "oneOf":
                    remove(key)
                result["anyOf"] = [project(child, refs) for child in value]
            elif key == "items" and isinstance(value, dict):
                result[key] = project(value, refs)
            elif key == "minItems" and kind == "anthropic" and value > 1:
                remove(key)
                result[key] = 1
            elif key == "const":
                result["enum"] = [value]
            else:
                result[key] = value
        if result.get("type") == "object" or "properties" in result:
            if result.get("additionalProperties") not in (None, False):
                raise ValueError("native strict JSON cannot represent open-ended objects")
            result["additionalProperties"] = False
            result.setdefault("properties", {})
            if kind == "openai":
                result["required"] = list(result["properties"])
        return result

    if kind == "qwen_vllm":
        result, diagnostics = project_vllm_response_format(deepcopy(response_format))
        transport = result["json_schema"]["schema"]
        removed = cast(dict[str, int], (diagnostics or {}).get("removed_keyword_counts", {}))
    else:
        root = _root_object_union(deepcopy(source))
        if root != source:
            remove("root_object_union")
        if root.get("type") != "object":
            raise ValueError("native strict JSON requires an object root")
        transport = project(root)
        result = deepcopy(response_format)
        result["json_schema"]["schema"] = transport
    result["json_schema"]["strict"] = True
    result["json_schema"].setdefault("name", "pmmc_response")
    return result, {
        "projection_version": f"step8-{kind}-schema-v1",
        "removed_keyword_counts": removed,
        "source_schema_sha256": _canonical_json_hash(source),
        "transport_schema_sha256": _canonical_json_hash(transport),
        "canonical_postvalidation_required": True,
    }


def validate_canonical_output(text: str, response_format: dict[str, Any]) -> None:
    schema = canonical_schema(response_format)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise ValueError("non-finite JSON value")

    payload = json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    validator_for(schema)(schema, format_checker=FormatChecker()).validate(payload)
