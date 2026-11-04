"""Import only explicitly requested release credential roles. Never source a dotenv."""

import base64
import re
import sys

from .config import DNS, Error, check
from .private_io import invoke, json_bytes, json_object, read_file
from .workloads import PROVIDER

ROLE_KEYS = {"providers": PROVIDER, "hub": ("HF_TOKEN",)}
DOTENV_LIMIT = 128 * 1024
MANAGED = "pmmc-release-import"
LABEL = "app.kubernetes.io/managed-by"


def select_roles(roles):
    check(
        isinstance(roles, (list, tuple))
        and roles
        and all(r in ROLE_KEYS for r in roles)
        and len(set(roles)) == len(roles),
        "select each required credential role once",
    )
    return {key for role in roles for key in ROLE_KEYS[role]}


def validate_values(values, roles):
    keys = select_roles(roles)
    check(
        isinstance(values, dict)
        and values.keys() <= keys
        and all(values.keys() & set(ROLE_KEYS[role]) for role in roles),
        "missing or unexpected credential field",
    )
    for value in values.values():
        check(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_./+=:@~-]{1,4096}", value),
            "unsupported or empty credential value",
        )


def parse_dotenv(raw, roles=None):
    keys = select_roles(roles if roles is not None else tuple(ROLE_KEYS))
    check(len(raw) <= DOTENV_LIMIT, "dotenv input exceeds size limit")
    try:
        lines = raw.decode("utf-8").split("\n")
    except UnicodeError:
        raise Error("dotenv must be UTF-8") from None
    values = {}
    for line in lines:
        line = line.strip(" \t\r")
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*(.*)", line)
        check(m, "invalid dotenv assignment; multiline values and shell commands are unsupported")
        key, raw_value = m.groups()
        if raw_value.startswith(("'", '"')):
            quote = raw_value[0]
            end = raw_value.find(quote, 1)
            check(
                end > 0 and re.fullmatch(r"[ \t]*(?:#.*)?", raw_value[end + 1 :]),
                "invalid quoted dotenv value",
            )
            value = raw_value[1:end]
        else:
            value = re.split(r"[ \t]+#", raw_value, maxsplit=1)[0].rstrip()
            check(not value.endswith("\\"), "dotenv continuation is unsupported")
        if key in keys:
            check(key not in values, "duplicate selected credential")
            values[key] = value
    validate_values(values, roles if roles is not None else present_roles(values))
    return values


def present_roles(values):
    return [role for role, keys in ROLE_KEYS.items() if values.keys() & set(keys)]


def read_dotenv(source, roles=None):
    check(source is not None, "explicit dotenv file or stdin required")
    raw = (
        sys.stdin.buffer.read(DOTENV_LIMIT + 1)
        if str(source) == "-"
        else read_file(source, DOTENV_LIMIT, private=True)
    )
    return parse_dotenv(raw, roles)


def validate_target(context, namespace, instance):
    check(
        isinstance(context, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,252}", context),
        "invalid Kubernetes context",
    )
    for name, maximum in ((namespace, 63), (instance, 20)):
        check(
            isinstance(name, str) and len(name) <= maximum and re.fullmatch(DNS, name),
            "invalid Kubernetes namespace or instance",
        )


def import_values(values, roles, *, context, namespace, instance, rotate=False):
    validate_target(context, namespace, instance)
    validate_values(values, roles)
    prefix = ["kubectl", "--context", context, "--namespace", namespace, "--request-timeout=30s"]
    actions = []
    status = {}
    # Preflight both roles before changing either; Kubernetes has no cross-Secret transaction.
    for role in roles:
        name = f"{instance}-{role}"
        desired = {
            k: base64.b64encode(values[k].encode()).decode() for k in ROLE_KEYS[role] if k in values
        }
        existing = invoke(
            [*prefix, "get", "secret", name, "--ignore-not-found=true", "-o", "json"], kube=True
        )
        if not existing.strip():
            status[role] = "created"
            body = {
                "apiVersion": "v1",
                "kind": "Secret",
                "type": "Opaque",
                "metadata": {"name": name, "namespace": namespace, "labels": {LABEL: MANAGED}},
                "data": desired,
            }
            actions.append(([*prefix, "create", "-f", "-"], body))
            continue
        obj = json_object(existing)
        meta = obj.get("metadata")
        check(
            obj.get("kind") == "Secret"
            and obj.get("type") == "Opaque"
            and isinstance(meta, dict)
            and meta.get("namespace") == namespace
            and meta.get("name") == name
            and isinstance(obj.get("data"), dict)
            and obj["data"].keys() == desired.keys(),
            "existing Secret has an incompatible identity, type or key set",
        )
        if obj["data"] == desired:
            status[role] = "unchanged"
            continue
        check(rotate, "credential drift; explicit rotation required")
        check(
            isinstance(meta.get("labels"), dict)
            and meta["labels"].get(LABEL) == MANAGED
            and not obj.get("immutable")
            and isinstance(meta.get("resourceVersion"), str)
            and meta["resourceVersion"],
            "refusing to rotate unmanaged or immutable Secret",
        )
        status[role] = "rotated"
        patch = [
            {"op": "test", "path": "/metadata/resourceVersion", "value": meta["resourceVersion"]},
            {"op": "replace", "path": "/data", "value": desired},
        ]
        actions.append(
            ([*prefix, "patch", "secret", name, "--type=json", "--patch-file=/dev/stdin"], patch)
        )
    for argv, body in actions:
        invoke(argv, json_bytes(body), kube=True)
    return status
