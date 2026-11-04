"""Validate and import only the two reviewed pricing files, never an approval."""

import base64
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

from .config import RATE_FILES, SECRET_RATE_PATH, Error, check
from .private_io import invoke, json_bytes, json_object, sha256
from .private_secrets import validate_target

CARD, REVIEW = RATE_FILES
RATE_PATH = SECRET_RATE_PATH
POLICY_KEYS = {CARD, REVIEW}
POLICY_LIMIT = 64 << 10
ABSENT = "not bundled; paid work awaits an independently supplied policy"
PRICE_KEYS = {
    "input_usd_per_million",
    "output_usd_per_million",
    "cached_input_usd_per_million",
    "cache_creation_input_usd_per_million",
}
LIMIT_KEYS = {
    "provider",
    "account_scope",
    "model_scope",
    "requests_per_minute",
    "tokens_per_minute",
}


def reviewed_yaml(raw, *, registry=False):
    # PyYAML is needed only for a pack that includes pricing files.
    try:
        import yaml
    except ImportError:
        raise Error("reviewed pricing inputs require PyYAML on the host") from None

    class StrictLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            check(
                not any(k.tag == "tag:yaml.org,2002:merge" for k, _ in node.value),
                "pricing YAML merges are unsupported",
            )
            result = super().construct_mapping(node, deep)
            check(len(result) == len(node.value), "duplicate pricing YAML key")
            return result

    try:
        text = raw.decode("utf-8")
        if not registry:
            check("#" not in text, "use a comment-free rate card reviewed at its exact byte hash")
        depth = 0
        for n, event in enumerate(yaml.parse(text, Loader=StrictLoader)):
            check(
                n < 8192 and not getattr(event, "anchor", None) and not getattr(event, "tag", None),
                "unsupported pricing YAML structure",
            )
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
                check(depth <= 16, "pricing YAML nesting exceeds limit")
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
        result = yaml.load(text, Loader=StrictLoader)
    except (UnicodeError, yaml.YAMLError, ValueError, TypeError, RecursionError, OverflowError):
        raise Error("invalid pricing YAML") from None
    check(isinstance(result, dict), "invalid pricing YAML object")
    return result


def validate_policy(pair, cfg, files):
    check(
        isinstance(pair, dict)
        and pair.keys() == POLICY_KEYS
        and all(isinstance(b, bytes) and 0 < len(b) <= POLICY_LIMIT for b in pair.values()),
        "expected exactly two bounded reviewed pricing files",
    )
    check(
        cfg["native"].get("rate_card") == RATE_PATH and cfg["billing"]["policy_secret"] is True,
        "bundled billing policy requires the configured pricing Secret mount",
    )
    path = cfg["native"]["model_registry"]
    registry = files.get(path.removeprefix("/workspace/"))
    check(
        path.startswith("/workspace/configs/") and registry and len(registry) <= 1 << 20,
        "pricing review requires a model registry from the verified source",
    )
    template = files.get("configs/release/prepare-models.yaml")
    check(
        template
        and len(template) <= POLICY_LIMIT
        and reviewed_yaml(template, registry=True).get("model_registry")
        == path.removeprefix("/workspace/"),
        "pricing registry differs from the public exploration campaign template",
    )
    review = json_object(pair[REVIEW])
    check(
        review.keys()
        == {
            "protocol",
            "currency",
            "pricing_basis",
            "reviewed_by",
            "reviewed_date",
            "rate_card_file",
            "rate_card_sha256",
            "model_registry_sha256",
            "pricing_sources",
        }
        and review["protocol"] == "pmmc-release-rate-review-v1"
        and review["currency"] == "USD"
        and review["pricing_basis"] == "standard_global_first_party_synchronous_before_tax"
        and review["rate_card_file"] == CARD
        and review["rate_card_sha256"] == sha256(pair[CARD])
        and review["model_registry_sha256"] == sha256(registry),
        "pricing review contract or exact card/registry checksum mismatch",
    )
    check(
        isinstance(review["reviewed_by"], str)
        and re.fullmatch(r"[\w .@+-]{1,128}", review["reviewed_by"])
        and review["reviewed_by"].strip(),
        "invalid rate reviewer",
    )
    try:
        when = date.fromisoformat(review["reviewed_date"])
        check(when.isoformat() == review["reviewed_date"], "invalid pricing review date")
    except (TypeError, ValueError):
        raise Error("invalid pricing review date") from None

    card = reviewed_yaml(pair[CARD])
    check(
        card.keys() == {"estimated", "models", "rate_limits"} and card["estimated"] is False,
        "estimated, incomplete or unexpected rate card fields",
    )
    prices, limits, sources = card["models"], card["rate_limits"], review["pricing_sources"]
    check(
        all(isinstance(x, dict) for x in (prices, limits, sources))
        and prices
        and prices.keys() == limits.keys() == sources.keys(),
        "review must cover exactly the priced models",
    )
    entries = reviewed_yaml(registry, registry=True).get("models")
    check(isinstance(entries, list), "invalid reviewed release registry")
    providers = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("provider_config"), dict):
            provider = entry["provider_config"].get("kind")
            if provider in ("openai", "anthropic") and isinstance(entry.get("provider_model"), str):
                providers[entry["provider_model"]] = provider
    scopes = {}
    for model, rates in prices.items():
        check(
            model in providers and isinstance(rates, dict) and rates.keys() == PRICE_KEYS,
            "unregistered model or unexpected price fields",
        )
        try:
            numbers = [Decimal(str(value)) for value in rates.values()]
            check(
                all(n.is_finite() and n >= 0 for n in numbers) and any(numbers),
                "paid model requires positive reviewed pricing",
            )
        except (InvalidOperation, ValueError, TypeError):
            raise Error("invalid model price") from None
        policy = limits[model]
        check(
            isinstance(policy, dict)
            and policy.keys() == LIMIT_KEYS
            and policy["provider"] == providers[model]
            and isinstance(policy["account_scope"], str)
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", policy["account_scope"])
            and policy["model_scope"] in (model, "*")
            and type(policy["requests_per_minute"]) is int
            and policy["requests_per_minute"] > 0
            and type(policy["tokens_per_minute"]) is int
            and policy["tokens_per_minute"] >= 32768,
            "invalid reviewed rate limits",
        )
        scope = (providers[model], policy["account_scope"], policy["model_scope"])
        caps = (policy["requests_per_minute"], policy["tokens_per_minute"])
        check(scopes.setdefault(scope, caps) == caps, "conflicting limits for one provider scope")
        url = sources[model]
        check(
            isinstance(url, str) and 0 < len(url) <= 2048 and not re.search(r"[\s\\?#]", url),
            "invalid pricing source URL",
        )
        try:
            parts = urlsplit(url)
            check(
                parts.scheme == "https"
                and parts.hostname
                and parts.username is None
                and parts.password is None,
                "expected credential-free HTTPS pricing source",
            )
            _ = parts.port  # Validate explicit port syntax.
        except ValueError:
            raise Error("invalid pricing source URL") from None


def import_policy(cfg, pair):
    check(pair is not None, ABSENT)
    check(
        cfg["profile"] == "kubernetes" and pair.keys() == POLICY_KEYS,
        "invalid pricing Secret input",
    )
    context, ns = cfg["cluster"]["context"], cfg["cluster"]["namespace"]
    validate_target(context, ns, cfg["instance"])
    name = cfg["instance"] + "-rates"
    prefix = ["kubectl", "--context", context, "--namespace", ns, "--request-timeout=30s"]
    data = {k: base64.b64encode(v).decode() for k, v in pair.items()}
    existing = invoke(
        [*prefix, "get", "secret", name, "--ignore-not-found=true", "-o", "json"], kube=True
    )
    if existing.strip():
        obj = json_object(existing)
        meta = obj.get("metadata")
        check(
            obj.get("apiVersion") == "v1"
            and obj.get("kind") == "Secret"
            and obj.get("type") == "Opaque"
            and isinstance(meta, dict)
            and meta.get("name") == name
            and meta.get("namespace") == ns
            and obj.get("immutable") is True
            and obj.get("data") == data,
            "existing pricing Secret differs; refusing replacement",
        )
        return {"pricing_policy": "unchanged"}
    invoke(
        [*prefix, "create", "-f", "-"],
        json_bytes(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": name, "namespace": ns},
                "type": "Opaque",
                "immutable": True,
                "data": data,
            }
        ),
        kube=True,
    )
    return {"pricing_policy": "created"}
