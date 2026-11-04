"""Resource-independent exploration intent, shared by the dependency-free host and workers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

BENCHMARKS = ("memlens", "memgallery")
MODELS = ("answer-qwen3.5-9b", "answer-qwen3.5-27b", "answer-gpt-6-luna", "answer-claude-haiku-5.5")
METHODS = (
    "full_context_captions",
    "full_context_raw_images",
    "caption_text_rag",
    "contrastive_multimodal_rag",
    "native_multimodal_rag",
)
VARIANTS = ("wr", "w", "r", "legacy")
ANALYSES = ("coverage", "cost", "failures", "ablation")
COMPONENTS = ("compile", "index", "match", "execute", "answer", "ground", "select", "score", "seal")
STAGES = ("main", "analysis", "baselines", "all")


def identifier(name: object) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}", name):
        raise ValueError("invalid exploration run identifier")
    return name


def normalize_plan(raw: Mapping) -> dict:
    known = {
        "run_id",
        "stage",
        "benchmarks",
        "models",
        "methods",
        "variants",
        "analyses",
        "components",
        "from_run",
        "with_deps",
    }
    if not isinstance(raw, Mapping) or raw.keys() - known:
        raise ValueError("invalid exploration plan fields")
    run_id, stage = identifier(raw.get("run_id")), raw.get("stage") or "main"
    if stage not in STAGES:
        raise ValueError("unknown exploration stage")

    def selected(key, choices, defaults):
        value = raw.get(key)
        value = defaults if value is None else value
        if (
            not isinstance(value, (tuple, list))
            or any(x not in choices for x in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(f"unknown or duplicated exploration {key}")
        return [x for x in choices if x in value]

    baselines, analysis = stage in ("baselines", "all"), stage in ("analysis", "all")
    analyses = selected("analyses", ANALYSES, ANALYSES if analysis else [])
    methods = selected("methods", METHODS, METHODS if baselines else [])
    variants = selected("variants", VARIANTS, VARIANTS if "ablation" in analyses else ("wr",))
    components = selected("components", COMPONENTS, [])
    benchmarks, models = (
        selected("benchmarks", BENCHMARKS, BENCHMARKS),
        selected("models", MODELS, MODELS),
    )
    source = raw.get("from_run")
    deps = raw.get("with_deps", False)
    if (
        not benchmarks
        or not models
        or not variants
        or bool(methods) != baselines
        or bool(analyses) != analysis
        or type(deps) is not bool
        or (source is not None and (identifier(source) == run_id or stage != "analysis"))
        or (components and stage not in ("main", "baselines"))
        or (stage == "baselines" and variants != ["wr"])
        or (
            stage == "baselines"
            and set(components) - {"index", "answer", "ground", "score", "seal"}
        )
        or (stage == "all" and "wr" not in variants)
    ):
        raise ValueError("incompatible or empty experiment scope")
    return dict(
        run_id=run_id,
        stage=stage,
        benchmarks=benchmarks,
        models=models,
        methods=methods,
        variants=variants,
        analyses=analyses,
        components=components,
        from_run=source,
        with_deps=deps,
    )


def plan_hash(plan: Mapping) -> str:
    data = json.dumps(
        normalize_plan(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(data.encode()).hexdigest()


def plan_from_environment(environ: Mapping[str, str]) -> dict | None:
    raw = environ.get("PMMC_RELEASE_REQUEST")
    return normalize_plan(json.loads(raw)) if raw is not None else None
