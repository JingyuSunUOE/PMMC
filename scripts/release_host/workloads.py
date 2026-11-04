"""Backend-neutral command and privilege boundary for each release workload."""

import secrets
from pathlib import Path

from .config import check
from .quota import DEFAULTS, GPU_ROLES

WORKER = (
    "PMMC_DATABASE_URL",
    "PMMC_DATABASE_EVALUATOR_URL",
    "PMMC_DATABASE_ONLINE_URL",
    "PMMC_OBJECT_STORE_ACCESS_KEY",
    "PMMC_OBJECT_STORE_SECRET_KEY",
    "PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY",
    "PMMC_OBJECT_STORE_EVALUATOR_SECRET_KEY",
)
ADMIN = "PMMC_DATABASE_ADMIN_URL"
ROOT_KEYS = ("PMMC_MINIO_ROOT_USER", "PMMC_MINIO_ROOT_PASSWORD")
IAM_KEYS = (*ROOT_KEYS, *WORKER[3:])
PROVIDER = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
BENCH = ["python", "-m", "pmmc", "benchmark"]
# Execute in the verified app image (no host package installation or credentials).
# A saved budget changes native job selection even if the host passes no keys.
VERIFY_POINTER = """import hashlib, json, os, sys
from pathlib import Path
from pmmc.infrastructure.settings import get_settings
from pmmc.experiments.release_inputs import exploration_campaign_directory
settings = get_settings()
if os.environ.get("PMMC_RELEASE_REQUEST"):
    from pmmc.experiments.release_reporting import verify_release_budget_pointer
    rate = os.environ.get("PMMC_STEP8_RATE_CARD_PATH")
    verify_release_budget_pointer(settings, run_id=sys.argv[1], variant=sys.argv[2],
        stage=sys.argv[3], approval_id=os.getenv("PMMC_STEP8_BUDGET_APPROVAL_ID"),
        rate_card_path=Path(rate) if rate else None)
    sys.exit(0)
directory = exploration_campaign_directory(settings, sys.argv[1], variant=sys.argv[2])
pointer = settings.outputs_path / "step8/exploration" / sys.argv[1] / sys.argv[2] / "pointer.json"
selection = json.loads(pointer.read_bytes())["selection_hash"]
manifests = directory / "manifests"
saved = manifests / f"budget-{selection}.json"
approval = os.getenv("PMMC_STEP8_BUDGET_APPROVAL_ID")
if approval or saved.exists():
    receipt = json.loads(saved.read_bytes())
    archived = manifests / f"rate-card-{selection}.json"
    configured = Path(os.getenv("PMMC_STEP8_RATE_CARD_PATH", str(archived)))
    if not (approval and receipt["approval_id"] == approval
            and receipt["selection_hash"] == selection
            and hashlib.sha256(archived.read_bytes()).hexdigest() == receipt["rate_card_hash"]
            and hashlib.sha256(configured.read_bytes()).hexdigest() == receipt["rate_card_hash"]):
        sys.exit("selection budget or rate-card mismatch")
"""
RESOURCES = DEFAULTS


def cold(role, argv):
    return ["python", "-c", Path(__file__).with_name("cold_start.py").read_text(), role, *argv]


def credentials(name):
    def token():
        return secrets.token_hex(24)

    values = {
        "PMMC_POSTGRES_ADMIN_PASSWORD": token(),
        **{key: token() for key in IAM_KEYS},
    }
    for key, user, pw in [
        (ADMIN, "pmmc_owner", values["PMMC_POSTGRES_ADMIN_PASSWORD"]),
        (WORKER[0], "pmmc_runtime", token()),
        (WORKER[1], "pmmc_evaluator", token()),
        (WORKER[2], "pmmc_online_worker", token()),
    ]:
        values[key] = f"postgresql+psycopg://{user}:{pw}@{name}-postgres:5432/pmmc"
    return values


def secret_refs(role, cfg, approval, plan=None):
    if role == "postgres":
        return {"POSTGRES_PASSWORD": "PMMC_POSTGRES_ADMIN_PASSWORD"}
    if role == "minio":
        return dict(zip(("MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"), ROOT_KEYS, strict=True))
    keys = list(WORKER) if role in ("bootstrap", "control", *GPU_ROLES, "prepare-capacity") else []
    if role in ("quote", "approve", "stop-budget", "spend"):
        keys.append(WORKER[0])
    if role in ("bootstrap", "control"):
        keys.append(ADMIN)
    if role == "control" and approval:
        check(
            isinstance(plan, dict) and isinstance(plan.get("models"), list) and plan["models"],
            "approved control requires an explicit model selection",
        )
        for model, key in (
            ("answer-gpt-6-luna", PROVIDER[0]),
            ("answer-claude-haiku-5.5", PROVIDER[1]),
        ):
            if model in plan["models"]:
                keys.append(key)
    if role == "models" and cfg["models"]["use_hf_token"]:
        keys.append("HF_TOKEN")
    refs = {k: k for k in keys}
    if role == "bootstrap":
        refs.update(
            zip(
                ("PMMC_OBJECT_STORE_ADMIN_ACCESS_KEY", "PMMC_OBJECT_STORE_ADMIN_SECRET_KEY"),
                ROOT_KEYS,
                strict=True,
            )
        )
    return refs


def environment(role, cfg, name, approval):
    endpoint = f"http://{name}-minio:9000"
    base = {
        "PMMC_OBJECT_STORE_ENDPOINT": endpoint,
        "PMMC_OBJECT_STORE_BUCKET": "pmmc-runtime",
        "PMMC_OBJECT_STORE_EVALUATION_BUCKET": "pmmc-evaluation",
    }
    if role == "iam":
        return base
    native = cfg["native"]
    base.update(
        {
            "PMMC_PROJECT_ROOT": "/workspace",
            "PMMC_BENCHMARK_ASSET_ROOT": "/pmmc-inputs",
            "PMMC_ARTIFACTS_DIR": "artifacts",
            "PMMC_OUTPUTS_DIR": "outputs",
            "PMMC_STEP8_OUTPUT_ROOT": "/workspace/outputs/step8",
            "PMMC_RUNS_DIR": "outputs/step8/runtime",
            "PMMC_OBJECT_STORE_MODEL_ENDPOINT": endpoint,
            "PMMC_OBJECT_STORE_SECURE": "false",
            "PMMC_OBJECT_STORE_PATH_STYLE": "true",
            "PMMC_STEP8_CONFIG": native["benchmark_config"],
            "PMMC_MODEL_REGISTRY_PATH": native["model_registry"],
            "PMMC_MODEL_LOCK_PATH": native["model_lock"],
            "PMMC_RUNTIME_MODEL_LOCK_PATH": native["model_lock"],
            "PMMC_RUNTIME_PROFILE_PATH": native["runtime_profile"],
            "PMMC_RUNTIME_CALIBRATION_DIR": "/workspace/artifacts/runtime/calibrations/release/"
            + cfg["images"]["app"].split("@sha256:")[1],
            "PMMC_MODEL_VERIFICATION_POLICY": "cached_receipt",
            "PMMC_MODEL_CACHE_DIR": "/pmmc-cache",
            "HF_HOME": "/pmmc-cache",
            "HF_DATASETS_CACHE": "/pmmc-cache/datasets",
            "PMMC_CONTAINER_IMAGE": cfg["images"]["app"],
            "PMMC_RUNTIME_PORT_START": "8100",
            "PMMC_RUNTIME_PORT_END": "8599",
            "PMMC_STEP8_REQUIRE_SELECTION_APPROVAL": "1",
            "PYTHONPATH": "/workspace/src",
        }
    )
    if role not in GPU_ROLES:
        base["NVIDIA_VISIBLE_DEVICES"] = "void"
    if "release_profile" in native:
        base["PMMC_RELEASE_PROFILE"] = native["release_profile"]
    if role in ("pointer", "control", "gpu") and approval:
        base["PMMC_STEP8_BUDGET_APPROVAL_ID"] = approval
        if "rate_card" in native:
            base["PMMC_STEP8_RATE_CARD_PATH"] = native["rate_card"]
    if role in GPU_ROLES:
        count = cfg["topology"]["gpus"]
        if not count:
            from .config import Error

            raise Error("GPU phase requires a positive single-node GPU allocation")
        threads = str(max(1, (cfg["resources"][role]["cpu"] - 2) // count))
        base.update(
            {
                "PMMC_RUNTIME_MODE": "managed_local",
                "PMMC_EXPECTED_GPU_COUNT": str(count),
                "PMMC_GPU_MEMORY_ACCOUNTING": "allocated_device",
                "PMMC_MODEL_CPU_THREADS_PER_REPLICA": threads,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "PMMC_MODEL_FOLLOWER_START_BATCH_SIZE": str(min(4, count)),
                "PMMC_LOCAL_SCRATCH_DIR": "/pmmc-scratch",
                "PMMC_STAGE_MODEL_SNAPSHOTS_TO_LOCAL": "1",
                "PMMC_TOKEN_COUNTER_CACHE_DIR": "/pmmc-scratch/token-counter",
            }
        )
    if role != "models" or not cfg["models"]["allow_download"]:
        base["HF_HUB_OFFLINE"] = "1"
    return base


def command(role, cfg, run, stage, request=None):
    request = request or {"variant": "wr"}
    variant = request["variant"]
    select = ["--run-id", run, "--variant", variant]
    if role == "iam":
        return ["python", "/opt/pmmc-step8/step8_iam.py"]
    if role == "assets":
        return ["python", "-m", "pmmc.experiments.release_assets"]
    if role == "analysis-report":
        return ["python", "-m", "pmmc.experiments.release_readouts"]
    if role in ("spend", "stop-budget"):
        if role == "spend":
            return [*BENCH, "spend-exploration", *select]
        return [
            *BENCH,
            "stop-exploration-budget",
            *select,
            "--approval-id",
            request["approval_id"],
            "--plan-hash",
            request["plan_hash"],
            "--selection-hash",
            request["selection_hash"],
            "--operator-id",
            request["operator_id"],
            "--reason-code",
            request["reason_code"],
        ]
    if role == "prepare-capacity":
        return cold(
            "gpu", ["python", "-c", 'print(\'{"protocol":"release-capacity-v2","ready":true}\')']
        )
    if role == "bootstrap":
        return [*BENCH, "bootstrap"]
    if role == "models":
        return cold(
            role,
            [
                *BENCH,
                "prepare-models",
                "--config",
                cfg["native"]["benchmark_config"],
                *(["--allow-download"] if cfg["models"]["allow_download"] else []),
            ],
        )
    if role == "prepare":
        if request.get("kind") in {"quote", "approve", "prepare"} and request.get("plan"):
            return ["python", "-m", "pmmc.experiments.release_prepare_variants", run, variant]
        return [*BENCH, "prepare-exploration", *select]
    if role == "quote":
        return [
            *BENCH,
            "quote-exploration",
            *select,
            "--stage",
            stage,
            "--rate-card",
            request["rate_card"],
        ]
    if role == "approve":
        return [
            *BENCH,
            "approve-exploration",
            *select,
            "--stage",
            stage,
            "--selection-hash",
            request["selection_hash"],
            "--rate-card",
            request["rate_card"],
            "--approver-id",
            request["approver_id"],
            *(["--uncapped"] if request.get("uncapped") else ["--limit-usd", request["limit_usd"]]),
        ]
    if role == "pointer":
        return ["python", "-c", VERIFY_POINTER, run, variant, stage]
    argv = [
        *BENCH,
        "run-exploration",
        *select,
        "--stage",
        stage,
        "--worker-pool",
        "gpu" if role == "gpu" else "control",
    ]
    return (
        cold(role, argv)
        if role == "gpu" and (request.get("plan") or cfg["models"]["calibrate"])
        else argv
    )
