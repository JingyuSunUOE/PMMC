from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.infrastructure.database.step8_repositories import BudgetExceededError
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import create_generation_gateway
from pmmc.models.providers import ProviderCallError
from pmmc.models.registry import load_model_registry
from pmmc.models.step8_accounting import (
    SelectionApproval,
    SettingsAccountingGateway,
    Step8AccountingGateway,
    load_rate_card,
)
from pmmc.schemas import (
    GenerationRequest,
    GenerationResult,
    ModelContentPart,
    ModelMessage,
    TokenUsage,
)

ROOT = Path(__file__).parents[1]
MODEL = "gpt-6-luna"
APPROVAL_ID = "a" * 64
SELECTION = "b" * 64
RATE_HASH = "c" * 64


def registry():
    return load_model_registry(ROOT / "configs/models/step8.yaml")


def request():
    return GenerationRequest(
        request_id="paid-request-1",
        model_alias="answer-gpt-6-luna",
        max_output_tokens=512,
        provider_max_output_tokens=1024,
        messages=[ModelMessage(role="user", parts=[ModelContentPart(kind="text", text="fixture")])],
    )


def response():
    return GenerationResult(
        request_id=request().request_id,
        model_alias=request().model_alias,
        actual_model_id=MODEL,
        text="fixture response",
        provider_request_id="provider-1",
        finish_reason="stop",
        usage=TokenUsage(
            input_tokens=105,
            output_tokens=20,
            total_tokens=125,
            metadata={
                "usage_reported": True,
                "uncached_input_tokens": 70,
                "cached_input_tokens": 30,
                "cache_creation_input_tokens": 5,
                "reasoning_tokens": 5,
            },
        ),
    )


class Repository:
    def __init__(self):
        self.approval = SimpleNamespace(
            approval_id=APPROVAL_ID,
            campaign_id="campaign",
            selection_hash=SELECTION,
            rate_card_hash=RATE_HASH,
            approver_id="human-reviewer",
            limit_usd=Decimal("10"),
        )
        self.rows = {}
        self.events = []
        self.open_sessions = 0
        self.fail_settle = False

    @asynccontextmanager
    async def sessions(self):
        self.open_sessions += 1
        try:
            yield self
        finally:
            self.open_sessions -= 1

    async def get(self, model, identity):
        assert self.open_sessions
        return self.approval if self.approval and identity == self.approval.approval_id else None

    async def reserve(self, approval_id, request_id, request_hash, provider, worst_case_usd):
        self.events.append("reserve")
        identity = (approval_id, request_hash, provider, worst_case_usd)
        if request_id in self.rows:
            assert self.rows[request_id]["identity"] == identity
            return self.rows[request_id]["state"]
        if worst_case_usd > self.approval.limit_usd:
            raise BudgetExceededError("approved_api_budget_exhausted")
        self.rows[request_id] = {
            "identity": identity,
            "state": "reserved",
            "actual": None,
            "response_hash": None,
        }
        return "reserved"

    async def settle(self, request_id, actual_usd, response_hash):
        self.events.append("settle")
        if self.fail_settle:
            self.fail_settle = False
            raise RuntimeError("database unavailable")
        row = self.rows[request_id]
        if row["state"] == "settled":
            assert (row["actual"], row["response_hash"]) == (actual_usd, response_hash)
            return
        if actual_usd is not None and actual_usd > row["identity"][-1]:
            raise BudgetExceededError("provider_exceeded_frozen_worst_case")
        row.update(
            state="unknown" if actual_usd is None else "settled",
            actual=actual_usd,
            response_hash=response_hash,
        )


class Store(CampaignArtifactStore):
    def __init__(self, root, events):
        super().__init__(root)
        self.events = events
        self.fail_write = False

    def journal_response(self, **kwargs):
        self.events.append("journal")
        if self.fail_write:
            raise OSError("journal unavailable")
        return super().journal_response(**kwargs)

    def restored_response(self, **kwargs):
        self.events.append("read")
        return super().restored_response(**kwargs)


class Gateway:
    def __init__(self, repository):
        self.repository = repository
        self.calls = 0
        self.error = None
        self.result = response()

    async def generate(self, request):
        assert self.repository.open_sessions == 0
        self.repository.events.append("provider")
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class RateControl:
    async def try_admit(self, limit, **kwargs):
        return 0

    async def reconciliation(self, request_id, request_hash):
        return None


def setup(tmp_path):
    repo = Repository()
    gateway = Gateway(repo)
    store = Store(tmp_path, repo.events)
    wrapped = Step8AccountingGateway(
        gateway,
        repository=repo,
        artifact_store=store,
        registry=registry(),
        approval=SelectionApproval("campaign", SELECTION, APPROVAL_ID, RATE_HASH),
        attempt_id="attempt-1",
        worst_case_cost=lambda req, entry, budget: Decimal("1"),
        actual_cost=lambda result, entry: Decimal("0.1"),
    )
    return wrapped, repo, gateway, store


def test_reserve_dispatch_journal_readback_settle_before_caller_commit(tmp_path):
    wrapped, repo, gateway, store = setup(tmp_path)
    result = asyncio.run(wrapped.generate(request()))
    repo.events.append("authoritative_component_commit")
    assert repo.events == [
        "reserve",
        "read",
        "provider",
        "journal",
        "read",
        "settle",
        "authoritative_component_commit",
    ]
    audit = result.metadata["step8_accounting"]
    assert store.load_json(audit["response_sha256"])["result"]["text"] == response().text
    assert repo.rows[request().request_id]["actual"] == Decimal("0.1")
    replay = asyncio.run(wrapped.generate(request()))
    assert replay == result and gateway.calls == 1


@pytest.mark.parametrize("alias", ["answer-gpt-6-luna", "answer-claude-haiku-5.5"])
def test_native_context_rejection_precedes_reservation_and_dispatch_marker(tmp_path, alias):
    wrapped, repo, gateway, _ = setup(tmp_path)
    oversized = request().model_copy(update={"model_alias": alias})
    oversized.messages[0].parts[0].text = "x" * 32768

    def no_quote(*args):
        raise AssertionError("context validation must precede cost quotation")

    wrapped.worst_case_cost = no_quote
    with pytest.raises(ValueError, match="shared 32768"):
        asyncio.run(wrapped.generate(oversized))
    assert repo.events == [] and repo.rows == {} and gateway.calls == 0
    assert not list(tmp_path.rglob("intent.json"))


@pytest.mark.parametrize(
    "field", ["campaign_id", "selection_hash", "rate_card_hash", "approver_id"]
)
def test_every_selection_needs_matching_explicit_approval(tmp_path, field):
    wrapped, repo, gateway, _ = setup(tmp_path)
    setattr(repo.approval, field, "")
    with pytest.raises(ProviderCallError, match="ExplicitSelectionApprovalRequired"):
        asyncio.run(wrapped.generate(request()))
    assert not repo.rows and gateway.calls == 0


def test_missing_or_exhausted_approval_never_calls_provider(tmp_path):
    wrapped, repo, gateway, _ = setup(tmp_path)
    repo.approval.limit_usd = Decimal("0.01")
    with pytest.raises(BudgetExceededError):
        asyncio.run(wrapped.generate(request()))
    repo.approval = None
    with pytest.raises(ProviderCallError, match="ApprovalRequired"):
        asyncio.run(wrapped.generate(request()))
    assert gateway.calls == 0


@pytest.mark.parametrize(
    "failure",
    [
        ProviderCallError("APITimeoutError", outcome="unknown"),
        TimeoutError("private details"),
        RuntimeError("private details"),
    ],
)
def test_unknown_outcome_holds_reservation_and_never_redispatches(tmp_path, failure):
    wrapped, repo, gateway, _ = setup(tmp_path)
    gateway.error = failure
    for _ in range(2):
        with pytest.raises(ProviderCallError) as error:
            asyncio.run(wrapped.generate(request()))
        assert error.value.outcome == "unknown" and not error.value.retryable
    row = repo.rows[request().request_id]
    assert row["state"] == "unknown" and row["actual"] is None
    assert row["identity"][-1] == Decimal("1") and gateway.calls == 1


def test_confirmed_failure_settles_zero_but_is_not_automatically_resent(tmp_path):
    wrapped, repo, gateway, _ = setup(tmp_path)
    gateway.error = ProviderCallError("BadRequestError", outcome="failed", provider_status_code=400)
    for _ in range(2):
        with pytest.raises(ProviderCallError, match="BadRequestError"):
            asyncio.run(wrapped.generate(request()))
    assert repo.rows[request().request_id]["actual"] == Decimal(0)
    assert gateway.calls == 1


def test_invalid_completed_response_still_journals_and_settles_usage(tmp_path):
    wrapped, repo, gateway, _ = setup(tmp_path)
    gateway.error = ProviderCallError(
        "CanonicalSchemaViolation", outcome="completed", result=response()
    )
    with pytest.raises(ProviderCallError, match="CanonicalSchemaViolation"):
        asyncio.run(wrapped.generate(request()))
    assert repo.rows[request().request_id]["actual"] == Decimal("0.1")


def test_unknown_usage_holds_reservation_but_replays_known_response(tmp_path):
    wrapped, repo, gateway, _ = setup(tmp_path)
    gateway.result.usage.metadata["usage_reported"] = False
    first = asyncio.run(wrapped.generate(request()))
    assert first.metadata["step8_accounting"]["reservation_state"] == "unknown"
    assert asyncio.run(wrapped.generate(request())) == first
    assert repo.rows[request().request_id]["actual"] is None and gateway.calls == 1


def test_database_failure_after_response_recovers_journal_without_paid_retry(tmp_path):
    wrapped, repo, gateway, store = setup(tmp_path)
    repo.fail_settle = True
    with pytest.raises(ProviderCallError, match="SettlementNeedsReconciliation"):
        asyncio.run(wrapped.generate(request()))
    assert list((store.root / "_store/responses").glob("*.json"))
    result = asyncio.run(wrapped.generate(request()))
    assert result.text == response().text and gateway.calls == 1


def test_journal_failure_is_unknown_and_dispatch_marker_prevents_retry(tmp_path):
    wrapped, repo, gateway, store = setup(tmp_path)
    store.fail_write = True
    with pytest.raises(ProviderCallError, match="ResponseJournalUnavailable"):
        asyncio.run(wrapped.generate(request()))
    store.fail_write = False
    with pytest.raises(ProviderCallError, match="NeedsReconciliation"):
        asyncio.run(wrapped.generate(request()))
    assert repo.rows[request().request_id]["actual"] is None and gateway.calls == 1


def test_hard_crash_after_dispatch_marker_is_not_resent(tmp_path):
    class WorkerTerminated(BaseException):
        pass

    wrapped, repo, gateway, _ = setup(tmp_path)
    gateway.error = WorkerTerminated()
    with pytest.raises(WorkerTerminated):
        asyncio.run(wrapped.generate(request()))
    gateway.error = None
    with pytest.raises(ProviderCallError, match="AlreadyClaimed"):
        asyncio.run(wrapped.generate(request()))
    assert repo.rows[request().request_id]["state"] == "reserved" and gateway.calls == 1


def test_concurrent_same_request_has_only_one_provider_dispatch(tmp_path):
    wrapped, repo, gateway, _ = setup(tmp_path)

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked(request):
            gateway.calls += 1
            started.set()
            await release.wait()
            return response()

        gateway.generate = blocked
        first = asyncio.create_task(wrapped.generate(request()))
        await started.wait()
        with pytest.raises(ProviderCallError, match="AlreadyClaimed"):
            await wrapped.generate(request())
        release.set()
        await first

    asyncio.run(exercise())
    assert gateway.calls == 1 and repo.rows[request().request_id]["state"] == "settled"


def test_cancellation_retains_reservation_and_is_not_resent(tmp_path):
    wrapped, repo, gateway, _ = setup(tmp_path)
    gateway.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(wrapped.generate(request()))
    with pytest.raises(ProviderCallError, match="ProviderCallCancelled"):
        asyncio.run(wrapped.generate(request()))
    assert repo.rows[request().request_id]["state"] == "unknown" and gateway.calls == 1


def rate_card_file(tmp_path):
    path = tmp_path / "rate-card.json"
    path.write_text(
        json.dumps(
            {
                "estimated": True,
                "rate_limits": {
                    MODEL: {
                        "provider": "openai",
                        "account_scope": "test-account",
                        "model_scope": MODEL,
                        "requests_per_minute": 60,
                        "tokens_per_minute": 100000,
                    }
                },
                "models": {
                    MODEL: {
                        "input_usd_per_million": "2",
                        "output_usd_per_million": "10",
                        "cached_input_usd_per_million": "0.2",
                        "cache_creation_input_usd_per_million": "2.5",
                    }
                },
            }
        )
    )
    return path


def test_native_factory_requires_accounting_even_for_injected_client():
    for alias in ("answer-gpt-6-luna", "answer-claude-haiku-5.5"):
        with pytest.raises(ValueError, match="accounting requires"):
            create_generation_gateway(
                Settings(_env_file=None), registry(), alias, provider_client=SimpleNamespace()
            )


def test_factory_closed_api_reserves_with_user_rates_and_journals_fake_transport(tmp_path):
    path = rate_card_file(tmp_path)
    rates = load_rate_card(path)
    repo = Repository()
    repo.approval.rate_card_hash = rates.sha256
    calls = []

    class Responses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            assert repo.rows[request().request_id]["state"] == "reserved"
            raw = {
                "model": MODEL,
                "id": "response-1",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 30},
                    "output_tokens_details": {"reasoning_tokens": 5},
                },
            }
            return SimpleNamespace(_request_id="provider-1", model_dump=lambda **_: raw)

    settings = Settings(
        _env_file=None,
        step8_budget_approval_id=APPROVAL_ID,
        step8_rate_card_path=path,
        step8_output_root=tmp_path / "campaign-store",
        step8_attempt_id="attempt-1",
    )
    wrapped = create_generation_gateway(
        settings,
        registry(),
        request().model_alias,
        provider_client=SimpleNamespace(responses=Responses()),
        accounting_repository=repo,
    )
    assert isinstance(wrapped, SettingsAccountingGateway)
    wrapped.provider_control = RateControl()
    result = asyncio.run(wrapped.generate(request()))
    assert len(calls) == 1
    row = repo.rows[request().request_id]
    assert row["identity"][-1] == Decimal("5.26536000")
    assert row["actual"] == Decimal("0.00034600")  # output already includes reasoning
    assert result.metadata["step8_accounting"]["price_estimated"] is True
    assert result.metadata["step8_accounting"]["rate_card_hash"] == rates.sha256
    assert asyncio.run(wrapped.generate(request())) == result and len(calls) == 1
    mismatched = request().model_copy(update={"metadata": {"step8_selection_hash": "d" * 64}})
    with pytest.raises(ProviderCallError, match="ApprovalRequired"):
        asyncio.run(wrapped.generate(mismatched))
    repo.approval.rate_card_hash = "d" * 64
    with pytest.raises(ProviderCallError, match="ApprovalRequired"):
        asyncio.run(wrapped.generate(request()))
    assert len(calls) == 1


def test_rate_card_is_user_supplied_explicit_estimate_and_exact_model_only(tmp_path):
    path = rate_card_file(tmp_path)
    rates = load_rate_card(path)
    assert rates.actual(response(), registry().get(request().model_alias)) == Decimal("0.0003585")
    raw = json.loads(path.read_text())
    raw.pop("estimated")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_rate_card(path)


@pytest.mark.parametrize(
    "field",
    [
        "uncached_input_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
    ],
)
@pytest.mark.parametrize("invalid", [None, -1, True, "1", 1.5])
def test_rate_card_unknown_or_invalid_usage_is_not_arithmetic(field, invalid, tmp_path):
    rates = load_rate_card(rate_card_file(tmp_path))
    result = response()
    result.usage.metadata[field] = invalid
    assert rates.actual(result, registry().get(request().model_alias)) is None


def test_rate_card_usage_components_must_still_match_total(tmp_path):
    rates = load_rate_card(rate_card_file(tmp_path))
    result = response()
    result.usage.metadata["uncached_input_tokens"] += 1
    assert rates.actual(result, registry().get(request().model_alias)) is None


def test_new_settings_are_exportable_to_children(monkeypatch, tmp_path):
    monkeypatch.setenv("PMMC_STEP8_BUDGET_APPROVAL_ID", APPROVAL_ID)
    monkeypatch.setenv("PMMC_STEP8_RATE_CARD_PATH", str(tmp_path / "rates.yaml"))
    monkeypatch.setenv("PMMC_STEP8_OUTPUT_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("PMMC_STEP8_ATTEMPT_ID", "attempt-env")
    settings = Settings(_env_file=None)
    assert settings.step8_budget_approval_id == APPROVAL_ID
    assert settings.step8_rate_card_path == tmp_path / "rates.yaml"
    assert settings.step8_output_root == tmp_path / "results"
    assert settings.step8_attempt_id == "attempt-env"
