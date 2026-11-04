"""Selection-approved, journal-first provider accounting for campaign workers.

The parent supplies frozen rate-card callbacks; this module contains no prices,
approval creation, model retry, or scientific receipt commit. A successful
generate() returns only after journaling and monetary settlement (or an explicit
unknown-usage hold). The caller may then commit its fenced component receipt.
All workers must share the same durable CampaignArtifactStore root.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.infrastructure.database.step8_models import BenchmarkBudgetApprovalModel
from pmmc.infrastructure.database.step8_repositories import (
    PostgresBenchmarkRepository,
    require_budget_binding,
)
from pmmc.infrastructure.settings import Settings
from pmmc.models.campaign_audit import current_campaign_audit
from pmmc.models.protocols import GenerationGateway
from pmmc.models.provider_images import LocalImageDimensions
from pmmc.models.provider_token_budget import SHARED_CONTEXT_TOKENS, ProviderTokenBudgetCounter
from pmmc.models.providers import ProviderCallError, provider_output_budget
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.models.step8_rate_limits import ProviderRateAdmission, ProviderRateLimit
from pmmc.release_plan import plan_from_environment, plan_hash
from pmmc.schemas import GenerationRequest, GenerationResult, ImageRef
from pmmc.schemas.base import BaseSchema

_HASH = re.compile(r"[a-f0-9]{64}\Z")
_ATTEMPT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_PROTOCOL = "step8-provider-accounting-v1"
_RECOVERY_KEY = "step8_accounting_recovery"
_USD_QUANTUM = Decimal("0.00000001")

WorstCaseCost = Callable[[GenerationRequest, ModelRegistryEntry, int], Decimal]
ActualCost = Callable[[GenerationResult, ModelRegistryEntry], Decimal | None]


@dataclass(frozen=True)
class SelectionApproval:
    campaign_id: str
    selection_hash: str
    approval_id: str
    rate_card_hash: str

    def __post_init__(self) -> None:
        if not self.campaign_id.strip() or not self.approval_id.strip():
            raise ValueError("explicit campaign and selection approval identity required")
        if not _HASH.fullmatch(self.selection_hash) or not _HASH.fullmatch(self.rate_card_hash):
            raise ValueError("selection and frozen rate card require SHA256 identities")


_RELEASE_SCOPE: ContextVar[tuple[dict[str, str], frozenset[str]] | None] = ContextVar(
    "release_budget_worker_scope", default=None)


def bind_release_worker(plan: dict, campaign_id: str, selection_hash: str,
                        scope_hash: str, job_ids: Sequence[str]) -> None:
    """Install the worker's verified native scope, not an environment-only capability."""
    _RELEASE_SCOPE.set((dict(plan_hash=plan_hash(plan), campaign_id=campaign_id,
                             selection_hash=selection_hash, scope_hash=scope_hash),
                        frozenset(job_ids)))


def _worker_binding() -> dict[str, str] | None:
    plan = plan_from_environment(os.environ)
    scope = _RELEASE_SCOPE.get()
    if plan is None and scope is None:
        return None
    audit = current_campaign_audit()
    if (plan is None or scope is None or audit is None or scope[0]["plan_hash"] != plan_hash(plan)
            or audit.campaign_id != scope[0]["campaign_id"] or audit.job_id not in scope[1]):
        raise ProviderCallError("ExplicitSelectionApprovalRequired", outcome="failed")
    return scope[0]


async def require_selection_approval(
    repository: PostgresBenchmarkRepository,
    approval: SelectionApproval,
) -> None:
    """Read an existing human approval; never inherit approval from an expansion."""
    async with repository.sessions() as session:
        row = await session.get(BenchmarkBudgetApprovalModel, approval.approval_id)
        if (
            row is None
            or row.rate_card_hash != approval.rate_card_hash
            or not row.approver_id.strip()
            or (row.limit_usd is not None and row.limit_usd <= 0)
        ):
            raise ProviderCallError("ExplicitSelectionApprovalRequired", outcome="failed")
        binding = _worker_binding()
        try:
            if binding and (binding["campaign_id"], binding["selection_hash"]) != (
                approval.campaign_id, approval.selection_hash
            ):
                raise ValueError("release_worker_selection_changed")
            require_budget_binding(row, campaign_id=approval.campaign_id,
                                   selection_hash=approval.selection_hash,
                                   plan_hash=binding["plan_hash"] if binding else None,
                                   scope_hash=binding["scope_hash"] if binding else None)
        except ValueError as exc:
            raise ProviderCallError("ExplicitSelectionApprovalRequired", outcome="failed") from exc


def _money(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or not 0 <= value < 10**12:
        raise ValueError("cost must be a finite nonnegative Decimal fitting NUMERIC(20,8)")
    return value.quantize(_USD_QUANTUM, rounding=ROUND_CEILING)


class VerifiedProviderOutcome(BaseSchema):
    """Output of a trusted proof verifier, not an operator-supplied assertion."""

    request_id: str = Field(min_length=1, max_length=128)
    request_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    approval_id: str = Field(min_length=1, max_length=128)
    disposition: Literal["completed", "not_executed"]
    actual_usd: Decimal = Field(ge=0)
    response_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    provider_request_id: str = Field(min_length=1, max_length=256)


EvidenceVerifier = Callable[[bytes], Awaitable[VerifiedProviderOutcome]]


async def reconcile_unknown_provider_call(
    *,
    provider_control: Any,
    artifact_store: CampaignArtifactStore,
    registry: ModelRegistry,
    evidence_sha256: str,
    reviewer_id: str,
    verifier_id: str,
    verify_evidence: EvidenceVerifier,
) -> dict[str, Any]:
    """Reconcile only verified immutable provider evidence, without dispatch.

    The trusted verifier must authenticate provider-native response/billing or
    nonexecution proof and bind it to the exact request/approval. A user JSON
    assertion, timeout, or absence of a bill is not such proof. The CLI must choose
    the verifier from an allowlist, never import arbitrary operator Python code.
    """
    if not _HASH.fullmatch(evidence_sha256):
        raise ValueError("reconciliation evidence requires a CAS SHA256")
    if not reviewer_id.strip() or not verifier_id.strip():
        raise ValueError("reconciliation requires reviewer and trusted verifier identities")
    evidence = artifact_store.verify(evidence_sha256).read_bytes()
    outcome = await verify_evidence(evidence)
    if not isinstance(outcome, VerifiedProviderOutcome):
        raise ValueError("trusted verifier must return VerifiedProviderOutcome, not decoded JSON")
    # Revalidate even constructed/copied models; their fields may have bypassed validation.
    # The type check is an adapter contract, not proof authentication by itself.
    outcome = VerifiedProviderOutcome.model_validate(outcome.model_dump())
    actual = _money(outcome.actual_usd)
    if outcome.disposition == "not_executed":
        if actual != 0 or outcome.response_sha256 is not None:
            raise ValueError("verified nonexecution must have zero spend and no response")
    else:
        if outcome.response_sha256 is None:
            raise ValueError("completed reconciliation requires immutable response evidence")
        response = artifact_store.load_json(outcome.response_sha256)
        if (
            not isinstance(response, dict)
            or response.get("protocol") != _PROTOCOL
            or response.get("request_id") != outcome.request_id
            or response.get("request_hash") != outcome.request_hash
            or not isinstance(response.get("approval"), dict)
            or response.get("approval", {}).get("approval_id") != outcome.approval_id
            or response.get("error") is not None
            or not isinstance(response.get("result"), dict)
        ):
            raise ValueError("reconciled response identity mismatch")
        result = GenerationResult.model_validate(response["result"])
        if (
            result.request_id != outcome.request_id
            or result.provider_request_id != outcome.provider_request_id
        ):
            raise ValueError("reconciled provider response identity mismatch")
        if result.actual_model_id != registry.get(result.model_alias).expected_actual_model_id:
            raise ValueError("reconciled provider model identity mismatch")
    payload: dict[str, Any] = {
        "protocol": "step8-provider-reconciliation-v1",
        **outcome.model_dump(mode="json"),
        "actual_usd": str(actual),
        "evidence_sha256": evidence_sha256,
        "reviewer_id": reviewer_id,
        "verifier_id": verifier_id,
        "next_action": "replay_verified_response"
        if outcome.disposition == "completed"
        else "explicit_new_request_required",
    }
    digest = canonical_content_hash(payload)
    # Save the audit receipt before the atomic DB settlement. Its presence alone
    # does not permit reuse: only the committed repository decision does.
    receipt_sha256 = artifact_store.put_json(payload)
    payload = {**payload, "reconciliation_hash": digest, "receipt_sha256": receipt_sha256}
    await provider_control.reconcile_verified(
        request_id=outcome.request_id,
        request_hash=outcome.request_hash,
        approval_id=outcome.approval_id,
        disposition=outcome.disposition,
        actual_usd=actual,
        response_hash=outcome.response_sha256,
        reconciliation_hash=digest,
        payload=payload,
    )
    return payload


class Step8AccountingGateway:
    """Wrap a configured generation gateway; expose the same generate(request) API.

    worst_case_cost(request, entry, provider_budget) must bound the entire call,
    including image input, JSON overhead, and reasoning. actual_cost(result,
    entry) uses the approved frozen rate card, returning None if not knowable.
    """

    def __init__(
        self,
        gateway: GenerationGateway,
        *,
        repository: PostgresBenchmarkRepository,
        artifact_store: CampaignArtifactStore,
        registry: ModelRegistry,
        approval: SelectionApproval,
        attempt_id: str,
        worst_case_cost: WorstCaseCost,
        actual_cost: ActualCost,
        rate_admission: ProviderRateAdmission | None = None,
        provider_control: Any = None,
        max_rate_limit_retries: int = 2,
        image_dimensions: Callable[[ImageRef], tuple[int, int]] | None = None,
    ) -> None:
        if not _ATTEMPT.fullmatch(attempt_id):
            raise ValueError("invalid accounting attempt ID")
        self.gateway = gateway
        self.repository = repository
        self.artifacts = artifact_store
        self.registry = registry
        self.approval = approval
        self.attempt_id = attempt_id
        self.worst_case_cost = worst_case_cost
        self.actual_cost = actual_cost
        if not 0 <= max_rate_limit_retries <= 8:
            raise ValueError("bounded provider 429 retries must be between zero and eight")
        self.rate_admission = rate_admission
        self.provider_control = provider_control
        self.max_rate_limit_retries = max_rate_limit_retries
        self.image_dimensions = image_dimensions

    def _claim_dispatch(self, request_hash: str, intent: dict[str, Any]) -> bool:
        """A durable exclusive marker closes reserve()'s new/existing ambiguity.

        Never remove this marker, including on timeout or worker termination.
        A marker without a response requires reconciliation, not another call.
        """
        parent = self.artifacts.root / "_store" / "provider-dispatch"
        for path in (parent.parent, parent):
            if path.is_symlink():
                raise ValueError("provider dispatch directory cannot be a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        claim = parent / request_hash
        if claim.is_symlink():
            raise ValueError("provider dispatch marker cannot be a symlink")
        try:
            claim.mkdir()
        except FileExistsError:
            return False
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        write_json(claim / "intent.json", {**intent, "attempt_id": self.attempt_id}, immutable=True)
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await self._generate(request)

    async def recover_result(
        self,
        request: GenerationRequest,
        result: GenerationResult,
    ) -> GenerationResult:
        """Reconcile a journaled completed response without a transport fallback."""
        envelope = result.metadata.get(_RECOVERY_KEY)
        if not isinstance(envelope, dict):
            raise ProviderCallError("ResponseRecoveryEnvelopeMissing", outcome="unknown")
        return await self._generate(request, recovery_envelope=envelope)

    async def _generate(
        self,
        request: GenerationRequest,
        *,
        recovery_envelope: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if len(request.request_id) > 128:
            raise ValueError("accounted provider request IDs must fit the reservation ledger")
        for message in request.messages:
            for part in message.parts:
                if part.image is not None and not _HASH.fullmatch(part.image.sha256 or ""):
                    raise ValueError("accounted image requests require immutable image SHA256")
        entry = self.registry.get(request.model_alias)
        plan = plan_from_environment(os.environ)
        if plan is not None and request.model_alias not in plan["models"]:
            raise ProviderCallError("ExplicitSelectionApprovalRequired", outcome="failed")
        if entry.provider_config is None:
            raise ValueError("accounting requires a configured Step 8 provider")
        budget = provider_output_budget(
            request,
            entry.provider_config.max_output_tokens,
            config=entry.provider_config,
        )
        if entry.provider_config.kind in {"openai", "anthropic"}:
            request_budget = ProviderTokenBudgetCounter(
                entry, self.image_dimensions).validate_request(request)
            if self.rate_admission is not None:
                limit = self.rate_admission.limit
                if (
                    limit.provider != entry.provider_config.kind
                    or limit.model_scope not in {"*", entry.provider_model}
                    or request_budget.total_tokens > limit.tokens_per_minute
                ):
                    raise ValueError("provider rate policy does not admit this request/model")
        worst_case = _money(self.worst_case_cost(request, entry, budget))
        if entry.provider_config.kind in {"openai", "anthropic"} and worst_case == 0:
            raise ValueError("paid provider calls require a positive worst-case reservation")
        await require_selection_approval(self.repository, self.approval)
        identity = {
            "protocol": _PROTOCOL,
            "approval": asdict(self.approval),
            "request": request.model_dump(mode="json"),
            "registry_entry_hash": entry.resolved_hash,
            "provider_max_output_tokens": budget,
            "worst_case_usd": str(worst_case),
        }
        request_hash = canonical_content_hash(identity)
        if recovery_envelope is not None:
            self._check_response_identity(request, request_hash, recovery_envelope)
        binding = _worker_binding()
        state = await self.repository.reserve(
            self.approval.approval_id,
            request.request_id,
            request_hash,
            entry.provider_config.kind,
            worst_case,
            **({"binding": binding} if binding else {}),
        )
        if self.provider_control is not None:
            decision = await self.provider_control.reconciliation(request.request_id, request_hash)
            if decision is not None:
                if decision["disposition"] == "not_executed":
                    raise ProviderCallError(
                        "ReconciledNonexecutionRequiresExplicitNewRequest", outcome="failed"
                    )
                envelope = self.artifacts.load_json(decision["response_sha256"])
                self._check_response_identity(request, request_hash, envelope)
                reconciled_result = GenerationResult.model_validate(envelope["result"])
                if (
                    reconciled_result.request_id != request.request_id
                    or reconciled_result.model_alias != request.model_alias
                ):
                    raise ProviderCallError("ResponseJournalIdentityMismatch", outcome="unknown")
                return reconciled_result.model_copy(
                    update={
                        "metadata": {
                            **reconciled_result.metadata,
                            "step8_accounting": {
                                "request_hash": request_hash,
                                "response_sha256": decision["response_sha256"],
                                "approval_id": self.approval.approval_id,
                                "selection_hash": self.approval.selection_hash,
                                "reservation_state": "settled",
                                "reconciliation_hash": decision["reconciliation_hash"],
                            },
                        }
                    }
                )
        if recovery_envelope is not None:
            return await self._finish(request, entry, request_hash, recovery_envelope)
        restored = self.artifacts.restored_response(
            request_id=request.request_id,
            request_hash=request_hash,
        )
        if restored is not None:
            return await self._finish(request, entry, request_hash, restored)
        if state != "reserved":
            raise ProviderCallError("ProviderReservationNeedsReconciliation", outcome="unknown")
        if not self._claim_dispatch(request_hash, identity):
            restored = self.artifacts.restored_response(
                request_id=request.request_id,
                request_hash=request_hash,
            )
            if restored is not None:
                return await self._finish(request, entry, request_hash, restored)
            raise ProviderCallError("ProviderDispatchAlreadyClaimed", outcome="unknown")

        result = None
        error = None
        cancelled = None
        rate_diagnostics: dict[str, Any] = {}
        try:
            result = await self._dispatch(request, request_hash, entry, rate_diagnostics)
        except asyncio.CancelledError as exc:
            cancelled = exc
            error = ProviderCallError("ProviderCallCancelled", outcome="unknown")
        except ProviderCallError as exc:
            error, result = exc, exc.result
        except Exception as exc:
            error = ProviderCallError.from_exception(exc)
        envelope = {
            "protocol": _PROTOCOL,
            "request_id": request.request_id,
            "request_hash": request_hash,
            "approval": asdict(self.approval),
            "result": result.model_dump(mode="json") if result is not None else None,
            "error": {
                "error_code": error.error_code,
                "outcome": error.outcome,
                "provider_request_id": error.provider_request_id,
                "provider_status_code": error.provider_status_code,
            }
            if error is not None
            else None,
            "rate_admission": rate_diagnostics,
        }
        if cancelled is not None:
            try:
                await self._finish(request, entry, request_hash, envelope)
            except ProviderCallError:
                pass
            raise cancelled
        return await self._finish(request, entry, request_hash, envelope)

    async def _dispatch(
        self,
        request: GenerationRequest,
        request_hash: str,
        entry: ModelRegistryEntry,
        diagnostics: dict[str, Any],
    ) -> GenerationResult:
        for ordinal in range(self.max_rate_limit_retries + 1):
            if self.rate_admission is not None:
                token_units = (
                    ProviderTokenBudgetCounter(
                        entry, self.image_dimensions).validate_request(request).total_tokens
                )
                waited = await self.rate_admission.acquire(
                    dispatch_id=canonical_content_hash([request_hash, "transport", ordinal]),
                    token_units=token_units,
                    request_hash=request_hash,
                )
                diagnostics.update(
                    scope_hash=self.rate_admission.limit.scope_hash,
                    reserved_token_units=token_units,
                    transport_count=ordinal + 1,
                    queue_wait_seconds=diagnostics.get("queue_wait_seconds", 0) + waited,
                )
            try:
                result = await self.gateway.generate(request)
                if self.rate_admission is not None:
                    result = result.model_copy(
                        update={
                            "metadata": {
                                **result.metadata,
                                "step8_rate_admission": dict(diagnostics),
                            }
                        }
                    )
                return result
            except ProviderCallError as error:
                if (
                    self.rate_admission is None
                    or error.provider_status_code != 429
                    or error.outcome != "failed"
                    or error.result is not None
                ):
                    raise
                retry_after = await self.rate_admission.retry_after(error)
                # Retain the rejection before permitting another transport attempt.
                rejection_hash = self.artifacts.journal_response(
                    request_id=f"{request.request_id}:429:{ordinal}",
                    request_hash=canonical_content_hash([request_hash, "429", ordinal]),
                    response={
                        "protocol": "step8-explicit-rate-rejection-v1",
                        "request_hash": request_hash,
                        "transport_ordinal": ordinal,
                        "provider_status_code": 429,
                        "outcome": "not_executed",
                        "retry_after_seconds": retry_after,
                    },
                    attempt_id=self.attempt_id,
                )
                diagnostics.setdefault("rate_rejection_sha256", []).append(rejection_hash)
                if retry_after is None or ordinal == self.max_rate_limit_retries:
                    raise
        raise AssertionError("bounded transport loop must return or raise")

    def _check_response_identity(
        self,
        request: GenerationRequest,
        request_hash: str,
        envelope: dict[str, Any],
    ) -> None:
        if any(
            envelope.get(key) != value
            for key, value in {
                "protocol": _PROTOCOL,
                "request_id": request.request_id,
                "request_hash": request_hash,
                "approval": asdict(self.approval),
            }.items()
        ):
            raise ProviderCallError("ResponseJournalIdentityMismatch", outcome="unknown")

    async def _finish(
        self,
        request: GenerationRequest,
        entry: ModelRegistryEntry,
        request_hash: str,
        envelope: dict[str, Any],
    ) -> GenerationResult:
        self._check_response_identity(request, request_hash, envelope)
        result = (
            GenerationResult.model_validate(envelope["result"])
            if envelope.get("result") is not None
            else None
        )
        error = envelope.get("error")
        if result is not None and (
            result.request_id != request.request_id or result.model_alias != request.model_alias
        ):
            raise ProviderCallError("ResponseJournalIdentityMismatch", outcome="unknown")
        if error is not None and error.get("outcome") not in {"failed", "unknown", "completed"}:
            raise ProviderCallError("ResponseJournalInvalidOutcome", outcome="unknown")
        pending_result = (
            result.model_copy(update={"metadata": {**result.metadata, _RECOVERY_KEY: envelope}})
            if result is not None
            else None
        )
        try:
            response_hash = self.artifacts.journal_response(
                request_id=request.request_id,
                request_hash=request_hash,
                response=envelope,
                attempt_id=self.attempt_id,
            )
            # Read-back checks both the request index and CAS content before any DB settlement.
            if (
                self.artifacts.restored_response(
                    request_id=request.request_id,
                    request_hash=request_hash,
                )
                != envelope
            ):
                raise ValueError("provider response journal read-back mismatch")
        except Exception as exc:
            try:
                await self.repository.settle(request.request_id, None, None)
            except Exception:
                pass  # A DB outage must not permit a second provider dispatch either.
            raise ProviderCallError(
                "ResponseJournalUnavailable", outcome="unknown", result=pending_result
            ) from exc
        try:
            actual = None
            if result is not None and result.usage.metadata.get("usage_reported") is True:
                quoted = self.actual_cost(result, entry)
                actual = _money(quoted) if quoted is not None else None
            elif result is None and error is not None and error["outcome"] == "failed":
                actual = Decimal(0)
            await self.repository.settle(request.request_id, actual, response_hash)
        except Exception as exc:
            raise ProviderCallError(
                "ProviderSettlementNeedsReconciliation", outcome="unknown", result=pending_result
            ) from exc
        if error is not None:
            raise ProviderCallError(
                error["error_code"],
                outcome=error["outcome"],
                provider_request_id=error.get("provider_request_id"),
                provider_status_code=error.get("provider_status_code"),
                result=result,
            )
        if result is None:
            raise ProviderCallError("ResponseJournalMissingResult", outcome="unknown")
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "step8_accounting": {
                        "request_hash": request_hash,
                        "response_sha256": response_hash,
                        "approval_id": self.approval.approval_id,
                        "selection_hash": self.approval.selection_hash,
                        "reservation_state": "settled" if actual is not None else "unknown",
                    },
                }
            }
        )


class ModelTokenRates(BaseSchema):
    input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(ge=0)
    cached_input_usd_per_million: Decimal = Field(ge=0)
    cache_creation_input_usd_per_million: Decimal = Field(ge=0)


class ProviderRateCard(BaseSchema):
    """User-supplied rates only. No discovery, defaults, or automatic price freezing."""

    estimated: bool
    models: dict[str, ModelTokenRates]
    rate_limits: dict[str, ProviderRateLimit] = Field(default_factory=dict)


@dataclass(frozen=True)
class LoadedRateCard:
    card: ProviderRateCard
    sha256: str

    def worst_case(
        self, request: GenerationRequest, entry: ModelRegistryEntry, provider_budget: int
    ) -> Decimal:
        del request
        rates = self.card.models[entry.provider_model]
        if entry.context_window is None:
            raise ValueError("a provider context bound is required for monetary reservation")
        # Reserve the full declared context, including images, at the highest
        # approved input rate. Never pass an estimated token count off as exact.
        input_rate = max(
            rates.input_usd_per_million,
            rates.cached_input_usd_per_million,
            rates.cache_creation_input_usd_per_million,
        )
        output_rate = rates.output_usd_per_million
        if entry.provider_model == "gpt-6-luna":
            # The context estimator is not provider-exact. Cover the published
            # >272K price tier even though assembly uses a 32K experiment budget.
            input_rate *= 2
            output_rate *= Decimal("1.5")
        elif entry.provider_model == "claude-haiku-5-5":
            # Short-context prices are 5x higher above 100K input. A local
            # conservative estimate is not provider-exact, so reserve that tier.
            input_rate *= 5
            output_rate *= 5
        return (
            entry.context_window * input_rate + provider_budget * output_rate
        ) / Decimal(1_000_000)

    def actual(self, result: GenerationResult, entry: ModelRegistryEntry) -> Decimal | None:
        if result.actual_model_id != entry.expected_actual_model_id:
            return None
        if entry.provider_model == "gpt-6-luna" and (
            result.metadata.get("provider_service_tier") not in {None, "default"}
            or result.usage.input_tokens > 272_000
        ):
            return None
        if entry.provider_model == "claude-haiku-5-5" and (
            result.metadata.get("provider_service_tier") not in {None, "standard"}
            or result.usage.input_tokens > 100_000
        ):
            # The approved card only covers synchronous standard short contexts.
            # Keep an unknown-cost hold instead of undercharging another tier.
            return None
        rates = self.card.models[entry.provider_model]
        usage = result.usage
        fields: list[int] = []
        for key in (
            "uncached_input_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = usage.metadata.get(key)
            if type(value) is not int or value < 0:
                return None
            fields.append(value)
        uncached, cached, creation = fields
        if uncached + cached + creation != usage.input_tokens:
            return None
        return (
            uncached * rates.input_usd_per_million
            + cached * rates.cached_input_usd_per_million
            + creation * rates.cache_creation_input_usd_per_million
            + usage.output_tokens * rates.output_usd_per_million
        ) / Decimal(1_000_000)


def load_rate_card(path: Path) -> LoadedRateCard:
    """Hash the exact supplied bytes; approve_budget must use this SHA256."""
    data = path.read_bytes()
    card = ProviderRateCard.model_validate(yaml.safe_load(data))
    if not card.models:
        raise ValueError("rate card requires concrete model identities")
    for rates in card.models.values():
        if any(not value.is_finite() for value in rates.model_dump().values()):
            raise ValueError("rate card prices must be finite")
    return LoadedRateCard(card, hashlib.sha256(data).hexdigest())


def preflight_provider_rate_card(
    path: Path, registry: ModelRegistry, model_aliases: Sequence[str]
) -> dict[str, Any]:
    """Pure preflight for CLI approval/planning, before sidecars or GPU startup.

    Null template values, omitted models/limits, and conflicting policies for one
    shared account scope fail here. No credentials, DB, provider or GPU are used.
    """
    loaded = load_rate_card(path)
    scopes: dict[str, ProviderRateLimit] = {}
    models = []
    for alias in sorted(set(model_aliases)):
        entry = registry.get(alias)
        if entry.provider_config is None or entry.provider_config.kind not in {
            "openai",
            "anthropic",
        }:
            continue
        limit = _model_rate_policy(loaded, entry)
        previous = scopes.get(limit.scope_hash)
        if previous is not None and previous != limit:
            raise ValueError("one provider account/model scope has conflicting RPM/TPM policies")
        scopes[limit.scope_hash] = limit
        models.append(
            {
                "model_alias": alias,
                "provider_model": entry.provider_model,
                "scope_hash": limit.scope_hash,
                "requests_per_minute": limit.requests_per_minute,
                "tokens_per_minute": limit.tokens_per_minute,
                "request_budget_ceiling": SHARED_CONTEXT_TOKENS,
            }
        )
    return {"state": "ready", "rate_card_hash": loaded.sha256, "models": models}


def _model_rate_policy(loaded: LoadedRateCard, entry: ModelRegistryEntry) -> ProviderRateLimit:
    if entry.provider_model not in loaded.card.models:
        raise ValueError("rate card does not include the exact configured provider model")
    limit = loaded.card.rate_limits.get(entry.provider_model)
    if (
        limit is None
        or entry.provider_config is None
        or limit.provider != entry.provider_config.kind
    ):
        raise ValueError("paid provider requires explicit shared RPM/TPM and account/model scope")
    if limit.model_scope not in {"*", entry.provider_model}:
        raise ValueError(
            "provider rate model scope must be exact model identity or account wildcard"
        )
    ProviderTokenBudgetCounter(entry)
    if limit.tokens_per_minute < SHARED_CONTEXT_TOKENS:
        raise ValueError("provider TPM must admit the frozen 32768-unit maximum request budget")
    return limit


def load_settings_rate_card(settings: Settings, entry: ModelRegistryEntry) -> LoadedRateCard:
    required = {
        "PMMC_STEP8_BUDGET_APPROVAL_ID": settings.step8_budget_approval_id,
        "PMMC_STEP8_RATE_CARD_PATH": settings.step8_rate_card_path,
        "PMMC_STEP8_OUTPUT_ROOT": settings.step8_output_root,
        "PMMC_STEP8_ATTEMPT_ID": settings.step8_attempt_id,
    }
    if not all(required.values()):
        raise ValueError(
            "Step 8 native API accounting requires: "
            + ", ".join(key for key, value in required.items() if not value)
        )
    if not _HASH.fullmatch(str(settings.step8_budget_approval_id)):
        raise ValueError("Step 8 budget approval ID must be the persisted approval SHA256")
    if not _ATTEMPT.fullmatch(str(settings.step8_attempt_id)):
        raise ValueError("invalid Step 8 attempt ID")
    assert settings.step8_rate_card_path is not None
    loaded = load_rate_card(settings.resolve_path(settings.step8_rate_card_path))
    _model_rate_policy(loaded, entry)
    return loaded


class SettingsAccountingGateway:
    """Mandatory factory boundary for native Step 8 calls, including injected clients."""

    def __init__(
        self,
        gateway: GenerationGateway,
        *,
        settings: Settings,
        registry: ModelRegistry,
        rate_card: LoadedRateCard,
        repository: PostgresBenchmarkRepository | None = None,
        provider_control: Any = None,
    ) -> None:
        self.gateway = gateway
        self.settings = settings
        self.registry = registry
        self.rate_card = rate_card
        self.repository = repository
        self.provider_control = provider_control
        self.image_dimensions = LocalImageDimensions(settings.benchmark_asset_path)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await self._generate(request)

    async def recover_result(
        self,
        request: GenerationRequest,
        result: GenerationResult,
    ) -> GenerationResult:
        return await self._generate(request, recovery_result=result)

    async def _generate(
        self,
        request: GenerationRequest,
        *,
        recovery_result: GenerationResult | None = None,
    ) -> GenerationResult:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.step8_provider_control import (
            PostgresProviderControlRepository,
        )

        engine = None
        repository = self.repository
        if repository is None:
            engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
            repository = PostgresBenchmarkRepository(sessions)
        try:
            async with repository.sessions() as session:
                row = await session.get(
                    BenchmarkBudgetApprovalModel, self.settings.step8_budget_approval_id
                )
                if row is None:
                    raise ProviderCallError("ExplicitSelectionApprovalRequired", outcome="failed")
                binding = _worker_binding()
                approval = SelectionApproval(
                    binding["campaign_id"] if binding else row.campaign_id,
                    binding["selection_hash"] if binding else row.selection_hash,
                    row.approval_id, self.rate_card.sha256,
                )
            # The approval ID is a selection-scoped capability exported by the
            # parent CLI. Explicit request scope, when supplied, must also match.
            for key, expected in (
                ("step8_campaign_id", approval.campaign_id),
                ("step8_selection_hash", approval.selection_hash),
            ):
                if key in request.metadata and request.metadata[key] != expected:
                    raise ProviderCallError("ExplicitSelectionApprovalRequired", outcome="failed")
            assert self.settings.step8_output_root is not None
            control = self.provider_control or PostgresProviderControlRepository(
                repository.sessions
            )
            entry = self.registry.get(request.model_alias)
            limit = self.rate_card.card.rate_limits.get(entry.provider_model)
            if limit is None:
                raise ValueError("paid provider requires explicit shared RPM/TPM admission")
            wrapped = Step8AccountingGateway(
                self.gateway,
                repository=repository,
                artifact_store=CampaignArtifactStore(
                    self.settings.resolve_path(self.settings.step8_output_root)
                ),
                registry=self.registry,
                approval=approval,
                attempt_id=str(self.settings.step8_attempt_id),
                worst_case_cost=self.rate_card.worst_case,
                actual_cost=self.rate_card.actual,
                rate_admission=ProviderRateAdmission(control, limit),
                provider_control=control,
                image_dimensions=self.image_dimensions,
            )
            result = (
                await wrapped.generate(request)
                if recovery_result is None
                else await wrapped.recover_result(request, recovery_result)
            )
            result.metadata["step8_accounting"]["price_estimated"] = self.rate_card.card.estimated
            result.metadata["step8_accounting"]["rate_card_hash"] = self.rate_card.sha256
            return result
        finally:
            if engine is not None:
                await engine.dispose()
