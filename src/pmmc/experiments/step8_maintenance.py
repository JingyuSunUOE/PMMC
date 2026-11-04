"""Public SOURCE: private campaign exceptions are not distributed."""


def _unavailable(*args, **kwargs):
    raise ValueError("private_campaign_exception_not_available")


APPROVED_TRANSITION = {"campaign_id": None}


def execution_maintenance(*, job, protocol, actual_source_fingerprint, **kwargs):
    if job.source_fingerprint != protocol.source_fingerprints.get(job.stage):
        raise ValueError("step8_maintenance_job_protocol_mismatch")
    if job.source_fingerprint != actual_source_fingerprint:
        raise ValueError("component_scientific_source_changed")
    return None

_digest = _unavailable
_read_json = _unavailable
_validate_approval = _unavailable
_verify_runtime_build = _unavailable
make_maintenance_approval = _unavailable
validate_maintenance_execution = _unavailable
verify_campaign_maintenance = _unavailable
