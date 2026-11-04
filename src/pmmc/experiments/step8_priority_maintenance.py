"""Public SOURCE: private campaign exceptions are not distributed."""


def _unavailable(*args, **kwargs):
    raise ValueError("private_campaign_exception_not_available")



approval = _unavailable
execution_maintenance = _unavailable
policy = _unavailable
retained_producer = _unavailable
revision = _unavailable
validate_approval = _unavailable
validate_execution = _unavailable
verify_build = _unavailable
