"""Public SOURCE: private campaign exceptions are not distributed."""


def _unavailable(*args, **kwargs):
    raise ValueError("private_campaign_exception_not_available")


CAMPAIGN_ID = None


def retained_producer(*args, **kwargs):
    return False

approval = _unavailable
digest = _unavailable
execution_maintenance = _unavailable
policy = _unavailable
validate_approval = _unavailable
validate_execution = _unavailable
verify_build = _unavailable
