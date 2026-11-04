"""Public SOURCE: private campaign exceptions are not distributed."""


def _unavailable(*args, **kwargs):
    raise ValueError("private_campaign_exception_not_available")


APPROVAL = "private/reuse/engineering-admission.json"


def require_engineering_admission(*, directory, **kwargs):
    if (directory / APPROVAL).exists() or (directory / APPROVAL).is_symlink():
        raise ValueError("private_engineering_admission_not_supported")
    return None

bound_identity = _unavailable
make_admission = _unavailable
policy = _unavailable
