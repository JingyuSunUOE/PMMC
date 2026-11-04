"""Lightweight shared runtime errors, independent of model and service imports."""


class ManagedRuntimeError(RuntimeError):
    """Managed runtime preflight or lifecycle failure."""
