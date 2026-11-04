"""Host CLI adapter for explicit, scoped release Secret import."""

from .config import check
from .private_secrets import import_values, present_roles, read_dotenv


def import_for_config(cfg, values, roles, *, rotate=False):
    check(cfg.get("profile") == "kubernetes", "Secret import requires the Kubernetes profile")
    check("hub" not in roles or cfg["models"]["use_hf_token"], "Hub credential role is disabled")
    return import_values(
        values,
        roles,
        context=cfg["cluster"]["context"],
        namespace=cfg["cluster"]["namespace"],
        instance=cfg["instance"],
        rotate=rotate,
    )


def import_secrets(cfg, env_file, *, roles=None, rotate=False):
    """Create or compare selected roles; drift fails unless rotation was explicitly requested."""
    values = read_dotenv(env_file, roles)
    return import_for_config(cfg, values, roles or present_roles(values), rotate=rotate)
