"""Stage 1 Domain Macro registry — versioned, deterministic expansion."""

from reckon.macros.expand import apply_domain_macros
from reckon.macros.registry import (
    DOMAIN_MACROS,
    get_macro,
    list_macros,
)

__all__ = [
    "DOMAIN_MACROS",
    "apply_domain_macros",
    "get_macro",
    "list_macros",
]
