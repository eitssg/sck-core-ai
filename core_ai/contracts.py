"""DEPRECATED: Use `core_framework.ai.contracts` instead.

This file remains temporarily to avoid breaking imports during the migration
window. All symbols are re-exported from the shared framework namespace.
It will be removed after downstream services are updated.
"""

from __future__ import annotations

import warnings as _warnings

from core_framework.ai.contracts import *  # noqa: F401,F403

_warnings.warn(
    "core_ai.contracts is deprecated; import from core_framework.ai.contracts instead",
    DeprecationWarning,
    stacklevel=2,
)
