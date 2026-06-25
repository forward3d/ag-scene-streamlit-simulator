"""
Version compatibility checks for Scene MX.

This repo is not distributed via PyPI, so we do not check for "latest".
Instead we warn if the installed version is below a minimum required version.
"""

from __future__ import annotations

import os

from packaging import version as _version


MIN_REQUIRED_VERSION = "1.0.1"


def warn_if_outdated(
    *,
    current_version: str,
    min_required_version: str = MIN_REQUIRED_VERSION,
) -> None:
    """Print a warning when the installed version is too old.

    Users can silence this with `SCENE_MX_SKIP_VERSION_CHECK=1`.
    """

    if os.getenv("SCENE_MX_SKIP_VERSION_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return

    try:
        if _version.parse(current_version) < _version.parse(min_required_version):
            print(
                f"\033[91mWarning: Your Scene MX version ({current_version}) is outdated. "
                f"Please install Scene MX (>= {min_required_version}).\033[0m",
                flush=True,
            )
    except Exception:
        # If version parsing fails for any reason, do not break execution.
        return

