"""Repo-relative path helpers.

Label files written on the original Windows workstation store absolute paths
like ``F:\\Claude_Projects\\baseball-biomechanics\\data\\...``. These helpers
re-root such paths at the current checkout, and write new ones repo-relative
so label files stay portable between machines.
"""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR_NAME = "baseball-biomechanics"


def resolve_repo_path(stored: Union[str, Path]) -> Path:
    """Map a path stored in a label file to a path in this checkout.

    - Relative paths (either slash style) are taken relative to PROJECT_ROOT.
    - Absolute paths from any machine that contain a ``baseball-biomechanics``
      directory are re-rooted at PROJECT_ROOT from that point.
    - Anything else is returned unchanged.
    """
    s = str(stored)
    is_windows = "\\" in s or s[1:2] == ":"
    pure = PureWindowsPath(s) if is_windows else PurePosixPath(s)
    parts = pure.parts

    if not pure.is_absolute() and not pure.drive:
        return PROJECT_ROOT.joinpath(*parts)
    if REPO_DIR_NAME in parts:
        i = parts.index(REPO_DIR_NAME)
        return PROJECT_ROOT.joinpath(*parts[i + 1:])
    return Path(s)


def to_repo_relative(path: Union[str, Path]) -> str:
    """Return ``path`` as a forward-slash string relative to PROJECT_ROOT.

    Paths outside the repo are returned as absolute strings.
    """
    # abspath, not resolve(): a symlinked data/ dir must still count as in-repo
    p = Path(os.path.abspath(resolve_repo_path(path)))
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)
