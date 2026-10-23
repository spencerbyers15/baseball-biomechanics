"""Repo-relative path helpers.

Label files written on the original Windows workstation store absolute paths
like ``F:\\Claude_Projects\\baseball-biomechanics\\data\\...``. These helpers
re-root such paths at the current project directory, and write new ones
relative to it so label files stay portable between machines.

Every helper takes a ``root`` (default: this checkout) so tools that accept
``--project-dir`` resolve against that directory instead.
"""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR_NAME = "baseball-biomechanics"

PathLike = Union[str, Path]


def _is_windows_path(s: str) -> bool:
    return "\\" in s or s[1:2] == ":"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_repo_path(stored: PathLike, root: PathLike = PROJECT_ROOT) -> Path:
    """Map a path stored in a label file to a path under ``root``.

    - Relative paths (either slash style) are taken relative to ``root``.
    - Absolute paths already under ``root`` are returned unchanged.
    - Absolute paths from another machine that contain a
      ``baseball-biomechanics`` directory are re-rooted at ``root`` from its
      last occurrence.
    - Anything else is returned unchanged.
    """
    root = Path(root)
    s = str(stored)
    is_windows = _is_windows_path(s)
    pure = PureWindowsPath(s) if is_windows else PurePosixPath(s)
    parts = pure.parts

    if not pure.is_absolute() and not pure.drive:
        return root.joinpath(*parts)
    if not is_windows and _is_within(Path(s), root):
        return Path(s)
    if REPO_DIR_NAME in parts:
        i = len(parts) - 1 - parts[::-1].index(REPO_DIR_NAME)
        return root.joinpath(*parts[i + 1:])
    return Path(s)


def to_repo_relative(path: PathLike, root: PathLike = PROJECT_ROOT) -> str:
    """Return a path as a forward-slash string relative to ``root``, for storing.

    ``path`` is a live path on this machine: a relative one is taken relative
    to the current directory. Paths outside ``root`` are returned absolute.
    """
    root = Path(os.path.abspath(root))
    s = str(path)
    if os.path.isabs(s) or _is_windows_path(s):
        s = str(resolve_repo_path(s, root))
    # abspath, not resolve(): a symlinked data/ dir must still count as in-repo
    p = Path(os.path.abspath(s))
    return p.relative_to(root).as_posix() if _is_within(p, root) else str(p)


def resolve_entries(entries: Iterable[dict], keys: Iterable[str],
                    root: PathLike = PROJECT_ROOT) -> None:
    """In place: resolve each present ``key`` of each loaded label entry."""
    keys = tuple(keys)
    for entry in entries:
        for key in keys:
            if entry.get(key):
                entry[key] = str(resolve_repo_path(entry[key], root))


def portable_entries(entries: Iterable[dict], keys: Iterable[str],
                     root: PathLike = PROJECT_ROOT) -> list:
    """Copies of ``entries`` with each present ``key`` made root-relative, for saving."""
    keys = tuple(keys)
    return [{**e, **{k: to_repo_relative(e[k], root) for k in keys if e.get(k)}}
            for e in entries]
