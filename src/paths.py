"""Repo-relative path helpers.

Label files written on the original Windows workstation store absolute paths
like ``F:\\Claude_Projects\\baseball-biomechanics\\data\\...``. These helpers
re-root such *stored* paths at the current project directory, and write *live*
paths relative to it so label files stay portable between machines.

Every helper takes a ``root`` (default: this checkout) so tools that accept
``--project-dir`` resolve against that directory instead.
"""

import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Iterable, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR_NAME = "baseball-biomechanics"

# Path-valued keys in label JSON entries
FRAME_PATH_KEYS = ("path", "source_video")

PathLike = Union[str, Path]


def _is_windows_path(s: str) -> bool:
    return "\\" in s or s[1:2] == ":"


def _is_foreign(s: str) -> bool:
    """A Windows-style path seen on a non-Windows machine (i.e. a stored one)."""
    return os.name != "nt" and _is_windows_path(s)


def _relative_to(path: PurePath, root: PurePath) -> Optional[PurePath]:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def resolve_repo_path(stored: PathLike, root: PathLike = PROJECT_ROOT) -> Path:
    """Map a path stored in a label file to a path under ``root``.

    - Relative paths (either slash style) are taken relative to ``root``.
    - Absolute paths already under ``root`` are returned unchanged.
    - Absolute paths from another machine that contain a
      ``baseball-biomechanics`` directory (any case) are re-rooted at ``root``
      from its last occurrence, skipping a ``.claude/worktrees/<name>`` segment.
    - Anything else is returned unchanged.
    """
    root = Path(root)
    s = str(stored)
    flavor = PureWindowsPath if _is_windows_path(s) else PurePosixPath
    pure = flavor(s)
    parts = pure.parts

    if not pure.is_absolute() and not pure.drive:
        return root.joinpath(*parts)
    if _relative_to(pure, flavor(str(root))) is not None:
        return Path(s)

    lowered = [p.lower() for p in parts]
    if REPO_DIR_NAME in lowered:
        i = len(lowered) - 1 - lowered[::-1].index(REPO_DIR_NAME)
        rest = parts[i + 1:]
        if [p.lower() for p in rest[:2]] == [".claude", "worktrees"]:
            rest = rest[3:]
        return root.joinpath(*rest)
    return Path(s)


def to_repo_relative(path: PathLike, root: PathLike = PROJECT_ROOT) -> str:
    """Return a live path as a forward-slash string relative to ``root``, for storing.

    Relative paths are taken relative to the current directory. A live path
    outside ``root`` (another checkout, a backup copy) stays absolute. Stale
    Windows paths that were never resolved are re-rooted first.
    """
    s = str(path)
    if _is_foreign(s):
        if PureWindowsPath(s).is_absolute():
            s = str(resolve_repo_path(s, root))
        else:
            s = s.replace("\\", "/")
    p = os.path.abspath(s)
    # Symlinks: abspath keeps a symlinked data/ dir in-repo; realpath catches a
    # symlinked ancestor of the checkout.
    for path_form, root_form in ((p, os.path.abspath(root)),
                                 (os.path.realpath(p), os.path.realpath(root))):
        rel = _relative_to(Path(path_form), Path(root_form))
        if rel is not None:
            return rel.as_posix()
    return p


def resolve_entries(entries: Iterable[dict], keys: Iterable[str] = FRAME_PATH_KEYS,
                    root: PathLike = PROJECT_ROOT) -> None:
    """In place: resolve each present ``key`` of each loaded label entry."""
    keys = tuple(keys)
    for entry in entries:
        for key in keys:
            if entry.get(key):
                entry[key] = str(resolve_repo_path(entry[key], root))


def portable_entries(entries: Iterable[dict], keys: Iterable[str] = FRAME_PATH_KEYS,
                     root: PathLike = PROJECT_ROOT) -> list:
    """Copies of ``entries`` with each present ``key`` made root-relative, for saving."""
    keys = tuple(keys)
    return [{**e, **{k: to_repo_relative(e[k], root) for k in keys if e.get(k)}}
            for e in entries]
