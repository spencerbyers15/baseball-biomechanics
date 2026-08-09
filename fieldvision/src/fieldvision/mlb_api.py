"""Shared constants + pipeline-version resolution for the mannequin endpoint.

MLB versions the FieldVision data path: /mannequin/{gamePk}/{version}/...
The version bumps occasionally (1.6.2 until ~2026-05-20, then 1.7.0) and
old games are not always reprocessed to the new version, so the working
version can differ per game. Nothing enumerates versions server-side (the
bare /mannequin/{pk}/ path 404s), so we resolve by probing manifest.json
across candidates, newest-known first.
"""

from __future__ import annotations

import os

MANNEQUIN_HOST = "https://fieldvision-hls.mlbinfra.com"

# Newest-known version first, then plausible future bumps, legacy last.
# When MLB bumps the pipeline again the probe walks this list; prepend the
# new version once it's known (or set FV_PIPELINE_VERSION to skip probing).
VERSION_CANDIDATES = ("1.7.0", "1.7.1", "1.8.0", "1.9.0", "2.0.0", "1.6.2")

# Last version that returned 200 for any game this process — tried first so
# steady-state resolution costs zero extra requests.
_version_hint: str | None = None


def game_base(game_pk: int | str, version: str) -> str:
    return f"{MANNEQUIN_HOST}/mannequin/{game_pk}/{version}"


def manifest_url(game_pk: int | str, version: str) -> str:
    return f"{game_base(game_pk, version)}/manifest.json"


def resolve_manifest(game_pk: int | str, http_get) -> tuple[str | None, int, bytes]:
    """Find the pipeline version that serves this game and fetch its manifest.

    http_get: callable (url) -> (status_code, body_bytes) with auth headers
    already bound (each caller has its own retry/backoff flavor).

    Returns (version, status, body):
      - (v, 200, manifest_bytes) on success
      - (None, 401/403, body) the moment auth fails — probing other
        versions can't help a rejected token
      - (None, status, body) when no candidate works; status is the last
        candidate's (404 = game expired from retention, or a version we
        don't know about yet — extend VERSION_CANDIDATES)

    FV_PIPELINE_VERSION overrides probing entirely.
    """
    global _version_hint

    override = os.environ.get("FV_PIPELINE_VERSION")
    if override:
        status, body = http_get(manifest_url(game_pk, override))
        return (override if status == 200 else None), status, body

    candidates = list(VERSION_CANDIDATES)
    if _version_hint in candidates:
        candidates.remove(_version_hint)
        candidates.insert(0, _version_hint)

    status, body = 0, b""
    for v in candidates:
        status, body = http_get(manifest_url(game_pk, v))
        if status == 200:
            _version_hint = v
            return v, 200, body
        if status in (401, 403):
            return None, status, body
    return None, status, body
