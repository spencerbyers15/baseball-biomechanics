"""Per-stadium wall geometry, extracted from MLB's own FieldVision ballpark
models and expressed in the tracking coordinate frame (feet, Y-up, home
plate at origin, +X toward 1B, -Z toward CF... note: -Z is toward CF in the
GLB frame; bearings below use atan2(x, -z) so 0° = straight-away CF).

Source: the Gameday 3D poser bundle pins an asset hash for
fv-assets.mlb.com; ballpark GLBs live at
  https://fv-assets.mlb.com/v/<40-hex>/models/ballparks/{venueId}_{ABBR}.glb
Each GLB carries a `{ABBR}_Field` node (playing surface + outfield/foul
walls + poles) with Draco-compressed meshes. Net vertex transform to the
tracking frame (verified against official park dimensions for CLE + BOS):
  cm -> m (*0.01), then +90deg about X ((x,y,z) -> (x,-z,y)), then
  m -> ft (*3.28084).

Building polylines needs `DracoPy` + `numpy` + network access; CONSUMING a
built cache needs numpy only (see load_polyline / distance_from_wall).

Build: python scripts/build_stadium_walls.py --out <cache_dir>
"""

from __future__ import annotations

import json
import math
import re
import struct
import urllib.request
from pathlib import Path

import numpy as np

POSER_BUNDLE_URL = ("https://prod-gameday.mlbstatic.com/app-mlb/5.50.0-mlb.5/"
                    "gd.@bvg_poser.min.js")
# Known-good fallback if the bundle regex ever fails (pinned 2026-08-09).
FALLBACK_ASSET_HASH = "f5daae477c086bc43c62e80274e3be95a6f6a47c"
ASSET_BASE = "https://fv-assets.mlb.com/v/{h}/models/ballparks/{vid}_{abbr}.glb"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

M_TO_FT = 3.28084


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def resolve_asset_hash() -> str:
    """The /v/<hash>/ prefix is mandatory (403 without it); it's pinned in
    the poser bundle."""
    try:
        js = _get(POSER_BUNDLE_URL).decode("utf-8", "replace")
        m = re.search(r"fv-assets\.mlb\.com/v/([0-9a-f]{40})", js)
        if m:
            return m.group(1)
    except Exception:
        pass
    return FALLBACK_ASSET_HASH


def venue_abbr_map() -> dict[int, str]:
    """venueId -> park abbreviation, parsed from the bundle's team array."""
    js = _get(POSER_BUNDLE_URL).decode("utf-8", "replace")
    pairs = re.findall(
        r'teamId:\d+,abbreviation:"([A-Z]+)",venueId:(\d+)', js)
    out = {int(vid): abbr for abbr, vid in pairs}
    if not out:
        raise RuntimeError("could not parse venueId map from poser bundle")
    return out


def _parse_glb(data: bytes) -> tuple[dict, bytes]:
    magic, _ver, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:  # 'glTF'
        raise ValueError("not a GLB")
    off = 12
    gltf = None
    binchunk = b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:  # JSON
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:  # BIN
            binchunk = chunk
        off += 8 + clen
    if gltf is None:
        raise ValueError("GLB missing JSON chunk")
    return gltf, binchunk


def field_mesh_feet(glb_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """(verts_feet[N,3], faces[M,3]) of the `*_Field` node — playing surface,
    outfield + foul walls, poles. Requires DracoPy (import deferred so cache
    consumers don't need it)."""
    import DracoPy

    gltf, binchunk = _parse_glb(glb_bytes)
    node = next(n for n in gltf["nodes"] if n.get("name", "").endswith("_Field"))
    mesh = gltf["meshes"][node["mesh"]]

    all_v, all_f = [], []
    voff = 0
    for prim in mesh["primitives"]:
        ext = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
        if ext is None:
            continue
        # The ad-plane primitive isn't stadium geometry.
        mat_idx = prim.get("material")
        if mat_idx is not None:
            mname = gltf.get("materials", [{}])[mat_idx].get("name", "")
            if "AdMaster" in mname:
                continue
        bv = gltf["bufferViews"][ext["bufferView"]]
        blob = binchunk[bv.get("byteOffset", 0):
                        bv.get("byteOffset", 0) + bv["byteLength"]]
        dec = DracoPy.decode(blob)
        v = np.asarray(dec.points, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(dec.faces, dtype=np.int64).reshape(-1, 3)
        all_v.append(v)
        all_f.append(f + voff)
        voff += len(v)
    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    # raw cm, Z-up  ->  feet, Y-up (see module docstring; verified vs official
    # park dimensions for CLE + BOS)
    m = verts * 0.01
    feet = np.column_stack([m[:, 0], -m[:, 2], m[:, 1]]) * M_TO_FT
    return feet, faces


def extract_wall_polyline(verts_ft: np.ndarray, faces: np.ndarray,
                          bearing_lo: float = -46.0, bearing_hi: float = 46.0,
                          step_deg: float = 1.0, min_r_ft: float = 200.0,
                          face_tol_ft: float = 4.0) -> list[dict]:
    """Ordered outfield-wall polyline with heights.

    Keeps near-vertical triangles, sweeps bearing bins from the 3B line to
    the 1B line (0 deg = straight-away CF), and per bin takes the nearest
    vertical face beyond min_r_ft as the wall; wall top = max height among
    vertices near that face. Foul poles appear as ~50-60 ft spikes at the
    +/-45 deg edges — they're flagged, not removed.
    """
    tri = verts_ft[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    vertical = np.abs(n[ok, 1] / norm[ok]) < 0.35
    vv = tri[ok][vertical].reshape(-1, 3)  # vertices of vertical tris

    x, y, z = vv[:, 0], vv[:, 1], vv[:, 2]
    r = np.hypot(x, z)
    bearing = np.degrees(np.arctan2(x, -z))

    out = []
    for b0 in np.arange(bearing_lo, bearing_hi, step_deg):
        sel = (bearing >= b0) & (bearing < b0 + step_deg) & (r > min_r_ft)
        if not sel.any():
            continue
        face_r = r[sel].min()
        near = sel & (np.abs(r - face_r) < face_tol_ft)
        top = float(y[near].max())
        mid = math.radians(b0 + step_deg / 2)
        out.append({
            "bearing_deg": round(b0 + step_deg / 2, 1),
            "x": round(face_r * math.sin(mid), 2),
            "z": round(-face_r * math.cos(mid), 2),
            "r": round(float(face_r), 2),
            "wall_top_ft": round(top, 2),
            "pole_spike": bool(top > 45.0 and abs(b0) > 42.0),
        })
    return out


def build_venue(venue_id: int, abbr: str, asset_hash: str) -> dict:
    glb = _get(ASSET_BASE.format(h=asset_hash, vid=venue_id, abbr=abbr))
    verts, faces = field_mesh_feet(glb)
    poly = extract_wall_polyline(verts, faces)
    return {"venue_id": venue_id, "abbr": abbr, "n_points": len(poly),
            "glb_bytes": len(glb), "polyline": poly}


# ── Cache consumers (numpy only) ───────────────────────────────────────────


def load_polyline(cache_dir: Path, venue_id: int) -> np.ndarray:
    """[N,3] array of (x, z, wall_top_ft) for a venue from a built cache."""
    d = json.loads((Path(cache_dir) / f"{venue_id}.json").read_text())
    return np.array([[p["x"], p["z"], p["wall_top_ft"]]
                     for p in d["polyline"]])


def distance_from_wall(x: float, z: float, poly: np.ndarray) -> float:
    """Min XZ-plane distance from a point to the wall polyline segments."""
    a = poly[:-1, :2]
    b = poly[1:, :2]
    p = np.array([x, z])
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab)
                / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))
