"""Command line.

    python -m throwtrack demo [--minutes 6] [--seed 0] [--out runs/demo]
    python -m throwtrack analyze VIDEO --out DIR [--calibration calib.json] [--roster roster.json]
                                     [--palette palette.json] [--velocity-model vm.json] [--overlay]
    python -m throwtrack fit-velocity --features DIR/throw_features.json --radar radar.csv --out vm.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .ball_type import BallPalette
from .cache import load_tracks, save_tracks
from .evaluate import evaluate
from .geometry import load_calibration
from .pipeline import DEFAULT_HEIGHT_M, SessionConfig, analyze_session
from .report import write_all
from .schema import METERS_PER_FOOT, MPS_PER_MPH, ThrowEvent
from .velocity import KinematicVelocityModel, assign_intensity, assign_velocity, kinematic_features, match_radar


def _load_roster(path) -> dict:
    """roster.json: {"<track_id>": {"name": "Smith", "height_in": 74}} (height optional)."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    out = {}
    for k, v in raw.items():
        v = dict(v)
        if "height_in" in v:
            v["height_m"] = float(v["height_in"]) * 0.0254
        out[int(k)] = v
    return out


def _feature_rows(throws: list[ThrowEvent], roster: dict, palette) -> list[dict]:
    masses = palette.masses if palette else {}
    rows = []
    for t in throws:
        f = kinematic_features(t, roster.get(t.track_id, {}).get("height_m", DEFAULT_HEIGHT_M),
                               masses.get(t.ball_type) if t.ball_type else None)
        rows.append({"time_s": t.time_s, "track_id": t.track_id, "ball_type": t.ball_type,
                     **{k: (None if v != v else v) for k, v in f.items()}})
    return rows


def cmd_demo(args) -> None:
    from .simulate import build_session

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Simulating a {args.minutes:g}-minute session (3 throwing pairs + 1 wall-drill player)...")
    sess = build_session(duration_s=args.minutes * 60, seed=args.seed)
    (out / "calibration.json").write_text(json.dumps(sess.calibration, indent=1))
    roster = {pid: {"name": f"Pitcher {pid}", "height_m": p.height_m} for pid, p in sess.players.items()}
    cfg = SessionConfig(fps=sess.fps, ground=load_calibration(sess.calibration), roster=roster)
    throws = analyze_session(sess.tracks, cfg)

    truth = [{"thrower": g["thrower"], "time_s": g["time_s"], "target": g["target"],
              "distance_ft": g["distance_m"] / METERS_PER_FOOT if g["target"] != "wall" else None,
              "mph": g["ball_speed_mps"] / MPS_PER_MPH, "ball_type": None} for g in sess.throws]

    # Pretend a radar gun covered the first half of the session; fit on it, test on the rest.
    half = sess.duration_s / 2
    radar = [{"time_s": g["time_s"], "mph": g["mph"], "track_id": g["thrower"]} for g in truth if g["time_s"] < half]
    pairs = match_radar(throws, radar, tol_s=0.25)
    heights = {pid: r["height_m"] for pid, r in roster.items()}
    feats = [kinematic_features(throws[i], heights[throws[i].track_id]) for i, _ in pairs]
    model = KinematicVelocityModel().fit(feats, [m for _, m in pairs], groups=[throws[i].track_id for i, _ in pairs])
    assign_velocity(throws, model, heights, {})
    assign_intensity(throws)

    write_all(throws, out, roster, title="ThrowTrack demo (simulated session)")
    res_all = evaluate(throws, truth)
    res_test = evaluate([t for t in throws if t.time_s >= half], [g for g in truth if g["time_s"] >= half])
    print(f"\nGround truth throws: {res_all['n_truth']}   detected: {res_all['n_detected']}")
    print(f"Throw detection     precision {res_all['precision']:.3f}   recall {res_all['recall']:.3f}"
          f"   release timing sd {res_all['timing_sd_ms']:.0f} ms")
    print(f"Receiver / wall     accuracy {res_all['target_accuracy']:.3f}")
    print(f"Partner distance    MAE {res_all['distance_mae_ft']:.1f} ft over {res_all['n_distance']} throws")
    print(f"Velocity (held-out) MAE {res_test['velo_mae_mph']:.1f} mph over {res_test['n_velo']} throws,"
          f" 90% interval coverage {res_test.get('velo_interval_coverage', float('nan')):.2f}"
          f"  (fit on {len(pairs)} radar throws)")
    print(f"\nWrote {out}/throws.csv, players.csv, throws.json, report.html")
    (out / "metrics.json").write_text(json.dumps({"all": res_all, "velocity_holdout": res_test}, indent=1))


def cmd_analyze(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.pose_cache) if args.pose_cache else out / "poses.npz"
    if cache.exists():
        print(f"Loading cached poses from {cache}")
        tracks, info = load_tracks(cache)
    else:
        from .video import extract_pose_tracks

        print(f"Pose pass over {args.video} (runs once; cached to {cache})")
        tracks, info = extract_pose_tracks(args.video, det_weights=args.det_weights, pose_onnx=args.pose_onnx,
                                           device=args.device)
        save_tracks(cache, tracks, info)
    roster = _load_roster(args.roster)
    palette = BallPalette.load(args.palette) if args.palette or args.ball_type else None
    model = (KinematicVelocityModel.from_dict(json.loads(Path(args.velocity_model).read_text()))
             if args.velocity_model else None)
    cfg = SessionConfig(fps=info["fps"], roster=roster, palette=palette, velocity_model=model,
                        ground=load_calibration(args.calibration) if args.calibration else None)
    ball_fn = None
    if palette is not None:
        from .video import make_ball_pixels_fn

        ball_fn = make_ball_pixels_fn(args.video, tracks, info["fps"])
    throws = analyze_session(tracks, cfg, ball_fn)
    start = datetime.fromisoformat(args.start_time) if args.start_time else None
    write_all(throws, out, roster, start, title=f"Throwing session: {Path(args.video).name}")
    (out / "throw_features.json").write_text(json.dumps(_feature_rows(throws, roster, palette), indent=1))
    if args.overlay:
        from .video import render_overlay

        render_overlay(args.video, out / "overlay.mp4", tracks, throws, roster)
    print(f"{len(throws)} throws, {sum(t.needs_review for t in throws)} flagged for review -> {out}/report.html")


def cmd_fit_velocity(args) -> None:
    feats = json.loads(Path(args.features).read_text())
    with open(args.radar) as f:
        radar = [{k: (float(v) if k in ("time_s", "mph") else v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in radar:
        r["track_id"] = int(r["track_id"]) if r.get("track_id") not in (None, "") else None
    stub = [ThrowEvent(track_id=r["track_id"], frame=0, time_s=r["time_s"], arm="", confidence=1) for r in feats]
    pairs = match_radar(stub, radar, tol_s=args.tol)
    rows = [{k: (float("nan") if feats[i][k] is None else feats[i][k]) for k in KinematicVelocityModel().feature_names}
            for i, _ in pairs]
    model = KinematicVelocityModel(l2=args.l2).fit(rows, [m for _, m in pairs],
                                                   groups=[feats[i]["track_id"] for i, _ in pairs])
    Path(args.out).write_text(json.dumps(model.to_dict(), indent=1))
    print(f"Fit on {len(pairs)} radar-matched throws; residual sd {model.resid_sd:.1f} mph -> {args.out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="throwtrack", description="Count and characterize non-mound throws from video.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the full pipeline on a simulated session (no GPU, no video)")
    d.add_argument("--minutes", type=float, default=6.0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="runs/demo")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("analyze", help="analyze a session video")
    a.add_argument("video")
    a.add_argument("--out", required=True)
    a.add_argument("--calibration", help="calibration.json (homography cones or person_height)")
    a.add_argument("--roster", help='roster.json: {"<track_id>": {"name": ..., "height_in": ...}}')
    a.add_argument("--palette", help="ball palette JSON (default palette if --ball-type is set)")
    a.add_argument("--ball-type", action="store_true", help="classify ball type with the default palette")
    a.add_argument("--velocity-model", help="model JSON from fit-velocity")
    a.add_argument("--pose-cache", help="where to cache poses (default OUT/poses.npz)")
    a.add_argument("--det-weights", default="yolo11m.pt")
    a.add_argument("--pose-onnx", default=None)
    a.add_argument("--device", default="cuda")
    a.add_argument("--start-time", help="wall-clock time of the first frame, ISO format")
    a.add_argument("--overlay", action="store_true", help="also write an annotated overlay.mp4")
    a.set_defaults(func=cmd_analyze)

    f = sub.add_parser("fit-velocity", help="calibrate the velocity model on radar readings")
    f.add_argument("--features", required=True, help="throw_features.json from analyze")
    f.add_argument("--radar", required=True, help="CSV with time_s, mph[, track_id]")
    f.add_argument("--out", required=True)
    f.add_argument("--tol", type=float, default=1.0, help="max seconds between radar and release")
    f.add_argument("--l2", type=float, default=1.0)
    f.set_defaults(func=cmd_fit_velocity)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
