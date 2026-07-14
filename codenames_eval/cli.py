"""Single CLI entry point for the eval harness.

Examples:
    python -m codenames_eval.cli run --mode live --seed 0 --out-dir runs/v1
    python -m codenames_eval.cli run --mode mock --n-dyads 5   # offline smoke
    python -m codenames_eval.cli pilot --mode live --out-dir runs/pilot
    python -m codenames_eval.cli report --run-dir runs/v1
    python -m codenames_eval.cli validate-histories

Runs are resumable: re-invoking `run` with the same config and out-dir skips
already-completed games (results.jsonl is the checkpoint).
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import EvalConfig


def _add_run_args(p: argparse.ArgumentParser, defaults: dict) -> None:
    p.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    p.add_argument("--n-dyads", type=int, default=defaults.get("n_dyads", 50))
    p.add_argument("--boards-per-dyad", type=int,
                   default=defaults.get("boards_per_dyad", 5))
    p.add_argument("--conditions", nargs="+",
                   default=["raw", "summary", "graph"])
    p.add_argument("--n-sessions", type=int, default=30)
    p.add_argument("--n-plants", type=int, default=6)
    p.add_argument("--history-mode", choices=["template", "llm"],
                   default="template")
    p.add_argument("--mode", choices=["mock", "live"], default="live")
    p.add_argument("--giver-model", default=None)
    p.add_argument("--partner-model", default=None)
    p.add_argument("--eaves-model", default=None)
    p.add_argument("--raw-token-budget", type=int, default=100_000)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--out-dir", default=defaults.get("out_dir", "runs/default"))


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
    kw = dict(
        seed=args.seed, n_dyads=args.n_dyads,
        boards_per_dyad=args.boards_per_dyad, conditions=args.conditions,
        n_sessions=args.n_sessions, n_plants=args.n_plants,
        history_mode=args.history_mode, mode=args.mode,
        raw_token_budget=args.raw_token_budget, max_turns=args.max_turns,
        out_dir=args.out_dir)
    for role in ("giver", "partner", "eaves"):
        model = getattr(args, f"{role}_model")
        if model:
            kw[f"{role}_model"] = model
    return EvalConfig(**kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="codenames_eval",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="full eval run (resumable) + report")
    _add_run_args(run_p, {"n_dyads": 50, "boards_per_dyad": 5,
                          "out_dir": "runs/full"})

    pilot_p = sub.add_parser("pilot", help="pilot run (10 dyads x 3 boards)")
    _add_run_args(pilot_p, {"n_dyads": 10, "boards_per_dyad": 3,
                            "out_dir": "runs/pilot"})

    rep_p = sub.add_parser("report", help="regenerate report for a run dir")
    rep_p.add_argument("--run-dir", required=True)

    val_p = sub.add_parser("validate-histories",
                           help="Phase 2 acceptance checks on 20 histories")
    val_p.add_argument("--n", type=int, default=20)
    val_p.add_argument("--history-mode", choices=["template", "llm"],
                       default="template")

    args = ap.parse_args(argv)

    if args.cmd == "report":
        from .eval.report import generate_report

        paths = generate_report(args.run_dir)
        print(f"report written: {paths['report_md']}")
        return 0

    if args.cmd == "validate-histories":
        from .history import validate as vmod

        sys.argv = ["validate", "--n", str(args.n), "--mode", args.history_mode]
        return vmod.main()

    cfg = _config_from_args(args)
    if cfg.mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: --mode live requires ANTHROPIC_API_KEY to be set. "
              "Use --mode mock for an offline smoke run.", file=sys.stderr)
        return 2

    from .eval.report import generate_report
    from .eval.runner import run_eval

    results = run_eval(cfg, progress=True)
    statuses = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print(f"\ncompleted {len(results)} games: {statuses}")
    paths = generate_report(cfg.out_dir)
    print(f"report written: {paths['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
