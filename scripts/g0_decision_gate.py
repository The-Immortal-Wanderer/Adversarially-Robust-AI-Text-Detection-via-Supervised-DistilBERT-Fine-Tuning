"""
scripts/g0_decision_gate.py — Pre-G1 decision gate

Runs evaluate.py for each of 4 ablation configs (baseline1, ablation_a,
ablation_b, ablation_c), collects F1 seen/unseen scores, computes RRD
as a percentage via compute_rrd() = (F1_seen - F1_unseen) / F1_seen × 100,
and makes a go/no-go decision: GO if the RRD spread (max − min) across all
4 ablations is less than the configured threshold (RRD_SPREAD_THRESHOLD = 6.0%), else NO-GO (trigger fallback
narrative).

Usage
-----
    python scripts/g0_decision_gate.py
    python scripts/g0_decision_gate.py --dataset raid --seed 123
    python scripts/g0_decision_gate.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ importable via pip install -e .

from src.evaluation.metrics import compute_rrd  # noqa: E402

ABLATIONS: list[str] = ["baseline1", "ablation_a", "ablation_b", "ablation_c"]
RRD_SPREAD_THRESHOLD: float = 6.0


def _build_checkpoint_path(dataset: str, ablation: str, checkpoint_dir: Path, seed: int = 42) -> Path:
    """Build checkpoint path with fallback to legacy naming patterns."""
    candidates = [
        checkpoint_dir / f"{dataset}_{ablation}_seed{seed}_best.pt",
        checkpoint_dir / f"{dataset}_{ablation}_best.pt",
        checkpoint_dir / f"{ablation}_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]  # Return primary path even if missing (caller handles MISSING)


def _run_evaluate(
    dataset: str,
    ablation: str,
    checkpoint: Path,
    output: Path,
    seed: int,
    dry_run: bool,
) -> dict | None:
    """Run evaluate.py for a single ablation and return the parsed JSON result."""
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--dataset", dataset,
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--seed", str(seed),
    ]

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return None

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))

    if result.returncode != 0:
        print(f"  WARNING: evaluate.py exited with code {result.returncode} for {ablation}", file=sys.stderr)
        if result.stderr.strip():
            print(f"    stderr: {result.stderr.strip()}", file=sys.stderr)
        return None

    if not output.exists():
        print(f"  WARNING: output file not found after evaluation: {output}", file=sys.stderr)
        return None

    with open(output, encoding="utf-8") as f:
        return json.load(f)


def _print_summary_table(results: list[dict]) -> None:
    """Print a clean summary table to stdout."""
    print()
    print("=" * 78)
    header = f"{'Ablation':<20s} {'F1 (seen)':>12s} {'F1 (unseen)':>12s} {'RRD':>12s} {'Status':>12s}"
    print(f"  {header}")
    print("  " + "-" * 20 + " " + "-" * 12 + " " + "-" * 12 + " " + "-" * 12 + " " + "-" * 12)
    for r in results:
        seen_str = f"{r['f1_seen']:.6f}" if r["f1_seen"] is not None else "N/A"
        unseen_str = f"{r['f1_unseen']:.6f}" if r["f1_unseen"] is not None else "N/A"
        rrd_str = f"{r['rrd']:.6f}" if r["rrd"] is not None else "N/A"
        print(f"  {r['ablation']:<20s} {seen_str:>12s} {unseen_str:>12s} {rrd_str:>12s} {r['status']:>12s}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-G1 decision gate — run 4 ablations and evaluate go/no-go.",
    )
    parser.add_argument(
        "--dataset",
        default="raid",
        help="Dataset name (default: raid).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/distilbert_detector",
        help="Directory containing checkpoint .pt files (default: artifacts/distilbert_detector; on Kaggle use ann-project-runlog/artifacts/).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to store per-ablation eval outputs and g0 decision JSON (default: results).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluate.py commands without executing them.",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation for each ablation ──────────────────────────────────
    ablation_results: list[dict] = []

    for ablation in ABLATIONS:
        ckpt = _build_checkpoint_path(args.dataset, ablation, checkpoint_dir, args.seed)

        if not ckpt.exists():
            print(f"  SKIP: checkpoint not found — {ckpt}")
            ablation_results.append({
                "ablation": ablation,
                "f1_seen": None,
                "f1_unseen": None,
                "rrd": None,
                "status": "MISSING",
            })
            continue

        output_path = results_dir / f"{args.dataset}_{ablation}_eval.json"
        data = _run_evaluate(args.dataset, ablation, ckpt, output_path, args.seed, args.dry_run)

        if data is None:
            status = "DRY-RUN" if args.dry_run else "FAILED"
            ablation_results.append({
                "ablation": ablation,
                "f1_seen": None,
                "f1_unseen": None,
                "rrd": None,
                "status": status,
            })
            continue

        f1_seen = float(data.get("test_metrics", {}).get("f1_macro", float("nan")))
        f1_unseen = float(data.get("unseen_metrics", {}).get("f1_macro", float("nan")))
        rrd = compute_rrd(f1_seen, f1_unseen)

        ablation_results.append({
            "ablation": ablation,
            "f1_seen": f1_seen,
            "f1_unseen": f1_unseen,
            "rrd": rrd,
            "status": "OK",
        })

    # ── Compute RRD spread ────────────────────────────────────────────────
    rrd_values = [r["rrd"] for r in ablation_results if r["rrd"] is not None]
    rrd_spread: float | None = max(rrd_values) - min(rrd_values) if rrd_values else None

    # ── Go / no-go decision ───────────────────────────────────────────────
    threshold: float = RRD_SPREAD_THRESHOLD
    if rrd_spread is None:
        decision = "NO-GO"
    else:
        decision = "GO" if rrd_spread < threshold else "NO-GO"

    if rrd_spread is None:
        msg = (
            "No valid ablation results available — cannot compute RRD spread. "
            "Trigger fallback: check checkpoint availability and training pipeline."
        )
    elif decision == "GO":
        msg = "Proceed to G1 multi-seed evaluation."
    else:
        msg = (
            f"RRD spread {rrd_spread:.6f} exceeds threshold {threshold}. "
            "Trigger fallback: review ablation stability, check training convergence, "
            "or consider model retraining before proceeding to G1."
        )

    # ── Print summary ─────────────────────────────────────────────────────
    has_real_results = any(r["status"] not in ("DRY-RUN",) for r in ablation_results)
    if not args.dry_run or has_real_results:
        _print_summary_table(ablation_results)

    spread_str = f"{rrd_spread:.6f}" if rrd_spread is not None else "N/A"
    print(f"\n  RRD spread      : {spread_str}")
    print(f"  Threshold       : {threshold}")
    print(f"  Decision        : {decision}")
    print(f"  Message         : {msg}")

    # ── Save structured JSON ──────────────────────────────────────────────
    output = {
        "ablation_results": ablation_results,
        "rrd_spread": rrd_spread,
        "threshold": threshold,
        "decision": decision,
        "fallback_message": msg,
    }

    output_path = results_dir / "g0_decision.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()