#!/usr/bin/env python3
"""
scripts/kaggle_run.py -- Kaggle orchestrator for ANN_Project.

Serial-only pipeline runner (no GPU parallelism -- single subprocess at a time)
that handles environment detection, dependency installation, training (all 4
ablations), evaluation, benchmarking, checkpoint persistence, and
cross-session resume.

Usage
-----
    # Local dry-run (see what would run without executing)
    python scripts/kaggle_run.py --dry-run --epochs 1

    # Local smoke test (1 epoch, 16 batch for RTX 4050 6GB)
    python scripts/kaggle_run.py --epochs 1 --batch-size 16

    # Local full run
    python scripts/kaggle_run.py --epochs 3 --batch-size 32

    # Kaggle full run (auto-detects environment)
    python scripts/kaggle_run.py

    # Resume a partial run (default -- skips completed ablations)
    python scripts/kaggle_run.py --resume

    # Force re-run all ablations
    python scripts/kaggle_run.py --no-resume

    # Run a single ablation
    python scripts/kaggle_run.py --ablation ablation_b
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Project root (kaggle_run.py lives in scripts/ => parent is project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ABLATIONS = ["baseline1", "ablation_a", "ablation_b", "ablation_c"]

# ── Retry helper for uploads (transient failures on Kaggle) ─────────
# Kaggle Dataset version-creation has undocumented rate limits and can
# fail transiently.  Simple exponential backoff mitigates this.


def _with_retry(fn, label: str = "operation", max_retries: int = 2, base_delay: float = 5.0) -> bool:
    """Call *fn* with retries and exponential backoff.  Returns True if *fn* succeeded."""
    for attempt in range(max_retries + 1):
        try:
            fn()
            return True
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"[RETRY] {label} failed ({e}). Retrying in {delay:.0f}s...", flush=True)
                time.sleep(delay)
            else:
                print(f"[RETRY] {label} failed after {max_retries + 1} attempts: {e}", flush=True)
    return False


# ── Three-level defensive timer (Kaggle session watchdog) ──────────
# Kaggle kills GPU sessions at ~9h (hard SIGKILL, variable ±15 min).
# Three-level cascade to prevent mid-upload corruption:
#
#   8.0 h  (COMPUTE_DEADLINE)  — Stop launching new ablation runs.
#                                Finish current run and save. Uploads still allowed.
#   8.25 h (UPLOAD_DEADLINE)   — Stop any mid-upload. Save what's uploaded.
#                                Flag partial results in run_log.json.
#   8.5 h  (HARD_ABORT)        — sys.exit(1) as last resort.
#                                Prevents Kaggle SIGKILL from corrupting in-progress uploads.
#
_COMPUTE_DEADLINE_SECONDS = 8.0 * 3600
_UPLOAD_DEADLINE_SECONDS = 8.25 * 3600
_HARD_ABORT_SECONDS = 8.5 * 3600

# Minimum wall time required to start training a single ablation (30 min).
_MIN_TRAIN_TIME_SECONDS = 30 * 60


# ---------------------------------------------------------------------------
# Section 6 -- CLI Interface
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ANN_Project Kaggle orchestrator -- serialised full-pipeline runner.",
    )
    parser.add_argument(
        "--dataset", default="raid",
        help="Dataset to use (default: raid)",
    )
    parser.add_argument(
        "--ablation", default=None, type=str,
        help="Specific ablation to run (default: all four)",
    )
    parser.add_argument(
        "--epochs", default=3, type=int,
        help="Training epochs (default: 3, use 1 for smoke test)",
    )
    parser.add_argument(
        "--batch-size", default=32, type=int, dest="batch_size",
        help="Batch size per GPU (default: 32, use 16 for RTX 4050 6 GB)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print pipeline steps without executing anything",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip completed runs by checking run_log.json (default: True)",
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Force re-run all ablations, ignoring previous run_log",
    )
    parser.add_argument(
        "--upload", action="store_true", default=None,
        help="Upload results archive via kagglehub (auto: True on Kaggle, False locally)",
    )
    parser.add_argument(
        "--no-upload", action="store_false", dest="upload",
        help="Disable kagglehub upload",
    )
    parser.add_argument(
        "--seeds", default=[42], type=int, nargs="+",
        help="Random seed(s) for training (default: 42). Pass --seeds 42 99 for multi-seed runs. "
             "Each seed runs the full ablation pipeline in serial.",
    )
    parser.add_argument(
        "--run-log-path", default=None, type=str,
        help="Path to run_log.json (default: <output_dir>/run_log.json). "
             "Use a persistent location for cross-session resume on Kaggle.",
    )
    parser.add_argument(
        "--kaggle-dataset", default="tetsujin007/ann-project-results",
        help="Kaggle Dataset handle for results archive upload (default: tetsujin007/ann-project-results)",
    )
    parser.add_argument(
        "--runlog-dataset", default=None, type=str,
        help="Kaggle Dataset handle for persistent run_log (default: auto-derived from --kaggle-dataset, "
             "e.g. tetsujin007/ann-project-runlog)",
    )
    parser.add_argument(
        "--lr", default=2e-5, type=float,
        help="Peak learning rate (default: 2e-5, passed through to train.py as --training.lr)",
    )
    parser.add_argument(
        "--weight-decay", default=0.01, type=float, dest="weight_decay",
        help="Weight decay (default: 0.01, passed through to train.py as --training.weight_decay)",
    )
    parser.add_argument(
        "--grad-clip-norm", default=1.0, type=float, dest="grad_clip_norm",
        help="Gradient clipping norm (default: 1.0, passed through to train.py as --training.grad_clip_norm)",
    )
    return parser


# ---------------------------------------------------------------------------
# Section 1 -- Environment Detection
# ---------------------------------------------------------------------------


def detect_environment() -> dict[str, Any]:
    """Detect runtime environment and return structured metadata."""
    kaggle_mode = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()

    # Torch / CUDA detection (graceful if torch not installed)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        gpu_names = (
            [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            if cuda_available else []
        )
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "N/A"
    except ImportError:
        cuda_available = False
        gpu_count = 0
        gpu_names = []
        torch_version = "not installed"
        cuda_version = "N/A"

    # Paths
    if kaggle_mode:
        output_dir = Path("/kaggle/working")
        project_root = _PROJECT_ROOT
    else:
        output_dir = _PROJECT_ROOT
        project_root = _PROJECT_ROOT

    return {
        "kaggle_mode": kaggle_mode,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "hostname": platform.node(),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
    }


def print_environment_info(env: dict[str, Any]) -> None:
    """Print environment information in a formatted block."""
    print("=" * 60, flush=True)
    print("ENVIRONMENT", flush=True)
    print("=" * 60, flush=True)
    print(f"  Kaggle mode    : {env['kaggle_mode']}", flush=True)
    print(f"  Hostname       : {env['hostname']}", flush=True)
    print(f"  Project root   : {env['project_root']}", flush=True)
    print(f"  Output dir     : {env['output_dir']}", flush=True)
    print(f"  Torch version  : {env['torch_version']}", flush=True)
    print(f"  CUDA version   : {env['cuda_version']}", flush=True)
    print(f"  CUDA available : {env['cuda_available']}", flush=True)
    print(f"  GPU count      : {env['gpu_count']}", flush=True)
    for i, name in enumerate(env["gpu_names"]):
        print(f"  GPU {i}          : {name}", flush=True)
    print(f"  Python version : {sys.version.split()[0]}", flush=True)
    print(f"  Platform       : {sys.platform}", flush=True)
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Section 2 -- Dependency Management
# ---------------------------------------------------------------------------


def install_deps(kaggle_mode: bool, requirements_path: Path) -> None:
    """Install / verify dependencies."""
    if not kaggle_mode:
        _print_package_versions()
        return

    if not requirements_path.exists():
        print(f"[DEPS] requirements.txt not found at {requirements_path}", flush=True)
        print("[DEPS] Skipping dependency installation", flush=True)
        _print_package_versions()
        return

    print(f"[DEPS] Installing dependencies from {requirements_path} ...", flush=True)
    start = time.monotonic()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    elapsed = time.monotonic() - start
    if result.returncode != 0:
        print(
            f"[DEPS] pip install failed (rc={result.returncode})"
            f" after {elapsed:.1f}s", flush=True,
        )
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-10:]:
                print(f"  ! {line}", flush=True)
        print("[DEPS] Continuing -- core deps may already be present", flush=True)
    else:
        print(f"[DEPS] Dependencies installed in {elapsed:.1f}s", flush=True)

    # Install project package in editable mode for src.* imports
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print("[DEPS] Project package installed in editable mode", flush=True)

    _print_package_versions()


def _print_package_versions() -> None:
    """Print versions of key Python packages."""
    # PyPI name to Python import name mapping
    packages: dict[str, str] = {
        "torch": "torch",
        "transformers": "transformers",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "sklearn",
        "pyyaml": "yaml",
    }
    print("[DEPS] Key package versions:", flush=True)
    for display_name, import_name in packages.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {display_name:<20s} {ver}", flush=True)
        except ImportError:
            print(f"  {display_name:<20s} NOT INSTALLED", flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# Section 3 -- Serialised Pipeline Helpers
# ---------------------------------------------------------------------------


_SESSION_START_TIME = time.monotonic()


_SESSION_STATE = {"compute_deadline": False, "upload_deadline": False}

def check_time_remaining(
    label: str,
    min_seconds: float = _MIN_TRAIN_TIME_SECONDS,
) -> bool:
    """Check whether enough session time remains for a pipeline step.

    Implements the three-level timer cascade:
      1. COMPUTE_DEADLINE – stop new runs, finish current (upload allowed)
      2. UPLOAD_DEADLINE  – abort uploads, flag partial results
      3. HARD_ABORT       – sys.exit(1) to prevent SIGKILL corruption
    """
    elapsed = time.monotonic() - _SESSION_START_TIME

    remaining_compute = _COMPUTE_DEADLINE_SECONDS - elapsed
    remaining_upload = _UPLOAD_DEADLINE_SECONDS - elapsed
    remaining_hard = _HARD_ABORT_SECONDS - elapsed

    # Level 3: Hard abort — prevent Kaggle SIGKILL from corrupting uploads
    if remaining_hard <= 0:
        print(
            f"\n[TIMER] HARD ABORT ({_HARD_ABORT_SECONDS / 3600:.1f} h) — "
            "Session expired. Exiting immediately to prevent SIGKILL corruption.",
            flush=True,
        )
        sys.exit(1)

    # Level 2: Upload deadline — abort mid-upload, keep what's saved
    if remaining_upload <= 0:
        print(
            f"\n[TIMER] UPLOAD_DEADLINE ({_UPLOAD_DEADLINE_SECONDS / 3600:.1f} h) — "
            "Aborting uploads. Saving partial results.",
            flush=True,
        )
        _SESSION_STATE["upload_deadline"] = True
        _SESSION_STATE["compute_deadline"] = True
        return False

    # Level 1: Compute deadline — finish current run, no new starts
    if remaining_compute <= 0:
        print(
            f"\n[TIMER] COMPUTE_DEADLINE ({_COMPUTE_DEADLINE_SECONDS / 3600:.1f} h) — "
            "No new training runs. Finishing current. Uploads still allowed.",
            flush=True,
        )
        _SESSION_STATE["compute_deadline"] = True
        return False

    # Standard: not enough time for this pipeline step
    if remaining_hard < min_seconds:
        print(
            f"\n[TIMER] Only {remaining_hard / 3600:.2f} h remaining,"
            f" but '{label}' needs ~{min_seconds / 3600:.2f} h.", flush=True,
        )
        print("[TIMER] Deferring to next session.", flush=True)
        return False

    print(
        f"[TIMER] ~{remaining_hard / 3600:.2f} h remaining. Proceeding with '{label}'.",
        flush=True,
    )
    return True


def run_pipeline_step(
    step_name: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: int | None = 43200,  # 12 h
) -> int:
    """Run a pipeline subprocess step with progress reporting."""
    print(f"\n{'-' * 60}", flush=True)
    print(f"STEP : {step_name}", flush=True)
    print(f"CMD  : {' '.join(str(c) for c in cmd)}", flush=True)
    print(f"{'-' * 60}", flush=True)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    step_start = time.monotonic()

    try:
        result = subprocess.run(
            cmd, env=merged_env, cwd=str(cwd) if cwd else None, timeout=timeout,
        )
        elapsed = time.monotonic() - step_start
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"[{step_name}] {status} in {elapsed:.1f}s", flush=True)
        return result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - step_start
        print(f"[{step_name}] TIMEOUT after {elapsed:.1f}s (limit={timeout}s)", flush=True)
        return -1
    except FileNotFoundError as e:
        print(f"[{step_name}] Command not found: {e}", flush=True)
        return -2
    except Exception as e:
        print(f"[{step_name}] Unexpected error: {e}", flush=True)
        return -3


def find_checkpoint(
    artifact_dir: Path, dataset: str, ablation: str, seed: int = 42,
) -> Path | None:
    """Locate a trained checkpoint for dataset + ablation + seed."""
    # Seed-specific checkpoint (multi-seed runs)
    ckpt = artifact_dir / f"{dataset}_{ablation}_seed{seed}_best.pt"
    if ckpt.exists():
        return ckpt
    # Legacy: no seed in name (default seed=42)
    ckpt = artifact_dir / f"{dataset}_{ablation}_best.pt"
    if ckpt.exists():
        return ckpt
    # Legacy fallback (checkpoint without dataset prefix)
    ckpt = artifact_dir / f"{ablation}_best.pt"
    if ckpt.exists():
        return ckpt
    return None


# ---------------------------------------------------------------------------
# Section 4 -- Cross-Session Resume
# ---------------------------------------------------------------------------


def load_run_log(log_path: Path) -> dict[str, Any]:
    """Load the run log from disk."""
    if not log_path.exists():
        return {"sessions": [], "completed_ablations": {}}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[RESUME] Warning: corrupt run log ({e}). Starting fresh.", flush=True)
        return {"sessions": [], "completed_ablations": {}}


def save_run_log(log_path: Path, run_log: dict[str, Any]) -> None:
    """Persist the run log to disk with an atomic write pattern.

    Writes to a ``.tmp`` file first, then renames to *log_path* so a crash
    mid-write never corrupts the existing log.  If ``.replace()`` is atomic
    on the host filesystem (POSIX; near-atomic on NTFS) the log is always
    either the old complete version or the new complete version.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = log_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, default=str)
    tmp_path.replace(log_path)
    print(f"[RESUME] Run log saved to {log_path}", flush=True)


def _download_previous_run_log(
    kaggle_mode: bool,
    runlog_dataset_handle: str,
    run_log_path: Path,
) -> None:
    """Fetch the previous session's run_log.json from the persistent runlog Dataset.

    Unlike the main results Dataset (uploaded once at the very end), this
    lightweight Dataset is updated after EACH completed ablation, so a
    session kill at any point loses at most one ablation's worth of work.

    Silent on first session (no previous upload).  Failures are logged but
    never fatal — falls back to a fresh start.
    """
    if not kaggle_mode:
        return
    if run_log_path.exists():
        return

    print(
        f"[RESUME] Checking runlog Dataset {runlog_dataset_handle} ...",
        flush=True,
    )
    try:
        import kagglehub

        download_path = Path(kagglehub.dataset_download(runlog_dataset_handle))
        prev_log = download_path / "run_log.json"

        if prev_log.exists():
            run_log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(prev_log), str(run_log_path))
            n_completed = len(
                json.loads(prev_log.read_text()).get("completed_ablations", {})
            )
            print(
                f"[RESUME] Restored from {runlog_dataset_handle}"
                f" ({n_completed} completed ablations -- will skip those)",
                flush=True,
            )
        else:
            print("[RESUME] No run_log.json in runlog Dataset (first session).", flush=True)
    except Exception as e:
        print(f"[RESUME] Could not fetch previous run log: {e}", flush=True)
        print("[RESUME] Starting fresh session.", flush=True)


def _upload_run_log(runlog_dataset_handle: str, run_log_path: Path) -> None:
    """Upload run_log.json to the persistent runlog Kaggle Dataset.

    Called after EACH completed ablation so the run log survives session
    kills.  The upload creates a new Dataset version with just the tiny
    JSON file (~2 KB) — fast and cheap.  ``_download_previous_run_log``
    fetches it on the next session's startup.
    """
    if _SESSION_STATE.get("upload_deadline"):
        return
    if not run_log_path.exists():
        return
    print(
        f"[RESUME] Uploading run log to {runlog_dataset_handle} ...",
        flush=True,
    )
    runlog_dir: Path | None = None
    try:
        import kagglehub
        import tempfile

        runlog_dir = Path(tempfile.mkdtemp(prefix="ann_project_runlog_"))
        shutil.copy2(str(run_log_path), str(runlog_dir / "run_log.json"))

        kagglehub.dataset_upload(
            handle=runlog_dataset_handle,
            local_dataset_dir=str(runlog_dir),
            version_notes="Run log checkpoint (auto-upload after ablation)",
        )
    except Exception as e:
        print(f"[RESUME] Run log upload failed (non-fatal): {e}", flush=True)
    finally:
        if runlog_dir is not None:
            shutil.rmtree(str(runlog_dir), ignore_errors=True)


def _upload_results_snapshot(
    kaggle_mode: bool,
    results_dir: Path,
    run_log_path: Path,
    dataset_handle: str,
) -> None:
    """Upload eval JSONs + run_log.json to the main results Dataset.

    Called after EACH ablation so the actual metrics survive session kills.
    Each upload creates a new Dataset version containing ALL eval JSONs
    accumulated so far (not just the latest one) plus the run_log.
    Eval files are ~2 KB each — the upload takes ~5 s.

    Silently no-ops when ``kaggle_mode`` is False (local execution).
    """
    if _SESSION_STATE.get("upload_deadline"):
        return
    if not kaggle_mode:
        return
    if not results_dir.exists() and not run_log_path.exists():
        return
    print(
        f"[SNAPSHOT] Saving eval results to {dataset_handle} ...",
        flush=True,
    )
    snap_dir: Path | None = None
    try:
        import kagglehub
        import tempfile

        snap_dir = Path(tempfile.mkdtemp(prefix="ann_project_snap_"))

        # Copy run_log
        if run_log_path.exists():
            shutil.copy2(str(run_log_path), str(snap_dir / "run_log.json"))

        # Copy all eval JSONs accumulated so far
        snap_results = snap_dir / "results"
        if results_dir.exists():
            snap_results.mkdir(parents=True, exist_ok=True)
            for f in sorted(results_dir.iterdir()):
                if f.suffix == ".json":
                    shutil.copy2(str(f), str(snap_results / f.name))

        kagglehub.dataset_upload(
            handle=dataset_handle,
            local_dataset_dir=str(snap_dir),
            version_notes="Eval results snapshot (per-ablation)",
        )
        print("[SNAPSHOT] Upload complete.", flush=True)
    except Exception as e:
        print(f"[SNAPSHOT] Upload failed (non-fatal): {e}", flush=True)
    finally:
        if snap_dir is not None:
            shutil.rmtree(str(snap_dir), ignore_errors=True)


def _restore_results_snapshot(
    kaggle_mode: bool,
    dataset_handle: str,
    results_dir: Path,
    run_log_path: Path,
) -> None:
    """Download the latest eval JSONs + run_log from the results Dataset.

    Runs at startup so previously completed ablations' metrics are
    available locally — even if the checkpoints are gone.
    """
    if not kaggle_mode:
        return
    if results_dir.exists() and any(results_dir.iterdir()):
        # Results already present (within-session resume)
        return
    print(
        f"[SNAPSHOT] Checking {dataset_handle} for previous results ...",
        flush=True,
    )
    try:
        import kagglehub

        download_path = Path(kagglehub.dataset_download(dataset_handle))

        # Restore eval JSONs
        dl_results = download_path / "results"
        if dl_results.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
            restored = 0
            for f in sorted(dl_results.iterdir()):
                if f.suffix == ".json" and not f.name.startswith("."):
                    dst = results_dir / f.name
                    shutil.copy2(str(f), str(dst))
                    restored += 1
            if restored > 0:
                print(f"[SNAPSHOT] Restored {restored} eval JSONs from Dataset", flush=True)

        # Restore run_log from results Dataset (backup source)
        dl_runlog = download_path / "run_log.json"
        if dl_runlog.exists() and not run_log_path.exists():
            run_log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(dl_runlog), str(run_log_path))
            print("[SNAPSHOT] Restored run_log.json from results Dataset", flush=True)

    except Exception as e:
        print(f"[SNAPSHOT] Could not restore previous results: {e}", flush=True)


def is_ablation_completed(
    run_log: dict[str, Any], dataset: str, ablation: str,
    epochs: int, batch_size: int, seed: int,
    lr: float | None = None, weight_decay: float | None = None,
    grad_clip_norm: float | None = None,
) -> bool:
    """Check whether an ablation was already completed with matching params.

    The run-log key includes the seed so that each seed+ablation combination
    is tracked independently.  Compares *epochs*, *batch_size*, *seed*, and
    hyperparameters (*lr*, *weight_decay*, *grad_clip_norm*) so that
    changing any of them triggers a re-run.
    """
    completed: dict = dict(run_log.get("completed_ablations") or {})
    entry = completed.get(f"{dataset}_{ablation}_seed{seed}")
    if entry is None:
        return False
    base_ok = (
        entry.get("epochs") == epochs
        and entry.get("batch_size") == batch_size
        and entry.get("seed") == seed
    )
    if not base_ok:
        return False
    # Optional hyperparameter checks (added in v2 — older entries won't have them)
    hp_checks = {
        "lr": lr, "weight_decay": weight_decay, "grad_clip_norm": grad_clip_norm,
    }
    for hp_name, hp_value in hp_checks.items():
        if hp_value is not None:
            stored = entry.get(hp_name)
            if stored is not None and stored != hp_value:
                return False
    return True


def mark_ablation_completed(
    run_log: dict[str, Any],
    dataset: str,
    ablation: str,
    epochs: int,
    batch_size: int,
    seed: int,
    checkpoint: str | None,
    training_time: float,
    eval_output: str | None,
    benchmark_ok: bool,
    lr: float | None = None,
    weight_decay: float | None = None,
    grad_clip_norm: float | None = None,
) -> None:
    """Record an ablation as completed in the run log.

    The key includes the seed so that multi-seed runs are tracked
    independently per seed+ablation combination.  Stores hyperparameters
    so that *is_ablation_completed* can detect config drift.
    """
    run_log.setdefault("completed_ablations", {})
    key = f"{dataset}_{ablation}_seed{seed}"
    run_log["completed_ablations"][key] = {
        "dataset": dataset,
        "ablation": ablation,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "checkpoint": checkpoint or "",
        "training_time_seconds": round(training_time, 2),
        "evaluation_output": eval_output or "",
        "benchmark_completed": benchmark_ok,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 5 -- Main Pipeline Sequence
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Environment detection
    env = detect_environment()
    print_environment_info(env)

    kaggle_mode = env["kaggle_mode"]
    project_root = Path(env["project_root"])
    output_dir = Path(env["output_dir"])

    if args.upload is None:
        args.upload = kaggle_mode

    if args.seeds != [42]:
        if len(args.seeds) == 1:
            print(f"[CONFIG] Overriding seed: {args.seeds[0]}", flush=True)
        else:
            print(f"[CONFIG] Multi-seed run: seeds={args.seeds}", flush=True)

    dataset = args.dataset
    ablation_names = [args.ablation] if args.ablation else list(_DEFAULT_ABLATIONS)

    # Directory layout
    # Checkpoints are saved by train.py relative to project_root (CWD at subprocess time),
    # so artifact_dir must point there, not to output_dir.
    artifact_dir = project_root / "artifacts" / "distilbert_detector"
    results_dir = output_dir / "results"
    run_log_path = Path(args.run_log_path) if args.run_log_path else output_dir / "run_log.json"
    data_dir = output_dir / "data" / "processed"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Derive runlog dataset handle (e.g. tetsujin007/ann-project-runlog)
    if args.runlog_dataset:
        runlog_dataset_handle = args.runlog_dataset
    else:
        owner = args.kaggle_dataset.split("/")[0]
        runlog_dataset_handle = f"{owner}/ann-project-runlog"

    # 2. Install deps (Kaggle only)
    requirements_path = project_root / "requirements.txt"
    install_deps(kaggle_mode, requirements_path)

    # 2b. Data setup (Kaggle only)
    # Always run in Kaggle mode — idempotent internally (skips existing files).
    # The guard must not check data_dir.exists() because that directory may
    # pre-exist from a prior session while the project_root mirror is missing.
    if kaggle_mode:
        _setup_kaggle_data(project_root, data_dir)

    # 2c. Cross-session resume: fetch previous run_log from persistent runlog Dataset
    _download_previous_run_log(kaggle_mode, runlog_dataset_handle, run_log_path)

    # 2d. Restore previous eval results from results Dataset
    _restore_results_snapshot(kaggle_mode, args.kaggle_dataset, results_dir, run_log_path)

    # 3. Load run log for resume
    run_log = load_run_log(run_log_path)
    current_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    session_info: dict[str, Any] = {
        "session_id": current_session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "kaggle_mode": kaggle_mode,
        "gpu_count": env["gpu_count"],
        "gpu_names": env["gpu_names"],
        "ablations_run": [],
    }

    # 4. Pipeline loop (train -> evaluate -> benchmark per ablation)
    seeds = args.seeds
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"PIPELINE : {len(ablation_names)} ablation(s)  |"
        f"  epochs={args.epochs}  batch_size={args.batch_size}  dataset={dataset}",
        flush=True,
    )
    print(f"  SEEDS   : {len(seeds)} seed(s)  |  {seeds}", flush=True)
    if args.dry_run:
        print("DRY RUN  : no commands will execute", flush=True)
    else:
        print("EXECUTING : serialised pipeline", flush=True)
    print(f"{'=' * 60}", flush=True)

    for current_seed in seeds:
        print(f"\n{'~' * 60}", flush=True)
        print(f"SEED={current_seed}  (starting seed loop iteration)", flush=True)
        print(f"{'~' * 60}", flush=True)

        for ablation_name in ablation_names:
            print(f"\n{'#' * 60}", flush=True)
            print(f"# ABLATION : {ablation_name}", flush=True)
            print(f"{'#' * 60}", flush=True)

            # Resume check
            if args.resume and is_ablation_completed(
                run_log, dataset, ablation_name,
                args.epochs, args.batch_size, current_seed,
                lr=args.lr, weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
            ):
                print(
                    f"[RESUME] {dataset}_{ablation_name} already completed"
                    f" (epochs={args.epochs}, batch={args.batch_size}, seed={current_seed}). Skipping.",
                    flush=True,
                )
                continue

            # Defensive timer
            if not check_time_remaining(f"train/{ablation_name}"):
                session_info["incomplete"] = True
                break

            # Build config overrides for train.py
            train_config_args = [
                f"--training.epochs={args.epochs}",
                f"--training.batch_size={args.batch_size}",
                f"--training.seed={current_seed}",
                f"--training.lr={args.lr}",
                f"--training.weight_decay={args.weight_decay}",
                f"--training.grad_clip_norm={args.grad_clip_norm}",
            ]

            # Train
            train_cmd = [
                sys.executable, "-u", "scripts/train.py",
                "--dataset", dataset,
                "--ablation", ablation_name,
                *train_config_args,
            ]

            train_ok = False
            training_time = 0.0
            if args.dry_run:
                print(f"[DRY-RUN] train cmd:       {' '.join(train_cmd)}", flush=True)
                train_ok = True
            else:
                t0 = time.monotonic()
                rc = run_pipeline_step(f"train/{ablation_name}", train_cmd, cwd=project_root)
                training_time = time.monotonic() - t0
                train_ok = rc == 0

            # Locate the resulting checkpoint
            checkpoint_path: Path | None = (
                artifact_dir / f"{dataset}_{ablation_name}_seed{current_seed}_best.pt"
                if args.dry_run
                else find_checkpoint(artifact_dir, dataset, ablation_name, current_seed)
            )

            if checkpoint_path is None:
                # No checkpoint -- nothing to salvage.  Do NOT mark completed so
                # that --resume will re-attempt this ablation from scratch.
                if not args.dry_run:
                    print(
                        f"[WARN] No checkpoint for {ablation_name} after training."
                        " Not marking completed -- will retry on next --resume run.",
                        flush=True,
                    )
                if not args.dry_run:
                    save_run_log(run_log_path, run_log)
                    _with_retry(lambda: _upload_run_log(runlog_dataset_handle, run_log_path), "runlog upload")
                    _with_retry(lambda: _upload_results_snapshot(kaggle_mode, results_dir, run_log_path, args.kaggle_dataset), "results snapshot")
                session_info["ablations_run"].append({
                    "ablation": ablation_name,
                    "checkpoint": None,
                    "training_ok": train_ok, "eval_ok": False, "benchmark_ok": False,
                })
                continue

            if not train_ok:
                if not args.dry_run:
                    print(
                        f"[WARN] Training for {ablation_name} returned non-zero."
                        " Not marking completed -- will retry on next --resume run.",
                        flush=True,
                    )
                if not args.dry_run:
                    save_run_log(run_log_path, run_log)
                    _with_retry(lambda: _upload_run_log(runlog_dataset_handle, run_log_path), "runlog upload")
                    _with_retry(lambda: _upload_results_snapshot(kaggle_mode, results_dir, run_log_path, args.kaggle_dataset), "results snapshot")
                session_info["ablations_run"].append({
                    "ablation": ablation_name,
                    "checkpoint": str(checkpoint_path),
                    "training_ok": False, "eval_ok": False, "benchmark_ok": False,
                })
                continue

            # Evaluate
            eval_output_path = results_dir / f"{dataset}_{ablation_name}_seed{current_seed}_eval.json"
            eval_cmd = [
                sys.executable, "-u", "scripts/evaluate.py",
                "--dataset", dataset,
                "--seed", str(current_seed),
                "--checkpoint", str(checkpoint_path.resolve()),
                "--output", str(eval_output_path.resolve()),
            ]

            eval_ok = False
            if args.dry_run:
                print(f"[DRY-RUN] eval cmd:       {' '.join(eval_cmd)}", flush=True)
                eval_ok = True
            else:
                rc = run_pipeline_step(f"eval/{ablation_name}", eval_cmd, cwd=project_root)
                eval_ok = rc == 0

            # Benchmark (only for ablation_b -- benchmark.py is hardcoded)
            benchmark_ok: bool | None = None
            if ablation_name == "ablation_b":
                benchmark_cmd = [sys.executable, "-u", "scripts/benchmark.py"]
                if args.dry_run:
                    print(f"[DRY-RUN] benchmark cmd: {' '.join(benchmark_cmd)}", flush=True)
                    benchmark_ok = True
                else:
                    rc = run_pipeline_step(f"benchmark/{ablation_name}", benchmark_cmd, cwd=project_root)
                    benchmark_ok = rc == 0
            else:
                msg = f"[BENCH] benchmark.py is hardcoded to ablation_b; skipping for {ablation_name}"
                if args.dry_run:
                    print(f"[DRY-RUN] {msg}", flush=True)
                else:
                    print(msg, flush=True)
                benchmark_ok = False

            # Persist run log
            if not args.dry_run:
                if eval_ok:
                    mark_ablation_completed(
                        run_log, dataset, ablation_name,
                        args.epochs, args.batch_size, current_seed,
                        checkpoint=str(checkpoint_path),
                        training_time=training_time,
                        eval_output=str(eval_output_path),
                        benchmark_ok=bool(benchmark_ok),
                        lr=args.lr, weight_decay=args.weight_decay,
                        grad_clip_norm=args.grad_clip_norm,
                    )
                else:
                    print(
                        f"[WARN] Evaluation for {ablation_name} failed."
                        " Not marking completed -- will retry on next --resume run.",
                        flush=True,
                    )
                save_run_log(run_log_path, run_log)
                _with_retry(lambda: _upload_run_log(runlog_dataset_handle, run_log_path), "runlog upload")
                _with_retry(lambda: _upload_results_snapshot(kaggle_mode, results_dir, run_log_path, args.kaggle_dataset), "results snapshot")

            session_info["ablations_run"].append({
                "ablation": ablation_name,
                "checkpoint": str(checkpoint_path),
                "training_ok": train_ok,
                "eval_ok": eval_ok,
                "benchmark_ok": benchmark_ok,
            })

    # 5. Save final run log
    if not args.dry_run:
        session_info["completed_at"] = datetime.now(timezone.utc).isoformat()
        run_log.setdefault("sessions", []).append(session_info)
        save_run_log(run_log_path, run_log)
        _with_retry(lambda: _upload_run_log(runlog_dataset_handle, run_log_path), "runlog upload")
        _with_retry(lambda: _upload_results_snapshot(kaggle_mode, results_dir, run_log_path, args.kaggle_dataset), "results snapshot")

    # 6. Upload outputs (Kaggle only)
    if not args.dry_run and kaggle_mode and args.upload:
        upload_outputs(output_dir, args.kaggle_dataset, artifact_dir, results_dir, run_log_path)

    # 7. Final summary
    _print_final_summary(session_info, args.dry_run)


# ---------------------------------------------------------------------------
# Helper: Data setup on Kaggle
# ---------------------------------------------------------------------------


def _setup_kaggle_data(project_root: Path, data_dir: Path) -> None:
    """Copy preprocessed parquet data from /kaggle/input/ to data/processed/."""
    print("[DATA] Looking for preprocessed data in /kaggle/input/ ...", flush=True)
    input_base = Path("/kaggle/input")
    if not input_base.exists():
        print("[DATA] /kaggle/input/ not found. Data setup skipped.", flush=True)
        return

    parquet_files = sorted(input_base.rglob("raid_train_pool.parquet"))
    if not parquet_files:
        print(
            "[DATA] No raid_train_pool.parquet found in /kaggle/input/.\n"
            "[DATA] Attach the preprocessed RAID dataset and re-run.", flush=True,
        )
        return

    src = parquet_files[0]
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] Copying parquet files from {src.parent} ...", flush=True)
    for f in sorted(src.parent.iterdir()):
        if f.suffix == ".parquet":
            dst = data_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                print(f"  {f.name} -> {dst}", flush=True)

    print(f"[DATA] Data ready at {data_dir}", flush=True)

    # Mirror at project_root/data/processed (train.py resolves relative to ROOT_DIR)
    project_data_dir = project_root / "data" / "processed"
    if project_data_dir != data_dir:
        project_data_dir.mkdir(parents=True, exist_ok=True)
        for f in data_dir.iterdir():
            dst = project_data_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
        print(f"[DATA] Mirror at project_root {project_data_dir}", flush=True)


# ---------------------------------------------------------------------------
# Helper: Upload via kagglehub
# ---------------------------------------------------------------------------


def upload_outputs(
    output_dir: Path,
    dataset_handle: str,
    artifact_dir: Path,
    results_dir: Path,
    run_log_path: Path | None = None,
) -> None:
    """Upload results/ + artifacts/ + run_log.json to Kaggle Dataset via kagglehub.

    Creates a temp directory with copies of all outputs, then calls
    ``kagglehub.dataset_upload()`` which expects a **directory** (not a file
    path). Also creates a ``.tar.gz`` archive as a manual-download fallback.

    Falls back gracefully if ``kagglehub`` is not installed or the upload fails.
    """
    if _SESSION_STATE.get("upload_deadline"):
        print("[UPLOAD] UPLOAD_DEADLINE reached — skipping full archive upload.", flush=True)
        return
    run_log = run_log_path if run_log_path else output_dir / "run_log.json"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"ann_project_run_{timestamp}.tar.gz"
    archive_path = output_dir / archive_name

    # ── Create archive (manual fallback) ──────────────────────────────────
    print(f"[UPLOAD] Creating archive: {archive_path}", flush=True)
    try:
        with tarfile.open(str(archive_path), "w:gz") as tar:
            for path, arcname in [
                (results_dir, "results"),
                (artifact_dir, "artifacts"),
                (run_log, "run_log.json"),
            ]:
                if path.exists():
                    tar.add(str(path), arcname=arcname)

        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"[UPLOAD] Archive size: {size_mb:.1f} MB", flush=True)
    except Exception as e:
        print(f"[UPLOAD] Failed to create archive: {e}", flush=True)
        return

    if not archive_path.exists():
        print(f"[UPLOAD] Archive not found: {archive_path}. Upload skipped.", flush=True)
        return

    # ── Upload via kagglehub (uses a temp directory, not the archive path) ──
    print("[UPLOAD] Attempting kagglehub upload ...", flush=True)
    temp_upload_dir: Path | None = None
    try:
        import kagglehub  # type: ignore[import-untyped]
        import tempfile

        # kagglehub.dataset_upload expects a *directory* path, not a file.
        # Create a temp dir and copy output files into it.
        temp_upload_dir = Path(tempfile.mkdtemp(prefix="ann_project_upload_"))
        for src_path, rel_dest in [
            (results_dir, "results"),
            (artifact_dir, "artifacts"),
            (run_log, "run_log.json"),
        ]:
            if src_path.exists():
                dest = temp_upload_dir / rel_dest
                if src_path.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_path), str(dest))
                elif src_path.is_dir():
                    shutil.copytree(str(src_path), str(dest), dirs_exist_ok=True)

        print(f"[UPLOAD] Uploading to Kaggle dataset: {dataset_handle}", flush=True)
        kagglehub.dataset_upload(
            handle=dataset_handle,
            local_dataset_dir=str(temp_upload_dir),
            version_notes=f"Auto-upload from Kaggle pipeline run {timestamp}",
        )
        print("[UPLOAD] Upload complete.", flush=True)

    except ImportError:
        print(
            "[UPLOAD] kagglehub not installed. To enable automatic upload:\n"
            "    pip install kagglehub\n"
            f"[UPLOAD] Archive saved at: {archive_path}", flush=True,
        )
    except Exception as e:
        print(f"[UPLOAD] kagglehub upload failed: {e}", flush=True)
        print(f"[UPLOAD] Archive saved at: {archive_path}", flush=True)
    finally:
        if temp_upload_dir is not None:
            shutil.rmtree(str(temp_upload_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Helper: Final summary
# ---------------------------------------------------------------------------


def _print_final_summary(session_info: dict[str, Any], dry_run: bool) -> None:
    """Print a final summary table of the pipeline run."""
    print(f"\n{'=' * 60}", flush=True)
    if dry_run:
        print("DRY RUN COMPLETE -- no commands were executed", flush=True)
    else:
        print("PIPELINE COMPLETE", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Dataset          : {session_info.get('dataset', 'N/A')}", flush=True)
    print(f"  Epochs           : {session_info.get('epochs', 'N/A')}", flush=True)
    print(f"  Batch size       : {session_info.get('batch_size', 'N/A')}", flush=True)

    ablations = session_info.get("ablations_run", [])
    if ablations:
        print(f"  Ablations run    : {len(ablations)}", flush=True)
        for a in ablations:
            parts = []
            if a.get("training_ok"):
                parts.append("TRAINED")
            if a.get("eval_ok"):
                parts.append("EVALED")
            if a.get("benchmark_ok"):
                parts.append("BENCHED")
            status = ", ".join(parts) if parts else "FAILED"
            print(f"    {a['ablation']:12s}: {status}", flush=True)

    if session_info.get("incomplete"):
        print(
            "\n  Pipeline incomplete -- some ablations remain.", flush=True,
        )
        print(
            "  Resume by re-running with --resume (default behaviour).", flush=True,
        )

    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()