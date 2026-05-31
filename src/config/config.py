"""Configuration loader for ANN_Project.

Provides:
  - ``TrainingConfig`` — top-level dataclass matching ``src/config/default.yaml``.
  - ``load_config(path=None)`` — reads YAML, validates types, applies CLI overrides.

Dependencies
------------
- ``pyyaml`` (>= 6.0)

Usage
-----
::

    from src.config import load_config, TrainingConfig

    cfg: TrainingConfig = load_config()
    print(cfg.training.batch_size)   # 32
    print(cfg.data.dataset)          # "raid"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

import yaml

# ── Project root detection ────────────────────────────────────────────────────
# config/ is three levels below project root → parent.parent.parent
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Data-related configuration."""

    dataset: str = "raid"
    tokenization_mode: str = "on_the_fly"
    processed_dir: Optional[str] = None
    samples_per_class: int = 60_000
    unseen_cap: int = 10_000
    max_length: int = 256


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters."""

    seed: int = 42
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_workers: int = 6
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_amp: bool = False  # Enable AMP (FP16 autocast + GradScaler) for Tensor Cores


@dataclass
class AblationConfig:
    """A single ablation arm configuration."""

    head_type: str = "single"  # "single" | "deep"
    freeze_layers: int = 0     # 0–6 (DistilBERT has 6 transformer layers)


@dataclass
class AblationsConfig:
    """All four ablation arms defined in the 2×2 study design."""

    baseline1: AblationConfig = field(
        default_factory=lambda: AblationConfig(head_type="single", freeze_layers=0)
    )
    ablation_a: AblationConfig = field(
        default_factory=lambda: AblationConfig(head_type="single", freeze_layers=4)
    )
    ablation_b: AblationConfig = field(
        default_factory=lambda: AblationConfig(head_type="deep", freeze_layers=4)
    )
    ablation_c: AblationConfig = field(
        default_factory=lambda: AblationConfig(head_type="deep", freeze_layers=0)
    )


@dataclass
class PathConfig:
    """Filesystem paths for artifacts and checkpoints."""

    artifact_dir: str = "artifacts/distilbert_detector"
    checkpoint_fallbacks: List[str] = field(
    default_factory=lambda: [
        # Seed-42 variants (primary fallback)
        "artifacts/distilbert_detector/raid_baseline1_seed42_best.pt",
        "artifacts/distilbert_detector/raid_ablation_a_seed42_best.pt",
        "artifacts/distilbert_detector/raid_ablation_b_seed42_best.pt",
        "artifacts/distilbert_detector/raid_ablation_c_seed42_best.pt",
        # Legacy non-seed variants (secondary fallback)
        "artifacts/distilbert_detector/raid_baseline1_best.pt",
        "artifacts/distilbert_detector/baseline1_best.pt",
        "artifacts/distilbert_detector/raid_ablation_a_best.pt",
        "artifacts/distilbert_detector/ablation_a_best.pt",
        "artifacts/distilbert_detector/raid_ablation_b_best.pt",
        "artifacts/distilbert_detector/ablation_b_best.pt",
        "artifacts/distilbert_detector/raid_ablation_c_best.pt",
        "artifacts/distilbert_detector/ablation_c_best.pt",
    ]
)


@dataclass
class TrainingConfig:
    """Top-level configuration matching the YAML schema."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    ablations: AblationsConfig = field(default_factory=AblationsConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Type validation ───────────────────────────────────────────────────────────

def _validate_field(value: Any, expected_type: Type, field_name: str) -> None:
    """Validate that *value* matches *expected_type* (supports Optional[X] and List[X])."""
    origin = getattr(expected_type, "__origin__", None)
    args = getattr(expected_type, "__args__", ())

    # Optional[X] → None is always valid
    # Check against Union directly: Optional[T].__origin__ is Union in PEP 484
    if origin is Union:
        if value is None:
            return
        # Unwrap Optional → validate the inner type
        inner_types = [t for t in args if t is not type(None)]
        for inner in inner_types:
            try:
                _validate_field(value, inner, field_name)
                return
            except TypeError:
                continue
        raise TypeError(
            f"'{field_name}' expected Optional[{' | '.join(t.__name__ for t in inner_types)}], "
            f"got {type(value).__name__}"
        )

    # List[X]
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(
                f"'{field_name}' expected list, got {type(value).__name__}"
            )
        if args:
            elem_type = args[0]
            for i, elem in enumerate(value):
                _validate_field(elem, elem_type, f"{field_name}[{i}]")
        return

    # bool check must come before int check because bool is a subclass of int
    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"'{field_name}' expected bool, got {type(value).__name__}"
            )
        return

    # Numeric / string / other
    if not isinstance(value, expected_type):
        raise TypeError(
            f"'{field_name}' expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _validate_dataclass(obj: Any, dc_type: Type) -> None:
    """Recursively validate a dataclass instance against its type annotations."""
    hints = get_type_hints(dc_type)
    for f in fields(dc_type):
        val = getattr(obj, f.name)
        expected = hints.get(f.name)
        if expected is None:
            continue

        # If the field itself is a dataclass, recurse
        if hasattr(expected, "__dataclass_fields__"):
            _validate_dataclass(val, expected)
        else:
            _validate_field(val, expected, f.name)


# ── YAML loading ──────────────────────────────────────────────────────────────

def _coerce_value(value: Any, expected_type: Type) -> Any:
    """Coerce *value* to *expected_type* when YAML parsing yields a wrong type.

    Handles edge cases like PyYAML parsing ``2e-5`` as a string instead of float.
    """
    if value is None:
        return value
    origin = getattr(expected_type, "__origin__", None)
    if origin is list:
        if isinstance(value, list):
            args = getattr(expected_type, "__args__", ())
            elem_type = args[0] if args else str
            return [_coerce_value(v, elem_type) for v in value]
        return value
    # Handle Union types (e.g., Optional[str]) — isinstance rejects typing.Union on Python <3.12
    if origin is Union:
        args = getattr(expected_type, "__args__", ())
        non_none = [t for t in args if t is not type(None)]
        if non_none:
            return _coerce_value(value, non_none[0])
        return value
    if isinstance(value, expected_type):
        return value
    # String → numeric / bool
    if isinstance(value, str):
        if expected_type is bool:
            return value.lower() in ("true", "yes", "1", "on")
        if expected_type is int:
            return int(value)
        if expected_type is float:
            return float(value)
    # Numeric cross-coercion (int → float, float → int)
    if expected_type is float and isinstance(value, int):
        return float(value)
    if expected_type is int and isinstance(value, float):
        if value == int(value):
            return int(value)
        raise TypeError(f"Cannot coerce float {value} to int without loss")
    # Int → bool coercion (e.g., --training.pin_memory=0)
    if expected_type is bool and isinstance(value, int):
        return bool(value)
    return value


def _dict_to_dataclass(data: Dict[str, Any], dc_type: Type) -> Any:
    """Recursively convert a nested dict into a dataclass instance."""
    init_kwargs: Dict[str, Any] = {}
    hints = get_type_hints(dc_type)

    for f in fields(dc_type):
        if f.name not in data:
            continue  # use default
        val = data[f.name]
        expected = hints.get(f.name)

        if expected is not None and hasattr(expected, "__dataclass_fields__"):
            # Nested dataclass — recurse
            init_kwargs[f.name] = _dict_to_dataclass(val, expected)
        elif expected is not None:
            inner_origin = getattr(expected, "__origin__", None)
            inner_args = getattr(expected, "__args__", ())
            # If the field is a list of dataclasses (not needed here, but for completeness)
            if inner_origin is list and inner_args and hasattr(inner_args[0], "__dataclass_fields__"):
                init_kwargs[f.name] = [_dict_to_dataclass(item, inner_args[0]) for item in val]
            else:
                # Coerce string values to expected type (handles YAML edge cases like "2e-5")
                init_kwargs[f.name] = _coerce_value(val, expected)
        else:
            init_kwargs[f.name] = val

    # Fill missing keys from defaults
    return dc_type(**init_kwargs)


# ── CLI overrides ─────────────────────────────────────────────────────────────

def _parse_cli_overrides() -> Dict[str, Any]:
    """Parse ``--section.key=value`` CLI arguments into a nested dict.

    Scans ``sys.argv`` directly (without argparse) so that ``--key=value``
    tokens are not rejected as unknown optional arguments.
    """

    def _parse_cli_value(value: str) -> Any:
        """Parse a CLI string into int / float / bool / str (in that order)."""
        if value.lower() == "null":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        lower = value.lower()
        if lower in ("true", "yes", "1", "on"):
            return True
        if lower in ("false", "no", "off"):
            return False
        return value

    overrides: Dict[str, Any] = {}
    for raw in sys.argv[1:]:
        if not raw.startswith("--"):
            continue
        stripped = raw[2:]
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)

        parsed_value = _parse_cli_value(value)

        parts = key.split(".")
        current = overrides
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = parsed_value

    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(path: Optional[str] = None) -> TrainingConfig:
    """Load and validate the training configuration.

    Parameters
    ----------
    path : str, optional
        Path to a YAML config file.  If ``None`` (default), loads
        ``<project_root>/src/config/default.yaml``.

    Returns
    -------
    TrainingConfig
        Fully validated configuration dataclass with CLI overrides applied.
    """
    if path is None:
        path = str(_PROJECT_ROOT / "src" / "config" / "default.yaml")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    # Apply CLI overrides on top of the YAML dict
    cli_overrides = _parse_cli_overrides()
    if cli_overrides:
        raw = _deep_merge(raw, cli_overrides)

    # Convert to dataclass
    cfg = _dict_to_dataclass(raw, TrainingConfig)

    # Validate types recursively
    _validate_dataclass(cfg, TrainingConfig)

    # Validate ablation-specific domain constraints
    for fld in fields(cfg.ablations):
        ablation: AblationConfig = getattr(cfg.ablations, fld.name)
        if ablation.head_type not in {"single", "deep"}:
            raise ValueError(
                f"ablation '{fld.name}': head_type must be "
                f"'single' or 'deep', got '{ablation.head_type}'"
            )
        if not 0 <= ablation.freeze_layers <= 6:
            raise ValueError(
                f"ablation '{fld.name}': freeze_layers must be "
                f"between 0 and 6, got {ablation.freeze_layers}"
            )

    return cfg


# ── CLI entry point (standalone usage) ────────────────────────────────────────

if __name__ == "__main__":
    import dataclasses
    cfg = load_config()
    print(yaml.dump(dataclasses.asdict(cfg), default_flow_style=False))