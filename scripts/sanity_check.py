"""Quick environment sanity checks — library versions and GPU availability.

Usage:
    python scripts/sanity_check.py
"""
from typing import Any

import torch


def main() -> None:
    accelerate = _import_optional("accelerate", "accelerate")
    datasets = _import_optional("datasets", "datasets")
    matplotlib = _import_optional("matplotlib", "matplotlib")
    np = _import_optional("numpy", "numpy")
    pd = _import_optional("pandas", "pandas")
    sns = _import_optional("seaborn", "seaborn")
    sklearn = _import_optional("sklearn", "scikit-learn")
    tqdm = _import_optional("tqdm", "tqdm")
    transformers = _import_optional("transformers", "transformers")

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU available'}")
    print(f"accelerate: {accelerate.__version__ if accelerate else 'N/A'}")
    print(f"datasets: {datasets.__version__ if datasets else 'N/A'}")
    print(f"transformers: {transformers.__version__ if transformers else 'N/A'}")
    print(f"numpy: {np.__version__ if np else 'N/A'}")
    print(f"pandas: {pd.__version__ if pd else 'N/A'}")
    print(f"matplotlib: {matplotlib.__version__ if matplotlib else 'N/A'}")
    print(f"seaborn: {sns.__version__ if sns else 'N/A'}")
    print(f"sklearn: {sklearn.__version__ if sklearn else 'N/A'}")
    print(f"tqdm: {tqdm.__version__ if tqdm else 'N/A'}")


def _import_optional(module_name: str, pip_name: str | None = None) -> Any:  # noqa: ANN401
    try:
        return __import__(module_name)
    except ImportError:
        pkg = pip_name or module_name
        print(f"[WARN] {module_name} not installed. Install with: pip install {pkg}")
    return None


if __name__ == "__main__":
    main()