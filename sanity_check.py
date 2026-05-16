import torch

import accelerate
import datasets
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tqdm
import transformers


def main() -> None:
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU available'}")
    print(f"accelerate: {accelerate.__version__}")
    print(f"datasets: {datasets.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"seaborn: {sns.__version__}")
    print(f"sklearn: {sklearn.__version__}")
    print(f"tqdm: {tqdm.__version__}")


if __name__ == "__main__":
    main()