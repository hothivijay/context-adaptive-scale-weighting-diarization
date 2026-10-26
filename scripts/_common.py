"""Shared bits for the numbered scripts: make `src/` importable without
installing the package, load the YAML configs, and seed every random source."""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_yaml(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Same seed -> same chunk order, dropout masks and initial weights.
    (cuDNN LSTM kernels can still differ in the last digits between GPUs.)"""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
