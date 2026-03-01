# grid_soccer/utils.py

from __future__ import annotations
import pickle
from typing import Any, Dict

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_get(d: Dict, key, default=0.0):
    return d.get(key, default)
