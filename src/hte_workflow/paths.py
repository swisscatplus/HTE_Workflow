from __future__ import annotations
from pathlib import Path
import os

# .../src/hte_workflow
PKG_DIR = Path(__file__).resolve().parent
# repo root = parent of src
PROJECT_ROOT = PKG_DIR.parents[1]

# Defaults: repo-root/data and repo-root/outputs
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR  = PROJECT_ROOT / "outputs"

# Allow overrides via env vars
DATA_DIR = Path(os.getenv("HTE_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
OUT_DIR  = Path(os.getenv("HTE_OUTPUT_DIR", DEFAULT_OUT_DIR)).resolve()

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
