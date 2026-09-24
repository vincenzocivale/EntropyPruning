#!/usr/bin/env python3
"""Evaluate immune/stromal spatial biology using frozen model outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.spatial.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
