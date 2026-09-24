#!/usr/bin/env python3
"""Export frozen full/EAF features and actual WSI tile selections."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.spatial.extract import main

if __name__ == "__main__":
    raise SystemExit(main())
