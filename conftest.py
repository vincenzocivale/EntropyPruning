import os
import sys

# Ensure the repo root is importable as `src` regardless of pytest's invocation dir.
sys.path.insert(0, os.path.dirname(__file__))
