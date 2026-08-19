#!/usr/bin/env python3
"""Validate tracked provenance metadata against an available local snapshot."""

import importlib
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

main = importlib.import_module("ai_ml_research_lab.provenance").main

if __name__ == "__main__":
    raise SystemExit(main())
