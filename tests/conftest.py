"""Test config: ensure the project root is on sys.path so test files can
import ``defect_detector`` regardless of where pytest is invoked.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force a non-interactive matplotlib backend so the suite is headless-safe.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
