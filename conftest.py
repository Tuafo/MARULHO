"""Root conftest.py — adds src/ to sys.path so 'hecsn' is importable in tests."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Force CPU for tests to avoid device-mismatch issues (GPU only for explicit
# scale-testing scripts, not unit tests).
os.environ.setdefault("HECSN_DEVICE", "cpu")
