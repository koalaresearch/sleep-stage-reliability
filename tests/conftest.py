"""Make the analysis scripts in ``src/`` importable by the tests without packaging them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
