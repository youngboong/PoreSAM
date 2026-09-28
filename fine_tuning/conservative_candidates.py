"""Compatibility entry point for the shared application selection logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pore_conservative import *  # noqa: F401,F403
