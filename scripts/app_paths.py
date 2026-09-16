"""Read-only application assets and writable user data are separate."""
import os
import sys
from pathlib import Path

ASSET_ROOT=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))

def output_root():
    override=os.environ.get('PORESAM_DATA_DIR')
    return Path(override).expanduser().resolve() if override else ASSET_ROOT/'outputs'

def checkpoint_path():
    override=os.environ.get('PORESAM_CHECKPOINT')
    return Path(override).expanduser().resolve() if override else ASSET_ROOT/'checkpoints/sam2.1_hiera_small.pt'
