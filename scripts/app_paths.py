"""Read-only application assets and writable user data are separate."""
import os
import sys
import json
from pathlib import Path

ASSET_ROOT=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))

def output_root():
    override=os.environ.get('PORESAM_DATA_DIR')
    return Path(override).expanduser().resolve() if override else ASSET_ROOT/'outputs'

def checkpoint_path():
    override=os.environ.get('PORESAM_CHECKPOINT')
    if override:
        return Path(override).expanduser().resolve()
    folder=ASSET_ROOT/'checkpoints'
    manifest=folder/'default_model.json'
    if manifest.is_file():
        name=json.loads(manifest.read_text(encoding='utf-8'))['checkpoint']
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.pt'):
            raise ValueError('Invalid default checkpoint filename.')
        return folder/name
    return folder/'sam2.1_hiera_small.pt'
