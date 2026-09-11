"""Shared first-pass layout; readers also support legacy and manual revisions."""
from pathlib import Path


ARTIFACT_PATHS = {
    'entrance_candidates.npz': 'masks/entrance_candidates.npz',
    'raw_masks.npz': 'sam_raw/raw_masks.npz',
    'raw_metadata.json': 'sam_raw/raw_metadata.json',
    **{name: f'images/{name}' for name in (
        'analysis_region_and_scale.png', 'comparison.png',
        'entrance_candidates_overlay.png', 'raw_sam_overlay.png', 'review_panels.png')},
}


def read_artifact(folder, name):
    folder = Path(folder)
    organized = folder / ARTIFACT_PATHS.get(name, name)
    return organized if organized.is_file() else folder / name


def write_artifact(folder, name):
    path = Path(folder) / ARTIFACT_PATHS.get(name, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
