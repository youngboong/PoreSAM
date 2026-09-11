"""Upgrade saved user revisions without changing their masks or measured values."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pore_editor import ROOT, save_images
from analyze_candidates import export_folder


def upgrade(folder):
    masks_path=folder/"entrance_candidates.npz"
    digest=hashlib.sha256(masks_path.read_bytes()).hexdigest()
    report=json.loads((folder/"report.json").read_text(encoding="utf-8"))
    summary_path=folder/"measurements/summary.json"
    before=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    gray=np.asarray(Image.open(ROOT/report["image"]).convert("L"))[:report["analysis_bottom_exclusive"]]
    with np.load(masks_path,allow_pickle=False) as data:
        masks={int(k.rsplit("_",1)[1]):data[k] for k in data.files}
    save_images(folder,gray,masks)
    report["overlay_relative_path"]="images/entrance_candidates_overlay.png"
    (folder/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    export_folder(folder)
    after=json.loads(summary_path.read_text(encoding="utf-8"))
    assert digest==hashlib.sha256(masks_path.read_bytes()).hexdigest(),"Masks changed during layout migration"
    if before is not None:
        assert before==after,"Measured values changed during layout migration"
    assert '../images/entrance_candidates_overlay.png' in (folder/"measurements/index.html").read_text(encoding="utf-8")
    print(f"Updated layout, masks and statistics preserved: {folder}",flush=True)


if __name__=="__main__":
    for revision in sorted((ROOT/"outputs/manual_edits").glob("*/revision_*")):
        if (revision/"edit.json").is_file():
            upgrade(revision)
