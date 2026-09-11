"""Audit selection and compare baseline with additional automatic crops."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import tifffile

from segment_first_pass import candidates_from_masks, overlay
from result_paths import read_artifact


def load_raw(folder):
    metadata = json.loads(read_artifact(folder, "raw_metadata.json").read_text())
    with np.load(read_artifact(folder, "raw_masks.npz")) as data:
        return [dict(m, segmentation=data[f"mask_{i}"]) for i, m in enumerate(metadata)]


def audit(raw, gray):
    counts = Counter()
    rows = []
    for i, item in enumerate(raw):
        mask = item["segmentation"]
        labels, n = ndi.label(mask)
        if not n:
            reason = "empty"
            contrast = None
            area = 0
        else:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest = labels == sizes.argmax()
            filled = ndi.binary_fill_holes(largest)
            area = int(filled.sum())
            ring = ndi.binary_dilation(filled, iterations=4) & ~filled
            contrast = float(gray[ring].mean() - gray[filled].mean()) if ring.any() else 0
            if largest.sum() < 0.9 * mask.sum():
                reason = "fragmented"
            elif area < 100:
                reason = "too_small"
            elif area > gray.size * 0.12:
                reason = "too_large"
            elif contrast < 8:
                reason = "low_contrast"
            else:
                reason = "eligible_before_containment"
        counts[reason] += 1
        rows.append(dict(raw_id=i, reason=reason, filled_area=area, ring_contrast=contrast))
    return dict(counts=counts, masks=rows)


def panels(gray, images, path):
    # Fixed review windows only; these are never supplied as segmentation prompts.
    regions = [("Lower left", (100, 410, 450, 690)),
               ("Lower right", (674, 465, 1024, 745))]
    w, h = 350, 280
    canvas = Image.new("RGB", (len(images) * w, len(regions) * (h + 45)), "#222222")
    draw = ImageDraw.Draw(canvas)
    for r, (name, box) in enumerate(regions):
        y = r * (h + 45)
        for c, (label, image) in enumerate(images):
            draw.text((c*w + 5, y + 7), f"{name} | {label}", fill="white")
            canvas.paste(image.crop(box), (c*w, y + 35))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--render-changes-only", action="store_true")
    args = parser.parse_args()
    baseline = Path("outputs/PI35_2kx-4_BSE8_first_pass")
    expanded = Path("outputs/PI35_2kx-4_BSE8_crop1")
    out = Path("outputs/PI35_2kx-4_BSE8_recovery")
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((baseline / "report.json").read_text())
    gray = tifffile.imread(report["image"])[:report["analysis_bottom_exclusive"]]
    original = Image.fromarray(gray).convert("RGB")
    if args.render_changes_only:
        result = json.loads((out / "report.json").read_text())
        canvas = np.asarray(original).copy()
        with np.load(out / "entrance_candidates.npz") as data:
            for item in result["candidates"]:
                mask = data[f"candidate_{item['candidate_id']}"]
                changed = item["best_iou"] < 0.5
                color = (40, 255, 90) if changed else (160, 160, 160)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 2 if changed else 1)
                if changed:
                    y,x = np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).argmax(), mask.shape)
                    cv2.putText(canvas, str(item["candidate_id"]), (int(x)-6,int(y)+4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 3)
                    cv2.putText(canvas, str(item["candidate_id"]), (int(x)-6,int(y)+4), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        image = Image.new("RGB", (gray.shape[1], gray.shape[0]+35), "#222222")
        image.paste(Image.fromarray(canvas), (0,35))
        ImageDraw.Draw(image).text((10,10), "Green: new/changed hypotheses (best baseline IoU < 0.5); NOT verified recovery", fill="white")
        image.save(out / "new_or_changed_candidates.png")
        return
    baseline_overlay = Image.open(read_artifact(baseline, "entrance_candidates_overlay.png")).convert("RGB")
    raw = load_raw(baseline)
    if args.audit_only:
        result = audit(raw, gray)
        result["selected_count"] = report["entrance_candidate_count"]
        (out / "baseline_selection_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        panels(gray, [("Original", original), ("Raw SAM", Image.open(read_artifact(baseline, "raw_sam_overlay.png"))),
                      ("Baseline selected", baseline_overlay)], out / "baseline_diagnostic.png")
        print(json.dumps(result["counts"]), flush=True)
        return
    extra = load_raw(expanded)
    # Preserve full-image hypotheses alongside the crop generator's output.
    selected = candidates_from_masks(raw + extra, gray)
    print(f"Selected {len(selected)} hypotheses from baseline + crop run", flush=True)
    combined_overlay = overlay(gray, [item["mask"] for item in selected], numbered=True)
    combined_overlay.save(out / "entrance_candidates_overlay.png")
    np.savez_compressed(out / "entrance_candidates.npz", **{f"candidate_{i+1}": item["mask"] for i,item in enumerate(selected)})
    crop_overlay = Image.open(read_artifact(expanded, "entrance_candidates_overlay.png")).convert("RGB")
    panels(gray, [("Original", original), ("Baseline", baseline_overlay),
                  ("Baseline + crops", combined_overlay)], out / "recovery_review.png")
    full = Image.new("RGB", (gray.shape[1]*2, gray.shape[0]+35), "#222222")
    full.paste(baseline_overlay, (0, 35))
    full.paste(combined_overlay, (gray.shape[1], 35))
    draw = ImageDraw.Draw(full)
    draw.text((10, 10), "Baseline", fill="white")
    draw.text((gray.shape[1]+10, 10), "Baseline + overlapping crops (UNVALIDATED)", fill="white")
    full.save(out / "before_after.png")
    with np.load(read_artifact(baseline, "entrance_candidates.npz")) as data:
        old_masks = [data[k] for k in data.files]
    matches = []
    for i,item in enumerate(selected):
        mask = item["mask"]
        ious = [np.count_nonzero(mask & old) / np.count_nonzero(mask | old) for old in old_masks]
        best = int(np.argmax(ious))
        source = "baseline" if item["raw_id"] < len(raw) else "crop_run"
        source_id = item["raw_id"] if source == "baseline" else item["raw_id"]-len(raw)
        matches.append(dict(candidate_id=i+1, source=source, source_raw_id=source_id,
                            closest_baseline_id=best+1, best_iou=float(ious[best]),
                            area=item["area"], contrast=item["contrast"]))
    result = dict(image=report["image"], scale=report["scale"], baseline_folder=str(baseline),
                  crop_folder=str(expanded), baseline_count=len(old_masks), combined_count=len(selected),
                  unmatched_at_iou_05=sum(m["best_iou"]<0.5 for m in matches),
                  status="candidate matching is not a measure of accuracy or recovered true pores",
                  changed="additional overlapping crops; same point density and selection thresholds",
                  candidates=matches)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "candidates"}), flush=True)


if __name__ == "__main__":
    main()
