"""Exploratory automatic SAM masks; entrance selection is a heuristic, not ground truth."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import tifffile
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam_runtime import configure_device, inference_context
from result_paths import read_artifact, write_artifact


class ProgressGenerator(SAM2AutomaticMaskGenerator):
    batches = 0

    def _process_batch(self, *args, **kwargs):
        result = super()._process_batch(*args, **kwargs)
        self.batches += 1
        if getattr(self, "progress_callback", None):
            self.progress_callback(self.batches)
        if self.batches % 24 == 0:
            print(f"Processed {self.batches} point batches", flush=True)
        return result


def detect_footer(gray):
    # This first-pass detector assumes a flat, full-width SEM information footer.
    row_std = gray.std(axis=1)
    for y in range(int(gray.shape[0] * 0.6), gray.shape[0] - 5):
        if np.all(row_std[y:y + 3] < 5) and np.mean(row_std[max(0, y - 10):y]) > 10:
            return y
    raise ValueError("Footer not detected reliably; inspect the source image.")


def detect_scale_bar(gray, footer_y, label_um):
    if not np.isfinite(label_um) or label_um <= 0:
        raise ValueError("Scale label must be a positive length in micrometers.")
    white = (gray[footer_y:] >= 230).astype(np.uint8)
    lines = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((1, 60), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(lines)
    candidates = [s for s in stats[1:] if s[2] >= 60 and s[3] <= 8]
    if not candidates:
        raise ValueError("Scale bar not detected.")
    x, y, w, h, _ = max(candidates, key=lambda s: s[2])
    # Pixel-center distance between the two end ticks.
    return dict(x=int(x), y=int(y + footer_y), width_pixels=int(w),
                height_pixels=int(h), length_pixels=int(w - 1),
                label_um=float(label_um), label_source="human read from this image; not OCR",
                um_per_pixel=float(label_um / (w - 1)))


def candidates_from_masks(raw, gray, min_contrast=8, min_area=100):
    if not np.isfinite(min_contrast) or not 0 <= min_contrast <= 255:
        raise ValueError("Brightness difference must be between 0 and 255.")
    if not np.isfinite(min_area) or min_area < 1:
        raise ValueError("Minimum area must be positive.")
    candidates = []
    for raw_id, item in enumerate(raw):
        mask = item["segmentation"]
        components, n = ndi.label(mask)
        if not n:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest = components == sizes.argmax()
        if largest.sum() < 0.9 * mask.sum():
            continue
        filled = ndi.binary_fill_holes(largest)
        area = int(filled.sum())
        if area < min_area or area > gray.size * 0.20:
            continue
        ring = ndi.binary_dilation(filled, iterations=4) & ~filled
        contrast = float(gray[ring].mean() - gray[filled].mean()) if ring.any() else 0
        # Dark interior surrounded by a brighter rim; this does not establish depth.
        if contrast < min_contrast:
            continue
        candidates.append(dict(raw_id=raw_id, mask=filled, area=area,
                               contrast=contrast, predicted_iou=float(item["predicted_iou"])))
    # Prefer a containing entrance hypothesis to nested internal structures.
    selected = []
    for candidate in sorted(candidates, key=lambda a: a["area"], reverse=True):
        if any(np.count_nonzero(candidate["mask"] & other["mask"]) / candidate["area"] > 0.8
               for other in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda a: tuple(np.argwhere(a["mask"]).mean(axis=0)))
    from pore_overlap import exclusive_existing
    exclusive,cleanup=exclusive_existing({i:item['mask'] for i,item in enumerate(selected)})
    selected=[dict(selected[i],mask=mask,area=int(mask.sum()),overlap_boundary_adjusted=i in cleanup['changed_ids']) for i,mask in exclusive.items()]
    return selected


def overlay(gray, masks, numbered=False):
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rng = np.random.default_rng(12)
    for i, mask in enumerate(masks):
        color = tuple(int(v) for v in rng.integers(70, 256, 3))
        canvas[mask] = (canvas[mask] * 0.76 + np.array(color) * 0.24).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
        if numbered:
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            cv2.putText(canvas, str(i + 1), (int(x)-8, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(canvas, str(i + 1), (int(x)-8, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return Image.fromarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/PI35_5kx-4_bse.tif"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_small.pt"))
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument("--output", type=Path, help="Defaults to outputs/<image stem>_first_pass")
    parser.add_argument("--scale-um", type=float, required=True, help="Visually checked scale-bar label in micrometers (not OCR)")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--crop-layers", type=int, default=0)
    parser.add_argument("--points-per-side", type=int, default=48)
    parser.add_argument("--min-contrast", type=float, default=8)
    parser.add_argument("--min-area", type=int, default=100)
    args = parser.parse_args()
    device = configure_device(args.device)
    if args.output is None:
        args.output = Path("outputs") / f"{args.image.stem}_first_pass"
    args.output.mkdir(parents=True, exist_ok=True)
    gray = tifffile.imread(args.image)
    footer_y = detect_footer(gray)
    scale = detect_scale_bar(gray, footer_y, args.scale_um)
    roi = gray[:footer_y]
    print(f"Analysis region: {roi.shape}; scale: {scale}", flush=True)
    inspection = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(inspection)
    draw.line((0, footer_y, gray.shape[1]-1, footer_y), fill="cyan", width=2)
    draw.rectangle((scale["x"]-3, scale["y"]-25, scale["x"]+scale["width_pixels"]+3, scale["y"]+5), outline="lime", width=2)
    inspection.save(write_artifact(args.output, "analysis_region_and_scale.png"))
    settings = dict(points_per_side=args.points_per_side, points_per_batch=8, pred_iou_thresh=0.8,
                    stability_score_thresh=0.92, crop_n_layers=args.crop_layers, min_mask_region_area=0)
    if args.reuse:
        previous_report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        if Path(previous_report["image"]).resolve() != args.image.resolve():
            raise ValueError("Saved masks belong to a different image.")
        settings = previous_report["settings"]
    started = time.monotonic()
    if args.reuse:
        metadata = json.loads(read_artifact(args.output, "raw_metadata.json").read_text())
        with np.load(read_artifact(args.output, "raw_masks.npz")) as data:
            raw = [dict(item, segmentation=data[f"mask_{i}"]) for i, item in enumerate(metadata)]
    else:
        print(f"Loading SAM 2.1 Small on {device.upper()}", flush=True)
        with inference_context(device):
            model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(args.checkpoint), device=device, apply_postprocessing=False)
            generator = ProgressGenerator(model, **settings)
            raw = generator.generate(cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB))
        metadata = [{k: v for k, v in item.items() if k != "segmentation"} for item in raw]
        write_artifact(args.output, "raw_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(write_artifact(args.output, "raw_masks.npz"), **{f"mask_{i}": a["segmentation"] for i, a in enumerate(raw)})
    print(f"SAM returned {len(raw)} masks", flush=True)
    selected = candidates_from_masks(raw, roi, args.min_contrast, args.min_area)
    raw_overlay = overlay(roi, [a["segmentation"] for a in sorted(raw, key=lambda a: a["area"], reverse=True)])
    selected_overlay = overlay(roi, [a["mask"] for a in selected], numbered=True)
    raw_overlay.save(write_artifact(args.output, "raw_sam_overlay.png"))
    selected_overlay.save(write_artifact(args.output, "entrance_candidates_overlay.png"))
    np.savez_compressed(write_artifact(args.output, "entrance_candidates.npz"), **{f"candidate_{i+1}": a["mask"] for i, a in enumerate(selected)})
    width, height = roi.shape[1], roi.shape[0]
    comparison = Image.new("RGB", (width * 2, height + 40), "#222222")
    comparison.paste(Image.fromarray(roi).convert("RGB"), (0, 40))
    comparison.paste(selected_overlay, (width, 40))
    draw = ImageDraw.Draw(comparison)
    draw.text((15, 12), "Original", fill="white")
    draw.text((width + 15, 12), "SAM + heuristic entrance candidates (UNVALIDATED)", fill="white")
    comparison.save(write_artifact(args.output, "comparison.png"))
    report = dict(image=str(args.image), analysis_bottom_exclusive=footer_y, scale=scale,
                  model="SAM 2.1 Small", settings=settings, raw_mask_count=len(raw),
                  inference_device=previous_report.get('inference_device','unknown') if args.reuse else device,
                  entrance_candidate_count=len(selected), elapsed_seconds=time.monotonic()-started,
                  status="exploratory, not validated pore counts or areas",
                  overlay_relative_path="images/entrance_candidates_overlay.png",
                  selection_settings=dict(min_contrast=args.min_contrast,min_area_pixels=args.min_area),
                  selection=f"largest component >=90%; area {args.min_area}px..20%; ring contrast >={args.min_contrast}; fill enclosed holes; suppress >=80% containment",
                  candidates=[{k:v for k,v in item.items() if k != "mask"} for item in selected])
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} entrance candidates to {args.output}", flush=True)


if __name__ == "__main__":
    main()
