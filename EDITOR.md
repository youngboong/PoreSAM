# PoreSAM Editor

English Windows desktop UI on the default `app` branch. Earlier browser editions remain on `english` and `korean`; their features and export behavior differ from this branch.

## Run

From the project directory:

```powershell
conda activate pore-app-cpu
python scripts/pore_app.py
```

A native Windows window opens and manages its own local server. CPU is the desktop default. See [DESKTOP.md](DESKTOP.md) for environment installation and packaging. To run the browser UI instead, use `python scripts/pore_editor.py --device cpu` and open http://127.0.0.1:8765; use `--port 8766` if occupied and Ctrl+C to stop the server.

## Workflow

1. **Load Image** always opens **New Image**. Import TIF, PNG or JPG (up to 32 MB). Importing an already analyzed file also opens preprocessing; use **Open Analysis** to resume its saved masks.
2. **Preprocess**: confirm the analysis region and calibration. Measure both ends of the scale bar, enter its physical length, then check **Confirm calibration**. Adjust processing while comparing the original and live preview. **Run Analysis** starts automatic segmentation.
3. **Analyze & Edit**: compare the original on the left with segmentation on the right. Edit pores and inspect measurements in **Pore Details**. **Save Images** opens a folder picker and saves `comparison.png` (original left, segmentation right), `segmentation.png`, and `pores_colored.png`. The original footer is retained; color, opacity, and ID visibility follow the display controls. Only saved pore masks are exported, not pending previews.
4. **Generate Report**: choose a destination with **Browse…**, then **Generate Report**. Each export creates a separate folder containing only `report.pdf` and standalone `report.html`. Image exports use **Save Images** in step 3; the uploaded original and internal measurement files remain in the application data directory.

## Processing

| Setting | Function |
|---|---|
| Brightness normalization | Normalize the 2nd–98th intensity percentiles |
| Background removal | Suppress thin bright structures; strength 0 turns it off |
| Smoothing filter | None, Gaussian, Mean, Median, Bilateral or Kuwahara |
| Strength | Shown when a smoothing filter is enabled |
| Detection Settings | Minimum contrast, minimum area and sampling density |

Processing order is normalization → background removal → smoothing. These operations use image intensity, not physical depth. Automatic candidates may occupy up to 20% of the analysis region; contrast, minimum area and containment filters also apply. Manual additions do not use that area ceiling.

For a new pore, draw a **Box** and click **Preview**. Each box is one target: choose one SAM mask and confirm with **Add Pore**. A large mask is not automatically replaced by smaller child candidates, and no local multi-pore search runs for a manual box. Include/Exclude and Update Preview refine that target. Multiple boxes may still be queued and reviewed individually, up to 32 regions.

## Editing

| Tool or shortcut | Action |
|---|---|
| Select | Click a pore to select it |
| Ctrl+click | Add or remove a pore from the selection |
| Select Boundary Pores | Select all pores touching the image boundary |
| Delete | Delete selected pores in one edit |
| Box / Ellipse | Drag a region, then preview SAM segmentation |
| Include + / Exclude − | Refine SAM with points |
| Polygon | Click vertices, preview, then add the drawn mask |
| Cut | Drag through a pore and apply a cut, 1–30 px wide |
| Undo Input / Clear Input | Undo the last drawing input or clear the current drawing |
| Ctrl+Z / Undo | Undo drawing inputs first, then the last saved edit |

Pore selection is disabled while drawing. Preview masks are applied only when **Add Pore** or **Replace Pore(s)** is clicked. Contained existing pores are replaced when at least 95% of their area lies within a larger new mask. Partial overlaps must be trimmed or resolved before applying.

Cut accepts up to 32 successive drag paths, including intersecting or duplicate paths. The width control applies to all pending cuts. **Apply Cuts (N)** applies their union as one saved edit; disconnected pieces become separate pores and the largest retains the original ID. Ctrl+Z removes the last pending stroke; after applying, it restores the entire cut operation. Clear Input removes all pending cuts. Multi-delete is also undone as one operation.

## Multiple Regions

Draw successive boxes or ellipses to queue them. Switch tools to mix shapes. Finish each polygon with **Enter** or **Finish Region** before drawing the next one. Up to 32 regions can be queued.

**Preview All** processes only new or changed regions without adding pores; unchanged previews are reused. Select a region, place Include/Exclude points, then click **Update Preview** to recompute only that region. Pending points remain attached to their region when selecting another queued region. Select a ready region in the list, inspect its options, then click **Add Pore** (or **Replace Pore(s)**) for that region. The next ready region opens automatically, but still needs its own confirmation. Overlap and containment are recalculated after every addition. **Trim Overlap** remains available.

Queued regions are temporary and clear when switching to selection mode, loading another result, or undoing a saved edit. Each added region is a separate saved edit. Use × to remove a queued region; failed regions can be removed and redrawn.

## Pore Details

In the Windows app, Pore Details is an independent modeless native window: move it outside the editor or onto another monitor and resize it from any border. It stays linked to its owning editor. The browser entry point uses an in-page popup instead. New manual pores appear in the list automatically.

- **Details**: ID, length, width, aspect ratio, equivalent diameter, roundness and area fraction; min, max, mean and sample standard deviation below the table. Click column headers to sort ascending or descending.
- **Pores** dropdown: filter the table and plots by ID.
- **Histogram**: select a measurement and bin count.
- **Scatter Plot**: select X and Y measurements. All points use one color. Clicking a point highlights its pore on the image.
- **Export CSV / Export Plot**: save the filtered table or plot.

Measurement formulas are available in the collapsed **Measurement Definitions** section. Shape metrics describe the 2D mask. Boundary pores contribute to total count and area fraction, but are excluded from the main size distributions.

## Saving

Editing saves mask revisions and updates on-screen measurements without generating reports. Report generation happens only in step 4.

- Default desktop data root: `%LOCALAPPDATA%/PoreSAM/English/outputs`; override with `--data-dir`.
- Original uploads: `<data root>/projects/<image ID>/input/`
- Automatic masks: `<data root>/projects/<image ID>/runs/run_XXXX/`
- Manual revisions: `<data root>/manual_edits/<analysis ID>/revision_XXXX/`
- Image export: a unique folder containing three PNG images.
- Report export: a separate unique folder containing PDF and standalone HTML only.

Older reports are regenerated in English on explicit report generation. The desktop default data root differs from the browser edition's project-local `outputs`. Additional desktop windows receive separate workspace directories. See [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for complete paths and Python examples.

### Automatic box assistance

Run Analysis shows elapsed time and a live estimate of the remaining SAM detection time after three point batches. Model loading is excluded from the batch-speed estimate. Filtering and saving are reported as separate final stages; the estimate is not a guarantee of total completion time.

**Automate** proposes up to 32 boxes around uncovered dark regions and evaluates SAM options in descending score order. This is a search-box limit, not a limit on total pore count; final consolidation separately revisits up to 16 touching groups. The minimum SAM quality score is 0.7 for both additions and merges; size, contrast, containment, and boundary checks still apply. A candidate that substantially covers an existing pore, or strongly overlaps the same dark interior, can merge with it; the smallest existing ID is retained. Otherwise overlapping pixels are trimmed while preserving existing pores. The largest remaining component satisfying size and contrast checks is added automatically. Tiny remnants (under 10% of the original candidate or the minimum area) are discarded. If no option qualifies, that box is skipped. Automate does not create review boxes; manual Add Pore previews still require individual confirmation.

Added and merged masks use the supplementary color and updated measurements. All changes are committed together when Automate finishes; Ctrl+Z restores the previous result, and Stop discards the current operation. Pending manual drafts are retained. These conservative image-based checks are heuristics, so merged pores still need visual review.

Before committing, Automate also revisits touching automatic additions, whether or not they needed trimming, including pieces already saved in a previous run. A joint SAM box must support the merge; shared boundaries are compared with the brighter outer rim so internal texture alone does not prevent merging. This final pass skips manually cut pieces and cannot add more fragments when a merge is rejected.

The shared-boundary check requires at least 50% of seam pixels to be darker than the outer-ring mean minus the configured minimum contrast. This is a global criterion, not an exception for particular pore IDs; the mean-contrast, coverage, connectivity, and size checks still apply.

If a whole neighborhood cannot merge, Automate evaluates touching pairs supported by the same SAM candidate, while preserving other existing masks. All reconciliation checks still apply; qualifying pairs are ranked by how completely SAM covers their existing masks. This prevents one unrelated neighbor from vetoing an otherwise valid pair. Manually cut masks are protected during ordinary candidate processing as well as final consolidation.

Overlap and boundary processing uses bounding boxes to skip distant masks and compute on a local region. This region contains the complete candidate and potentially overlapping masks, with four pixels of contrast-ring context; area limits still use the full image size. This optimization does not change SAM sampling density, merge thresholds, or candidate limits.

Supplementary SAM and polygon additions use the **Added** display color. Selection highlighting chooses a different color from both existing and added pores. Comparison and single-color segmentation exports use the chosen common pore color for all pores; the separate per-pore multicolor export remains available.
