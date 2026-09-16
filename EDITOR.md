# PoreSAM Editor


## Run

From the project directory:

```powershell
conda activate pore
python scripts/pore_editor.py --device cpu
```

Open http://127.0.0.1:8765. Stop the server with Ctrl+C. Use `--port 8766` if the default port is occupied. Omitting `--device cpu` selects CUDA when available and CPU otherwise. See [SETUP.md](SETUP.md) for environment installation.

## Workflow

1. **Load Image** always opens **New Image**. Import TIF, PNG or JPG (up to 32 MB). Importing an already analyzed file also opens preprocessing; use **Open Analysis** to resume its saved masks.
2. **Preprocess**: confirm the analysis region and calibration. Measure both ends of the scale bar, enter its physical length, then check **Confirm calibration**. Adjust processing while comparing the original and live preview. **Run Analysis** starts automatic segmentation.
3. **Analyze & Edit**: compare the original on the left with segmentation on the right. Edit pores and inspect measurements in **Pore Details**. **Save Images** opens a folder picker and saves `comparison.png` (original left, segmentation right), `segmentation.png`, and `pores_colored.png`. The original footer is retained; color, opacity, and ID visibility follow the display controls. Only saved pore masks are exported, not pending previews.
4. **Generate Report**: choose a destination with **Browse…**, then **Generate Report**. Each export creates a separate folder containing only `report.pdf` and standalone `report.html`. Use **Save Images** in step 3 for image exports. The uploaded original and internal measurement files remain under the project's `outputs` directory.

## Processing

| Setting | Function |
|---|---|
| Brightness normalization | Normalize the 2nd–98th intensity percentiles |
| Background removal | Suppress thin bright structures; strength 0 turns it off |
| Smoothing filter | None, Gaussian, Mean, Median, Bilateral or Kuwahara |
| Strength | Shown when a smoothing filter is enabled |
| Detection Settings | Minimum contrast, minimum area and sampling density |

Processing order is normalization → background removal → smoothing. These operations use image intensity, not physical depth. Automatic candidates may occupy up to 20% of the analysis region; contrast, minimum area and containment filters also apply. Manual additions do not use that area ceiling.

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

The modeless popup stays open inside the browser while editing; it is not a separate Windows window. Drag its title bar to move it within the page, or any corner or edge to resize. Escape closes it. New manual pores appear in the list automatically.

- **Details**: ID, length, width, aspect ratio, equivalent diameter, roundness and area fraction; min, max, mean and sample standard deviation below the table.
- **Pores** dropdown: filter the table and plots by ID.
- **Histogram**: select a measurement and bin count.
- **Scatter Plot**: select X and Y measurements. All points use one color. Clicking a point highlights its pore on the image.
- **Export CSV / Export Plot**: save the filtered table or plot.

Measurement formulas are available in the collapsed **Measurement Definitions** section. Shape metrics describe the 2D mask. Boundary pores contribute to total count and area fraction, but are excluded from the main size distributions.

## Saving

Editing saves mask revisions and updates on-screen measurements without generating reports. Report generation happens only in step 4.

- Original uploads: `outputs/projects/<image ID>/input/`
- Automatic masks: `outputs/projects/<image ID>/runs/run_XXXX/`
- Manual revisions: `outputs/manual_edits/<analysis ID>/revision_XXXX/`
- Image export: a unique folder containing three PNG images.
- Report export: a separate unique folder containing PDF and standalone HTML only.


### Automatic box assistance

**Automate** proposes up to 16 boxes around uncovered dark regions and automatically adds candidates passing quality, area, contrast and overlap checks. Existing pores are preserved; small overlaps are trimmed. Added masks are marked unreviewed, appear in measurements immediately, and can be undone one addition at a time. Pending manual drafts are retained.

Supplementary SAM and polygon additions use the **Added** display color. Selection highlighting chooses a different color from both existing and added pores. Comparison and single-color segmentation exports use the chosen common pore color for all pores; the separate per-pore multicolor export remains available.
