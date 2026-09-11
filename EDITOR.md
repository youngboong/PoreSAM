# PoreSAM Editor

English UI on the `english` branch. The Korean edition remains on `korean`.

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
3. **Analyze & Edit**: compare the original on the left with segmentation on the right. Edit pores and inspect measurements in **Pore Details**.
4. **Report**: choose a destination with **Browse…**, then **Generate & Export**. Each export creates a separate folder containing the original file, segmentation and comparison images, measurements, PDF and HTML reports.

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

Cut removes mask pixels along the path. Disconnected pieces become separate pores; the largest retains the original ID. Multi-delete is undone as one operation.

## Pore Details

The modeless window stays open while editing. Drag its title bar to move it, or any corner or edge to resize. Escape closes it. New manual pores appear in the list automatically.

- **Details**: ID, length, width, aspect ratio, equivalent diameter and roundness; min, max, mean and sample standard deviation below the table.
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
- Export: a unique folder under the chosen destination with `original/`, `images/`, and `measurements/`

Older reports are regenerated in English on explicit report generation. Existing masks and measurement formulas are unchanged by the UI language. English and Korean branches share local data paths when run from the same checkout.
