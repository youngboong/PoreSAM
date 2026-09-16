# PoreSAM

SEM pore segmentation, interactive editing and 2D measurement with SAM 2.1 Small. Run the local server on Windows and use the browser interface.

## Features

- Four-step workspace: Load Image → Preprocess → Analyze & Edit → Generate Report
- Live normalization, background removal and smoothing preview
- CPU and CUDA support
- Automatic segmentation and box/ellipse/point-guided SAM refinement
- Polygon additions, drag cuts, overlap resolution and undo
- Original/segmentation comparison with adjustable mask color and opacity
- Ctrl multi-selection and batch deletion
- Resizable Pore Details popup inside the browser, with measurements, histogram and scatter plot
- Save Images exports comparison, segmentation and per-pore color PNGs; Generate Report exports PDF and standalone HTML separately
- Automate assistance with up to 16 search boxes

## Run

Use the Python 3.11 `pore` environment described in [SETUP.md](SETUP.md). Place the SAM 2.1 Small weights at `checkpoints/sam2.1_hiera_small.pt`.

```powershell
conda activate pore
python scripts/pore_editor.py --device cpu
```

Open http://127.0.0.1:8765. Omit `--device cpu` to use CUDA when available. See [EDITOR.md](EDITOR.md) for the workflow and shortcuts.

## Repository

- `ui/`: browser workspace
- `scripts/`: processing, server, measurements and verification
- `requirements*.txt`: runtime and development dependencies
- `data/`, `checkpoints/`, `outputs/`: local inputs, weights and results; excluded from Git

Measurements describe 2D pore masks. Roundness is not a 3D sphericity measurement.
