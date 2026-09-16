# PoreSAM

SEM pore segmentation, editing and 2D measurement with SAM 2.1 Small. The default `app` branch contains the current English Windows desktop application.

The desktop application opens its English workspace in a native Windows window using WebView2. It starts and stops its own local server automatically, with CPU inference by default. See [DESKTOP.md](DESKTOP.md) for running, storage, and packaging.

## Editions

| Branch | Interface | Runtime |
|---|---|---|
| [app](https://github.com/youngboong/PoreSAM/tree/app) — default | English | Current Windows desktop app; CPU by default |
| [english](https://github.com/youngboong/PoreSAM/tree/english) | English | Earlier browser edition, started with `scripts/pore_editor.py` |
| [korean](https://github.com/youngboong/PoreSAM/tree/korean) | Korean | Earlier browser edition, started with `scripts/pore_editor.py` |

These branches are separate versions, not a language toggle. Desktop updates and recent Automate changes are on `app`; the browser branches do not automatically receive them.

## Features

- Four-step workspace: Load Image → Preprocess → Analyze & Edit → Generate Report
- Live normalization, background removal and smoothing preview
- CPU-only Windows packaging; optional CUDA inference from a compatible source environment
- Automatic segmentation and box/ellipse/point-guided SAM refinement
- Polygon additions, drag cuts, overlap resolution and undo
- Original/segmentation comparison with adjustable mask color and opacity
- Ctrl multi-selection and batch deletion
- Merge selected pores, synchronized wheel zoom and cross-image pointer tracking
- Independent, resizable Pore Details window with sortable measurements, histogram and scatter plot
- Automate supplementation with up to 32 search boxes, overlap handling and consolidation
- Stop controls for Run Analysis and Automate; undoable saved edits
- Separate image exports and PDF/HTML report exports

## Run

Create the Python 3.11 `pore-app-cpu` environment using [DESKTOP.md](DESKTOP.md). Place the SAM 2.1 Small weights at `checkpoints/sam2.1_hiera_small.pt`.

```powershell
conda activate pore-app-cpu
python scripts/pore_app.py
```

No browser or fixed port is required. The original browser entry point remains `python scripts/pore_editor.py`. See [EDITOR.md](EDITOR.md) for the workflow and shortcuts.

Desktop data is stored under `%LOCALAPPDATA%/PoreSAM/English/outputs`. `Save Images` exports three PNGs to a chosen folder; `Generate Report` exports PDF and standalone HTML to a separate folder. The repository contains source and build instructions, not the local EXE, model weights, input images, or saved analyses.

For Python entry points, callable functions, copyable PowerShell examples, and output locations, see [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) (Korean).

## Repository

- `ui/`: browser workspace
- `scripts/`: processing, server, measurements and verification
- `requirements*.txt`: runtime and development dependencies
- `data/`, `checkpoints/`, `outputs/`: local inputs, weights and results; excluded from Git

Measurements describe 2D pore masks. Roundness is not a 3D sphericity measurement.
