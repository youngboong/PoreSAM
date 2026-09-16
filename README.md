# PoreSAM

SEM pore segmentation, editing and 2D measurement with SAM 2.1 Small. This is the earlier English browser edition on the `english` branch. The current English Windows desktop app is on the default [app branch](https://github.com/youngboong/PoreSAM/tree/app); the earlier Korean browser edition is on [korean](https://github.com/youngboong/PoreSAM/tree/korean).

This branch runs a local server with a browser UI. It does not contain the desktop launcher or packaging scripts. Use `app` for the Windows desktop host, CPU-only build instructions, independent Pore Details windows, and recent Automate updates.

| Branch | Language | Runtime |
|---|---|---|
| [app](https://github.com/youngboong/PoreSAM/tree/app) — default | English | Current Windows desktop app |
| english — this branch | English | Earlier local browser UI |
| [korean](https://github.com/youngboong/PoreSAM/tree/korean) | Korean | Earlier local browser UI |

These are separate versions, not a language toggle. Features are not automatically synchronized between branches.

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
- Automate assistance with up to 16 search boxes in this branch; later 32-box and consolidation changes are on `app`

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
