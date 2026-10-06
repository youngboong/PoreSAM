# PoreSAM

![SEM pore segmentation example](docs/example.png)

Illustrative comparison on a training image: original SEM, base SAM, and fine-tuned SAM with two-stage nested selection. The example is for this introduction; the downloadable app contains no saved analyses.

A Windows application for SEM pore segmentation, interactive editing and 2D measurement using SAM 2.1 Small. CPU inference is the default.

## Features

- Load Image, Preprocess, Analyze & Edit, Generate Report
- Live brightness normalization, background removal and smoothing
- Automatic segmentation and prompted SAM refinement
- Box, ellipse and polygon additions; cuts, merging, deletion and undo
- Automate supplementation with up to 32 search boxes
- Synchronized image comparison, zoom and pointer tracking
- Independent Pore Details window with sortable measurements and plots
- Four-file image export and editable multi-image reports with PDF + HWPX export
- PDF preview, folder imports, custom histograms/scatter plots, and publication-style tables

## Run

Download the [Windows app](https://github.com/youngboong/PoreSAM/releases/latest), extract the complete ZIP, and open `PoreSAM/PoreSAM.exe`. Keep `_internal` beside the executable. Python is not needed for this packaged version.

Follow [DESKTOP.md](DESKTOP.md) to install Python 3.11, CPU dependencies and the model checkpoint, then run:

```powershell
conda activate pore-app-cpu
python scripts/pore_app.py
```

Defaults: normalization enabled, background removal 2, Gaussian smoothing strength 0, minimum contrast 2, minimum pore area 100 px², and Detailed sampling (48 points per side). Gaussian strength 0 applies no smoothing.

## Repository

- `scripts/`: application and analysis code
- `ui/`: interface assets
- `packaging/`: Windows build configuration
- `requirements.txt`: runtime dependencies

Input images, trained weights, analysis results and compiled applications are excluded from Git. Measurements describe 2D masks; they are not direct measurements of 3D porosity or sphericity.

The `app` branch contains application and build sources. Local experiments, training scripts, test tools, input images and analysis results are excluded. Compiled Windows apps are distributed through Releases.
