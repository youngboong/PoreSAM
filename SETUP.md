# English browser environment

These instructions apply to the `english` browser branch on Windows PowerShell. For the current Windows desktop app and its separate CPU build environment, use [app / DESKTOP.md](https://github.com/youngboong/PoreSAM/blob/app/DESKTOP.md).

## CPU

```powershell
conda create -n pore python=3.11 -y
conda activate pore
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python scripts/check_environment.py --device cpu
```

Download the SAM 2.1 Small checkpoint separately from the [official SAM 2 repository](https://github.com/facebookresearch/sam2) and place it at `checkpoints/sam2.1_hiera_small.pt`. Input images, weights and generated results are not included in Git.

```powershell
python scripts/pore_editor.py --device cpu
```

Open http://127.0.0.1:8765 in a browser. Use `--port 8766` if needed; stop the server with Ctrl+C. See [EDITOR.md](EDITOR.md) for the workflow. This branch does not include `pore_app.py` or Windows desktop packaging.

## NVIDIA GPU

In a separate environment, install CUDA PyTorch before the remaining requirements:

```powershell
conda create -n pore-gpu python=3.11 -y
conda activate pore-gpu
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python scripts/check_environment.py --device cuda
python scripts/pore_editor.py --device cuda
```

The default `--device auto` selects CUDA when available, otherwise CPU. A CUDA-compatible GPU and driver are required for `--device cuda`. The optional SAM CUDA extension is omitted on Windows; CUDA inference still uses PyTorch. SAM extension-dependent small-hole/speckle postprocessing is unavailable.

`requirements-lock.txt` records the earlier CUDA development environment. For CPU, use `requirements.txt` after installing CPU PyTorch instead of that CUDA lock file.

## Development and checks

Select the environment through VS Code's **Python: Select Interpreter**. Optional browser checks use `requirements-dev.txt` and an installed Microsoft Edge.

`check_environment.py` checks imports, device operations, TorchVision NMS, SAM configuration construction and TIFF reading. It does not download trained weights or validate segmentation accuracy. Browser data is stored in the project's `outputs` directory.
