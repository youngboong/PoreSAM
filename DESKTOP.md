# Install and build PoreSAM

Windows x64, Python 3.11 and Microsoft Edge WebView2 Runtime are required. Run the commands in PowerShell from the project root.

## Install

```powershell
conda create -n pore-app-cpu python=3.11 pip -y
conda activate pore-app-cpu
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install setuptools==83.0.0 wheel==0.47.0
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

Download the SAM 2.1 Small checkpoint from the [official SAM 2 repository](https://github.com/facebookresearch/sam2) and place it at `checkpoints/sam2.1_hiera_small.pt`. The checkpoint is not stored in this repository.

```powershell
python scripts/pore_app.py
```

The app opens a native window and manages its local server. CPU is the default. The optional browser entry point is `python scripts/pore_editor.py --device cpu`; open http://127.0.0.1:8765 for that mode.

## Storage

Print the desktop's existing default data directory:

```powershell
python -c "import sys; sys.path.insert(0, 'scripts'); from pore_app import default_data_dir; print(default_data_dir())"
```

Use `python scripts/pore_app.py --data-dir .\outputs` to choose the project's output directory. The browser entry point defaults to project-local `outputs`. Avoid editing the same data directory from both entry points concurrently.

The desktop keeps uploads in `<data root>/projects`, revisions in `<data root>/manual_edits`, and logs in the sibling `logs` folder. Installation updates do not remove those files. Additional windows receive independent workspaces and do not share live edits. See [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for the full layout.

Pore Details is a separate native window that can move outside its owning editor. CSV and plot exports use a Windows Save As dialog.

## Build

Install the build tool only when packaging:

```powershell
conda activate pore-app-cpu
python -m pip install pyinstaller==6.22.3
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

Output: `dist/PoreSAM/PoreSAM.exe`. Distribute the entire `dist/PoreSAM` folder, including `_internal`. The build includes the checkpoint and rejects CUDA-enabled PyTorch to keep the distribution CPU-only.

To create an installer, compile `packaging/PoreSAM.iss` using Inno Setup 6 after building the app. It creates `dist/installer/PoreSAM-Setup-0.1.0.exe` and does not require administrator rights.

## Verification

See [tests/README.md](tests/README.md). Browser integration tests use an optional Playwright installation; it is not a runtime dependency.
