# PoreSAM environment

The default `app` branch is the English Windows desktop edition. Use the Python 3.11 **`pore-app-cpu`** environment in [DESKTOP.md](DESKTOP.md) for installation, CPU-only dependencies and packaging. Model weights must be prepared separately at `checkpoints/sam2.1_hiera_small.pt`.

After installing that environment:

```powershell
conda activate pore-app-cpu
python scripts/pore_app.py
```

The native window manages its own local server. Data is stored under `%LOCALAPPDATA%/PoreSAM/English/outputs` by default. See [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for function examples and output paths.

## Optional browser entry point

The desktop branch retains a browser entry point for development:

```powershell
conda activate pore-app-cpu
python scripts/pore_editor.py --device cpu
```

Open http://127.0.0.1:8765. This entry point defaults to the project's `outputs` directory, unlike the desktop launcher. Do not use both entry points to edit the same data directory concurrently.

## Earlier browser editions

- [English browser setup](https://github.com/youngboong/PoreSAM/blob/english/SETUP.md)
- [Korean browser setup](https://github.com/youngboong/PoreSAM/blob/korean/SETUP.md)

These branches have their own `pore` environment instructions and older feature sets. They do not contain the desktop launcher. The CUDA development lock file in `requirements-lock.txt` is not the dependency list for the CPU Windows build; use `requirements-app-cpu.txt` as described in DESKTOP.md.
