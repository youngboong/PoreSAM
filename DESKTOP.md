# PoreSAM English desktop

The `app` branch adds a Windows desktop host for the existing English UI. Analysis algorithms and mask formats are shared with the browser edition.

## Run from source

Create the CPU environment using the build instructions below, then:

```powershell
conda activate pore-app-cpu
python scripts/pore_app.py
```

Requires Windows x64 and Microsoft Edge WebView2 Runtime. The host explicitly uses Edge Chromium; it does not fall back to Internet Explorer. CPU is the default. `--device auto` enables CUDA when available.

## Storage

- Application data: `%LOCALAPPDATA%/PoreSAM/English/outputs`
- Logs: `%LOCALAPPDATA%/PoreSAM/English/logs/desktop.log`
- WebView profile: `%LOCALAPPDATA%/PoreSAM/English/webview`
- Exports: the folder chosen through the Windows folder dialog

CSV and plot downloads open a Windows Save As dialog.

In the Windows app, **Pore Details** opens an independent native window. Move it outside the editor or onto another monitor, and resize it using standard window borders. Its table, histogram and scatter plot stay linked to the owning editor's current measurements and selection. Delete and Ctrl+Z also work from the details window. Closing the editor closes its details window; closing only the details window leaves the editor open. The browser edition retains the in-page popup.

Installation files are read-only assets. Updating or uninstalling the program does not remove analysis data. Launch the EXE again to open another window. Each window reserves an independent workspace with its own analysis history, model state, local port and WebView profile. The first window uses the original data folder; additional windows use `%LOCALAPPDATA%/PoreSAM/English/outputs_workspaces/workspace_0002/outputs` (then `0003`, etc.), and display their workspace number in the title bar. The first available workspace is reused on later launches, preserving its saved analyses. Windows do not share live edits or saved-analysis lists. Exports still go to the folder you choose. These local ports are independent of a browser server on 8765.

To continue the existing repository analyses during development, explicitly select their output directory:

```powershell
python scripts/pore_app.py --data-dir "C:/Users/YoungJin/Documents/pore_seg/outputs"
```

Do not run the browser edition against the same data directory concurrently. Pending drawing and previews are temporary; accepted masks remain saved. The close action waits for active processing and asks before discarding pending drafts.

## Build a Windows executable

Build on Windows using the separate Python 3.11 `pore-app-cpu` environment. The build rejects CUDA PyTorch to prevent accidentally shipping CUDA libraries. The existing `pore` environment is unchanged.

```powershell
conda create -n pore-app-cpu python=3.11 pip -y
conda activate pore-app-cpu
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install setuptools==83.0.0 wheel==0.47.0
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -r requirements-app-cpu.txt
python -m pip check
```

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

Output: `dist/PoreSAM/PoreSAM.exe`. Distribute the entire `dist/PoreSAM` directory, including `_internal`; the EXE alone is insufficient. The SAM 2.1 Small checkpoint must be present in `checkpoints/` when building and is included as an application asset. Build products, weights, input images and analysis outputs stay outside Git.

To create a per-user installer, install Inno Setup 6 and compile `packaging/PoreSAM.iss` after building the executable. Setup is English, requires no administrator rights and can create a desktop shortcut. Users must have Microsoft Edge WebView2 Runtime installed. Signing and distribution licensing review are release steps, not part of the development build.

## Verification

The hidden smoke-test option opens the real Windows window, checks English navigation and the default New Image tab, writes a JSON result, and closes the window/server:

```powershell
python scripts/pore_app.py --data-dir outputs/desktop_checks/data --smoke-test outputs/desktop_checks/smoke.json
```

After building, run `python scripts/check_desktop_bundle.py --check-stop` to test the EXE with isolated synthetic data. This checks cancellation during real CPU SAM inference, then runs automatic and box-prompted inference, exports three images, and generates PDF/HTML reports. Verification files go to `outputs/desktop_checks/`; the test app closes afterward.

Run Analysis and Automate change to a red **Stop** button while running. Stopping waits for the current model operation to reach a cancellation boundary. A stopped analysis is not added to saved analyses. Automate stages all discoveries and commits them together only on completion; stopping discards the entire batch and preserves the existing pores, measurements and saved revision. A completed Automate batch can be undone as one edit.

Native rendering: https://pywebview.flowrl.com/guide/web_engine
Packaging: https://pywebview.flowrl.com/guide/freezing
