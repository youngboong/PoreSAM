# Core checks

Run from the project root in the application's Python environment.

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

These checks cover containment, measurements, CPU selection and time estimates.

For UI checks, install the optional test dependency and use an installed Microsoft Edge:

```powershell
python -m pip install playwright==1.62.0
python tests/check_automate_merge.py
python tests/check_operation_stop.py
python tests/check_region_queue.py
python tests/check_separate_exports.py
```

These tests use isolated data under `outputs/ui_checks/` and verify supplementation, merging, Stop, Undo, queued regions and image/report exports.

After building the Windows app:

```powershell
python tests/check_desktop_bundle.py --bundle dist/PoreSAM
python tests/check_desktop_windows.py --bundle dist/PoreSAM
```

The first runs CPU SAM and export checks in the packaged app; the second checks independent windows and workspace reuse. They create isolated data under `outputs/desktop_checks/`. Playwright and PyInstaller are not application runtime requirements.
