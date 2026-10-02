# Core checks

Run from the project root in the application's Python environment.

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

These checks cover containment, deferred measurements, two-stage nested selection and cached hypotheses, CPU selection and time estimates.

For UI checks, install the optional test dependency and use an installed Microsoft Edge:

```powershell
python -m pip install playwright==1.62.0
python tests/check_automate_merge.py
python tests/check_operation_stop.py
python tests/check_region_queue.py
python tests/check_details_columns.py
python tests/check_separate_exports.py
```

These tests use isolated data under `outputs/ui_checks/` and verify supplementation, merging, Stop, Undo, queued regions, Skip, horizontal calibration, explicit details/summary refresh, 28 selectable/reorderable columns, CSV/plot exports, the detached details page and image/report exports.

After building the Windows app:

```powershell
python tests/check_desktop_bundle.py --bundle dist/PoreSAM
python tests/check_details_bundle.py --bundle dist/PoreSAM
python tests/check_desktop_windows.py --bundle dist/PoreSAM
```

The first runs CPU SAM and export checks in the packaged app. The details check uses a temporary local debugging port to verify the actual native details window, column picker, selection, explicit refresh and keyboard undo. The windows check verifies independent workspaces and slot reuse. They create isolated data under `outputs/desktop_checks/`. Playwright and PyInstaller are not application runtime requirements.

Report composer checks: `python tests/check_report_composer.py` exercises selection, captions, reordering, added images, editable text, persistence, HTML escaping, zero pores, PDF pagination and Korean output. PDF rendering uses a test-only PyMuPDF installation under outputs/report_test_deps. `python tests/check_report_bundle.py` opens the packaged WebView with isolated synthetic data, checks actual button clicks, preserves edited text after a pore deletion, exports PDF/HWPX, and closes only its test process.

The current Details checks also cover resizing, high-DPI plots, readable scatter axes, integer histogram boundaries, and matching screen/export bin counts. `python tests/check_multi_report.py` checks multi-image report persistence and export.

`python tests/check_folder_reports.py` checks direct four-file exports, byte-identical original TIFF preservation, filename collisions, multi-folder figures, source-file independence, persistence, and PDF/HWPX export. The native report check covers the same folder workflow in the packaged app.

`python -m unittest discover -s tests -p test_report_layout.py` checks seamless composition with different aspect ratios, histogram sizing, and HWPX spacing.
