# SEM footer correction — 2026-09-21

The original saved analysis ROI included the acquisition-information and scale-bar panel for `top_1_A_1kx-1`, `top_2_B_1kx-1`, and `top_3_A_10kx-1`. Their inputs were 1024 × 888 rather than 1024 × 768 SEM pixels. Export pixel matching confirmed the saved annotations, but did not establish that every annotation was scientifically valid.

Three reference masks were wholly inside the information panel:

| Image | Original pore ID | Pixels removed |
|---|---:|---:|
| top_2_B_1kx-1 | 135 | 2,722 |
| top_2_B_1kx-1 | 136 | 26,554 |
| top_3_A_10kx-1 | 89 | 2,541 |

The corrected snapshot crops raw inputs and masks to y=[0,768) before applying the application's 2nd/98th-percentile normalization. No valid SEM-region label pixels are changed and no real pore is partially truncated. The other thirteen images are unchanged. Total reference count changes from 2,021 to 2,018. Original exports, previous models and previous metric files are retained.

Historical CV5, refit15, LOO16 and derived candidate experiments used the affected data. Even evaluations whose test image was correctly cropped used models trained on other affected images. These scores must not be treated as clean-training results; top-image scores also included the wrong evaluation region. Cropping predictions from an old model does not fix its training provenance.

Corrected run: [footer_clean_5_20260921_104127](runs/footer_clean_5_20260921_104127/index.html). Exact audit: [footer_audit.json](runs/footer_clean_5_20260921_104127/footer_audit.json).

This repair retrains the same five selected folds (7, 8, 6, 12, 15), each from original SAM, with the respective test image excluded. It does not retrain the remaining eleven LOO folds or create an all-images deployment model. The desktop application continues to use its original checkpoint.

Follow-up: [clean16_methods_20260921_110028](runs/clean16_methods_20260921_110028/index.html) completes all sixteen corrected 15/1 models, reusing those five and training the other eleven. It compares fixed Relaxed-only and Two-stage selection with area metrics on all sixteen images, plus separately measured CPU inference/selection on three preselected images. There is still no corrected all-images deployment refit; the app is unchanged.
