# PoreSAM fine-tuning

SAM 2.1 Small training runs and their immutable evaluation records are managed here.

## Current deployment (2026-09-28)

The app now selects `sam2.1_hiera_small_native16_20260923.pt`, a 300-update full corrected 16-image refit using native-resolution BCE + Dice and unchanged score-head supervision. Two-stage nested selection remains enabled. See `refit_native16_deployment.py` and [desktop deployment notes](../DESKTOP.md#current-app-snapshot-2026-09-28). Weights and experiment outputs are local artifacts excluded from Git. The full-data refit has no independent test score.

The dated sections below document earlier experiments and deployments.

## App deployment (2026-09-21)

The selected application pipeline is now Two-stage nested. A separate [full corrected 16-image refit](runs/clean16_deployment_20260921_162614/model_verification.json) uses the same fixed 300 updates, batch 4, learning rate `1e-5` and seed 260917. All sixteen images train the final checkpoint; it has no independent test score. The earlier LOO and nested replay scores continue to describe their held-out models, not this refit.

`checkpoints/default_model.json` selects `sam2.1_hiera_small_clean16_20260921.pt` (SHA-256 `7894048986b1b83d68de336903f95c83f8d1db46d4b46d3761f074a956e2f4a1`). Original SAM weights remain available for training. An explicit `--checkpoint` still overrides the default. Both automatic analysis and prompted additions load this default; cached automatic candidates require the same checkpoint hash and execution device. The packaged application includes the selected checkpoint and manifest. See `refit_clean16_deployment.py` for the refit procedure and its CPU compatibility record in the run folder.

## Experimental nested-pore selection (2026-09-21)

`nested_candidates.py` compares each existing parent mask with internal SAM candidates before global box NMS. It replaces a parent only when at least two mostly contained, disjoint children have supported dark-to-bright rims and the remaining gap contains a broad, consistently bright region relative to the local exterior. A distance transform rejects thin gaps. Decisions use image pixels and predicted candidates only; reference labels are never passed to the selector. Children are clipped to their parent, unchanged instance IDs are preserved, and children are not recursively split.

This is an optional experiment. Both `relaxed_nested` and `two_stage_nested` are supported by `compare_methods16.select`; the original variants and application defaults remain unchanged. The replacement runs **after** each base method finishes, so supplementation cannot add the rejected parent back. It can remove false-positive area but also remove true pore area where children are missing or incomplete. It does not add new SAM prompts or infer a 3D surface from SEM intensity.

Replay the existing corrected 16-image candidate pools without retraining or inference:

```powershell
python fine_tuning/evaluate_nested.py
python fine_tuning/evaluate_nested.py --folds 10 --out fine_tuning/runs/nested_pi35_trial
python fine_tuning/render_nested.py --run fine_tuning/runs/nested16_20260921_153240
python -m unittest discover -s tests -p "test_*candidates.py"
```

Each run creates a new output directory with `index.html`, `metrics.csv`, `summary.json`, frozen settings and source copies, and per-image masks, comparison images and decision logs. `fold_10/cases.png` compares the same A–D crops from the earlier PI35 diagnosis using actual automatic predictions. Area IoU and one-to-one instance F1 at IoU >= 0.5 are both reported. Candidate preparation and resolution times measure only the added CPU pass on saved GPU pools, not end-to-end CPU inference.

Comparisons use a 40% gold overlay to show the full positive mask area, plus instance outlines. `render_nested.py` regenerates figures from existing masks and records rendering provenance separately; it verifies that masks, metrics, decisions and the original evaluation plan remain unchanged.

PI35_5kx informed the rule design; all sixteen images have previously been inspected. These results are development evaluations, not independent test performance. Keep the recorded settings fixed across images when comparing results, and inspect regressions as well as the PI35 examples.

First completed replay: [16-image results and PI35 A–D crops](runs/nested16_20260921_153240/index.html). On PI35_5kx, Two-stage area IoU rises from 0.8017 to 0.8532 and instance F1 from 0.6780 to 0.7360; A–C are split while D is preserved. Missed reference area rises from 1.03% to 3.17%, so the small-pore recovery is incomplete. Across 16 images, mean area IoU changes from 0.7610 to 0.7624 (4 improved, 3 worse, 9 unchanged). The regressions are PI100_5kx, PI35_5kx-2 and PI35_5kx-4_bse. This supports continued development, not switching the app default. Ten behavioral/integration/metric tests pass; all 32 replay outputs preserve original pixels outside replaced parents and add no pixels outside the original masks. Observed additional CPU time averages 13.09 seconds for candidate preparation plus 1.28 seconds for Two-stage resolution, excluding inference and I/O; this was not an isolated performance benchmark.

## Current data correction (2026-09-21)

Completed full comparison: [16-image area accuracy and CPU timing](runs/clean16_methods_20260921_110028/index.html). Two-stage wins area IoU on 13/16 images (mean 0.7610 versus 0.7389 for Relaxed-only); total-area error averages 12.75% versus 20.46%. Relaxed-only misses less area but adds more false-positive area. CPU measurements on three preselected images average 225.08 versus 212.48 seconds per image (+12.60 seconds, +5.93%) for inference plus selection. These are development results on this collection; CPU subset and full GPU metrics remain separate. No app defaults have changed.

The follow-up full comparison uses `compare_clean16.py` to reuse the five verified corrected models and train the remaining eleven with the identical 15/1 recipe. `compare_methods16.py --run <new-run>` measures the fixed Relaxed-only and Two-stage pipelines on all sixteen held-out images. It saves `methods_gpu/metrics.csv`, `timing.csv`, `summary.json`, an HTML report and per-image reference/error overlays. Both methods share one SAM generation per image; their complete selection paths are timed separately, with Two-stage charged for strict selection as well as supplementation. Training, model loading, metrics, file writes and plots are excluded from analysis time. This is an experimental comparison, not an application default change.

```powershell
python fine_tuning/compare_clean16.py
python fine_tuning/compare_methods16.py --run fine_tuning/runs/clean16_methods_<timestamp>
python fine_tuning/compare_methods16.py --run fine_tuning/runs/clean16_methods_<timestamp> --device cpu --folds 1 7 15 --repeats 2
```

CPU measurements use the app's float32 inference, 8 threads, batch 8 and unchanged Detailed 48×48 point grid. The three CPU images are preselected for dense PI, difficult PI300 and complex top structures. GPU accuracy/timing uses bfloat16, 4 threads and batch 4; CPU results are kept separate because precision can change detections. No elapsed GPU time is represented as CPU time. Postprocessing repetitions alternate method order; reported post time is their median. SAM generation is measured once per image, so its single-run timing remains subject to ordinary system variability.

The historical snapshots below contained SEM information panels in three `top_*` images and three footer-only reference masks. Their models and scores are retained for provenance, but must not be presented as results from clean SEM-only training. See [the data quality notice](DATA_QUALITY_NOTICE.md).

The corrected snapshot is `runs/footer_clean_5_20260921_104127/dataset/`: 16 images, 2,018 reference instances. Raw images and masks are cropped to the visually verified SEM boundary at y=768 **before** brightness normalization. All valid SEM-region labels are preserved. Five independent 15-train/1-held-out models (folds 7, 8, 6, 12, 15) are retrained from original SAM with the same 300-update recipe, then compared using area metrics. The other eleven historical folds are not retrained by this correction.

```powershell
python fine_tuning/retrain_clean_footer.py
```

This dataset-specific repair creates new dated run/model folders and does not modify source exports or the app. The 768-row boundary is verified for these 16 images, not a general rule for other SEM acquisitions. For future datasets, verify the actual information-panel boundary: `prepare_dataset.py` recovers saved analysis masks, and those saved dimensions alone do not prove that a footer was excluded.

Outputs: `runs/footer_clean_5_<timestamp>/index.html`, `footer_audit.json`, `two_stage_evaluation/`; weights: `models/footer_clean_5_<timestamp>/fold_*.pt`. `models/latest_clean_experiment.json` records these held-out models; there is no clean all-images refit yet.

## Dataset

The initial training snapshot uses 14 reviewed exports from `data/260917/` (42 PNGs). Their underlying saved revisions contain **1,772 pore instances**. The user confirmed these are manually reviewed final results. Later additions to the source folder are not automatically included in the frozen training snapshot.

`prepare_dataset.py` locates saved app projects and revisions, checks that the original pixels match, and regenerates the colored export **pixel-for-pixel** to verify the exact revision. It never treats color overlays or ID text as model input or converts them into approximate labels. Overlapping/empty masks and duplicate input images are rejected.

Prepared data live in `datasets/260917_v1/`:

- `manifest.json`: source revisions, hashes, pore counts, image groups and provenance.
- `<image>/image.png`: imported image cropped to the saved analysis area. In this historical snapshot, three `top_*` images incorrectly retain their SEM information footer; use the corrected snapshot above.
- `<image>/instances.tif`: 16-bit instance IDs, with 0 as background.
- `<image>/source_masks.npz`: a copy of the exact saved masks, preserving original pore IDs.

No additional brightness normalization, background removal or smoothing is applied for this experiment. Source app settings are recorded for provenance, but are not applied to training images.

## Environment and commands

Run from the repository root. The existing `pore` environment has GPU-enabled PyTorch and SAM2. No additional training requirements file or package installation is needed for this machine. Training does not change the CPU distribution environment.

```powershell
conda activate pore
python fine_tuning/prepare_dataset.py
```

Preparation refuses to overwrite an existing dataset. Use `--out fine_tuning/datasets/<new-version>` for a new snapshot.

The current experiment uses five-fold cross-validation:

```powershell
python fine_tuning/cross_validate.py
```

Each execution creates a new dated run and model folder. Checkpoints and raw data are excluded from Git.

## Cross-validation protocol

- Five folds keep similarly named sample/magnification captures together. Validation fold sizes are 4, 3, 3, 2 and 2 images. Every image is held out once.
- Each fold starts independently from the original checkpoint, with a fresh optimizer.
- Fixed 300 updates, batch size 4 pore prompts, learning rate `1e-5`, AdamW, gradient clipping, seed 260917. No fold checkpoint is selected using its validation score.
- Only mask decoder parameters are trained. The image encoder, prompt encoder and cached high-resolution projections remain frozen.
- Training uses horizontal/vertical flips, random interior points and boxes with up to 10% outward expansion. Validation prompts are fixed and identical for both models.
- Loss combines binary cross-entropy and Dice for the single-mask token and the best of the three ambiguous mask tokens, plus IoU score regression. At evaluation, mask choice uses the **predicted score**, not the known reference IoU.
- Frozen image features are cached and shared across folds. Reference masks and gradients from held-out images never enter optimization.
- Prompted IoU is reported separately from automatic detection. Automatic analysis uses the existing app filters, a 48-by-48 point grid, predicted IoU threshold 0.8 and stability threshold 0.92. Both models use identical image-specific area/contrast settings. Automate is not run.
- Automatic comparison uses four prompts per batch and CUDA bfloat16 autocast for both models to fit the 6 GB GPU. Grid density is unchanged. The interrupted FP32 comparison is archived and excluded from scored results.
- Automatic precision/recall/F1 use one-to-one instance matches at IoU >= 0.5. The number of matches is maximized before total IoU.

After cross-validation, a separate model is refit on all 14 images with the same fixed settings. Cross-validation scores describe models trained on the other folds, not this full-data refit. Later new-image evaluations are stored separately under `runs/new_image_*`.

These are exploratory results: an earlier 9/3/2 pilot already used this collection. Filename grouping does not establish independent physical specimens. There is no external test set. Fold standard deviation is variability, not a confidence interval or calibrated uncertainty.

## Saved files

```text
fine_tuning/
  models/
    latest_experiment.json
    260917_cv5_<timestamp>/
      fold_1.pt ... fold_5.pt
      all_images.pt
  runs/
    260917_cv5_<timestamp>/
      plan.json
      source/                       # exact experiment scripts
      features/                     # frozen feature cache
      fold_1/ ... fold_5/
        config.json
        dataset_manifest.json
        training.csv
        result.json
        evaluation/                 # point/box per-pore CSV and summaries
        automatic_evaluation/       # masks, comparison PNGs, HTML, metrics
      final_all_images/
      summary.json
      out_of_fold_predictions.csv
      automatic_metrics.csv
      index.html                    # start here to review results
```

Checkpoints contain a standard SAM2 `model` state dictionary plus `finetuning` metadata. They can be loaded with `build_sam2` and the existing `configs/sam2.1/sam2.1_hiera_s.yaml` config on either CPU or GPU. They are **not automatically installed in the app**.

Verify model compatibility and frozen weights after completion:

```powershell
python fine_tuning/verify_models.py --run fine_tuning/runs/<run-folder>
```

Output: `<run-folder>/model_verification.json`. This checks all six checkpoints, verifies that only declared decoder weights changed, and runs a CPU box prediction with the all-images model.

`train.py` and `evaluate_automatic.py` also retain the initial pilot workflow. Its 9/3/2 split is superseded by the five-fold protocol; its results remain in a separate dated folder for provenance.

## Compare a new image

Keep new evaluation data separate from the training snapshot:

```powershell
python fine_tuning/prepare_dataset.py --only "<export-folder-name>" --out fine_tuning/datasets/external_<name> --split test --label-review "User-supplied reference"
python fine_tuning/evaluate_new_image.py --dataset fine_tuning/datasets/external_<name>
```

This compares the original checkpoint and the registered all-images checkpoint without training or selecting new thresholds. It rejects exact duplicates of training images and saves automatic instance metrics, combined area overlap, prompted boundary scores and comparison images under `runs/new_image_<timestamp>/index.html`. Similar material or magnification can still be present in training; a new filename alone does not establish an independent physical specimen.

Use `--training-run fine_tuning/runs/<refit-run>` to compare a specific all-images model instead of the registered model.

Add `--normalize` to apply the application's 2nd/98th percentile brightness normalization to the cropped evaluation ROI for both models. The normalized input and unchanged reference labels are saved inside the new run's `normalized_dataset/`. Automatic filtering and prompted comparisons use this same input. This does not retrain the model: earlier training snapshots remain unnormalized. Reports record evaluation preprocessing explicitly.

## Refit on 15 images and evaluate image 16

The 15-image snapshot adds the previously evaluated `PI100_5kx-4` to the initial 14 images: 1,885 pore instances. `PI100_5kx-5` remains outside training, with 136 reference instances. Its saved masks were verified against the uploaded export pixel-for-pixel.

```powershell
python fine_tuning/refit.py --datasets fine_tuning/datasets/260917_v1 fine_tuning/datasets/external_PI100_5kx-4_20260918 --out-dataset fine_tuning/datasets/260917_train15_v1 --test-dataset fine_tuning/datasets/external_PI100_5kx-5_20260918
```

This creates a new immutable dataset, checks for matching original-file hashes and duplicate test pixels before training, and starts from the original SAM checkpoint. Training settings remain at 300 updates, batch 4 and learning rate `1e-5`; it does not continue training the old fine-tuned checkpoint or run another five-fold experiment. The new test image does not select a checkpoint or stopping point.

Outputs are `runs/refit_15_<timestamp>/` for training/provenance and `models/refit_15_<timestamp>/all_images.pt` for weights. The script verifies finite weights and unchanged frozen tensors, updates the experimental model registry, and then creates a separate `runs/new_image_<timestamp>/` comparison. The desktop application's checkpoint is unchanged. Dataset preparation and refit refuse to overwrite existing dataset folders; choose a new `--out-dataset` for a repeat.

This is a sequential exploratory comparison: the previous test image is now training data, so its old score is not an independent test of the new 15-image model. Similar PI100 captures occur on both sides; independent physical specimens are not established.

## 16-image leave-one-out: area evaluation

```powershell
python fine_tuning/leave_one_out.py
# Resume completed folds without retraining them:
python fine_tuning/leave_one_out.py --resume fine_tuning/runs/loo16_area_<timestamp>/plan.json
```

Each of the 16 images is held out once; each separate model starts from original SAM and trains on the other 15 images for 300 updates. Both training and inference use app brightness normalization (2nd/98th percentile), with no background removal or blur. This differs from the earlier unnormalized training experiments. Automatic analysis uses the same 48-by-48 point grid and image-specific contrast/minimum-area filters for both models; Automate and reference-derived prompts are not used. The test image does not select a checkpoint or stopping time.

Results: `runs/loo16_area_<timestamp>/index.html`, updated after each completed image; `area_metrics.csv`; per-fold `area_evaluation/` images and metrics. Models: `models/loo16_area_<timestamp>/fold_1.pt` through `fold_16.pt`. The application and experimental model registry stay unchanged.

Area IoU is the main metric. Dice, pixel precision/recall, missed area, extra area and total-area error supplement it. Means give each image equal weight. Total-area error alone can hide equal amounts of missing and extra area. These metrics compare the union of pore pixels, not pore counts or individual diameter accuracy. Per-reference coverage CSVs include all reference pores without selecting only easy matches.

`comparison.png` shows reference/prediction overlays and error maps. `area_errors_comparison.png` is an additional side-by-side error image: **cyan = reference pore missed**, **magenta = predicted pore outside reference**. Native-resolution error maps are `baseline_area_errors.png` and `finetuned_area_errors.png`.

This is exploratory, image-wise leave-one-out, not specimen-grouped validation. Similar captures may appear on both sides; these previously examined data do not establish accuracy on independent new specimens.

## Five-image two-stage candidate trial

```powershell
python fine_tuning/trial_two_stage.py
```

Runs folds 7, 8, 6, 12 and 15 from the 16-image experiment using each image's held-out fine-tuned checkpoint. No retraining or app changes. Checkpoint hashes and excluded-image membership are verified. Existing traced candidate pools for 7/8 are reused; the other three are generated with the same normalized input, grid and CUDA precision.

`conservative_candidates.py` keeps strict detections intact and adds mostly uncovered candidates from a relaxed pool only when contrast and local boundary evidence support them. It takes no reference labels. Fixed parameters are declared before evaluation; additions are exclusive and existing pixels are preserved. It does not implement box proposals, mask replacement, or merging.

Outputs under `runs/two_stage_5_<timestamp>/`: `index.html`, `metrics.csv`, `summary.json`, candidate decision logs, per-fold masks, overlays and two-color error comparisons. Three variants use the same weights: strict .80/.92 quality thresholds, relaxed .70/.85 with existing filtering, and guarded additions. These five previously examined images constitute a development trial, not independent validation. CPU application runtime has not been benchmarked by the CUDA inference trial.

Implementation references: [Meta SAM2 training](https://github.com/facebookresearch/sam2/blob/main/training/README.md), [image predictor](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py), [mask decoder](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam/mask_decoder.py). This is a lightweight image-only decoder experiment using Meta's model modules, not the official multi-GPU video training recipe.
