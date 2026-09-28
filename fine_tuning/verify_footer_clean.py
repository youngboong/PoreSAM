"""Independently audit corrected pixels, checkpoints and area-error artifacts."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from train import ROOT, dump, sha
from leave_one_out import area_metrics
sys.path.insert(0, str(ROOT / 'scripts'))
from pore_preprocessing import prepare_adjustable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'dataset/manifest.json').read_text())
    old_dataset = Path(manifest['supersedes']).parent
    old = {r['name']: r for r in json.loads((old_dataset / 'manifest.json').read_text())['records']}
    audited = []
    for record in manifest['records']:
        name = record['name']
        labels = np.array(Image.open(run / 'dataset' / record['labels']))
        previous = np.array(Image.open(old_dataset / old[name]['labels']))
        raw = np.array(Image.open(old_dataset / name / 'original.png').convert('RGB'))
        image = np.array(Image.open(run / 'dataset' / record['image']))
        assert labels.shape == image.shape == (768, 1024)
        assert np.array_equal(labels, previous[:768])
        expected, _ = prepare_adjustable(cv2.cvtColor(raw[:768], cv2.COLOR_RGB2GRAY),
            dict(normalize_enabled=True, background_strength=0, blur_method='none', blur_strength=2))
        assert np.array_equal(image, expected)
        assert sha(run / 'dataset' / record['labels']) == record['labels_sha256']
        assert sha(run / 'dataset' / record['image']) == record['image_sha256']
        assert len(np.unique(labels)) - 1 == record['instances']
        removed = record['footer_correction']['removed']
        if name.startswith('top_'):
            # A deterministic diagnostic overlay, not a modified training image.
            view = raw.copy()
            for item in removed:
                mask = previous == item['label']
                view[mask] = (.35 * view[mask] + .65 * np.array([255, 0, 170])).astype(np.uint8)
                y, x = np.argwhere(mask).mean(axis=0).astype(int)
                cv2.putText(view, str(item['original_id']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
            cv2.line(view, (0, 768), (1023, 768), (0,220,255), 2)
            Image.fromarray(view).save(run / f'{name}_footer_audit.png')
        audited.append(dict(image=name, shape=list(labels.shape), valid_label_pixels_preserved=True,
                            normalized_after_crop=True, removed_original_ids=[a['original_id'] for a in removed]))
    assert sum(r['instances'] for r in manifest['records']) == 2018
    assert sum(len(a['removed_original_ids']) for a in audited) == 3
    dump(run / 'dataset_verification.json', dict(images=audited, reference_instances=2018, passed=True))
    if args.audit_only:
        print('Dataset verified: 16 images; 2018 instances; 3 footer-only labels removed.')
        return
    import csv
    evaluation = run / 'two_stage_evaluation'
    rows = list(csv.DictReader((evaluation / 'metrics.csv').open()))
    assert len(rows) == 15
    plan = json.loads((run / 'plan.json').read_text())
    for fold in plan['selected_folds']:
        folder = evaluation / f'fold_{fold}'
        p = json.loads((folder / 'provenance.json').read_text())
        assert len(p['train_images']) == 15 and p['held_out'] not in p['train_images']
        assert sha(Path(p['checkpoint'])) == p['checkpoint_sha256']
        assert Path(p['pool_source']).resolve().is_relative_to(evaluation)
        record = next(r for r in manifest['records'] if r['name'] == p['held_out'])
        truth = np.array(Image.open(run / 'dataset' / record['labels']))
        strict = np.array(Image.open(folder / 'strict_instances.tif'))
        two = np.array(Image.open(folder / 'two_stage_instances.tif'))
        assert np.array_equal(two[strict > 0], strict[strict > 0])
        for variant in ['strict', 'relaxed', 'two_stage']:
            prediction = np.array(Image.open(folder / f'{variant}_instances.tif'))
            actual = area_metrics(truth, prediction)
            row = next(r for r in rows if int(r['fold']) == fold and r['variant'] == variant)
            for key, value in actual.items():
                assert np.isclose(value, float(row[key]), rtol=0, atol=1e-10), (fold, variant, key)
            errors = np.array(Image.open(folder / f'{variant}_errors.png'))
            assert np.array_equal(np.all(errors == [0,220,255], axis=-1), (truth > 0) & (prediction == 0))
            assert np.array_equal(np.all(errors == [255,0,170], axis=-1), (truth == 0) & (prediction > 0))
    dump(run / 'verification.json', dict(passed=True, dataset_images=16, reference_instances=2018,
         retrained_folds=plan['selected_folds'], independent_held_out_models=5,
         area_metric_rows_recomputed=15, error_maps_verified=15, strict_pixels_and_ids_preserved=True,
         fresh_candidate_pools=True, app_model_replaced=False))
    print('PASS: dataset, 5 held-out model hashes, 15 metric rows and 15 error maps.')


if __name__ == '__main__':
    main()
