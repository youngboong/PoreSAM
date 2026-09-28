"""Refit on explicit frozen datasets, keeping a new evaluation image held out."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from train import ROOT, dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs='+', required=True)
    parser.add_argument('--out-dataset', type=Path, required=True)
    parser.add_argument('--test-dataset', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.out_dataset.resolve()
    assert not dataset.exists(), 'Use a new immutable dataset folder.'
    sources = [p.resolve() for p in args.datasets]
    heldout = args.test_dataset.resolve()
    test = json.loads((heldout/'manifest.json').read_text())
    test_images = [np.array(Image.open(heldout/r['image']).convert('RGB')) for r in test['records']]
    test_hashes = {r['original_file_sha256'] for r in test['records']}
    records = []; names = set(); image_hashes = set(); provenance = []
    for source in sources:
        manifest = json.loads((source/'manifest.json').read_text())
        provenance.append(dict(dataset=str(source), manifest_sha256=sha(source/'manifest.json'),
                               label_review=manifest['label_review']))
        for original in manifest['records']:
            record = copy.deepcopy(original)
            assert record['name'] not in names, 'Duplicate name.'
            assert record['image_sha256'] not in image_hashes, 'Duplicate training image.'
            for field in ['image', 'labels']:
                assert sha(source/record[field]) == record[field+'_sha256']
            assert record['original_file_sha256'] not in test_hashes, 'Test original in training.'
            pixels = np.array(Image.open(source/record['image']).convert('RGB'))
            assert not any(np.array_equal(pixels, t) for t in test_images), 'Test pixels in training.'
            names.add(record['name']); image_hashes.add(record['image_sha256'])
            record['split'] = 'train'
            record['prepared_source'] = str(source)
            records.append(record)
    assert records and test['records']
    dataset.mkdir(parents=True)
    for record in records:
        source = Path(record['prepared_source'])
        shutil.copytree(source/record['name'], dataset/record['name'])
    dump(dataset/'manifest.json', dict(version=2, records=records, provenance=provenance,
         label_review='Source-specific review status is retained in provenance.',
         input_processing='App imported grayscale/RGB image cropped to analysis ROI; no background removal or extra blur.',
         split_notes='All listed images train. Separate test dataset never enters optimization; related sample captures can occur across datasets.'))
    name = f'refit_{len(records)}_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT/'fine_tuning/runs'/name; out.mkdir(parents=True)
    models = ROOT/'fine_tuning/models'/name; models.mkdir(parents=True)
    base = ROOT/'checkpoints/sam2.1_hiera_small.pt'
    plan = dict(dataset=str(dataset), dataset_manifest_sha256=sha(dataset/'manifest.json'),
                base_checkpoint=str(base), base_checkpoint_sha256=sha(base),
                model_directory=str(models), seed=260917, steps=300, batch=4, lr=1e-5, folds=[],
                protocol='Full-data refit from original checkpoint, fixed 300 updates; no validation selection.',
                test_dataset=str(heldout), test_manifest_sha256=sha(heldout/'manifest.json'),
                train_images=len(records), train_instances=sum(r['instances'] for r in records),
                limitations='Previous test image promoted to training by user request. This is a sequential exploratory experiment; related sample images occur in training and test.')
    dump(out/'plan.json', plan)
    (out/'source').mkdir()
    for script in Path(__file__).parent.glob('*.py'):
        shutil.copy2(script, out/'source'/script.name)
    print(f'REFIT {out}: {len(records)} images, {plan["train_instances"]} pores; test excluded.', flush=True)
    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('cross_validate.py')),
                    '--worker-plan', str(out/'plan.json'), '--fold', '0'], check=True)
    result = json.loads((out/'final_all_images/result.json').read_text())
    checkpoint = Path(result['checkpoint'])
    baseline = torch.load(base, map_location='cpu', weights_only=True)['model']
    tuned = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    config = json.loads((out/'final_all_images/config.json').read_text())
    trainable = set(config['trainable_names']); changed = []
    assert baseline.keys() == tuned.keys()
    for key, value in tuned.items():
        assert value.shape == baseline[key].shape and torch.isfinite(value).all(), key
        if not torch.equal(value, baseline[key]):
            assert key in trainable, f'Frozen tensor changed: {key}'
            changed.append(key)
    assert changed and sha(base) == plan['base_checkpoint_sha256']
    assert sha(dataset/'manifest.json') == plan['dataset_manifest_sha256']
    assert sha(heldout/'manifest.json') == plan['test_manifest_sha256']
    dump(out/'model_verification.json', dict(checkpoint_sha256=sha(checkpoint), changed_tensors=changed,
         frozen_tensors_unchanged=True, finite=True, app_model_replaced=False))
    del baseline, tuned
    dump(ROOT/'fine_tuning/models/latest_experiment.json', dict(
         run=str(out.relative_to(ROOT/'fine_tuning')),
         checkpoint=str(checkpoint.relative_to(ROOT/'fine_tuning')),
         status='experimental_all_images_refit', app_model_replaced=False))
    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('evaluate_new_image.py')),
                    '--dataset', str(heldout), '--training-run', str(out)], check=True)
    print(f'COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
