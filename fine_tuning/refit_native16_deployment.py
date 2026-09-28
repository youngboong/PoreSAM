"""Build a deployment checkpoint from all 16 corrected training images.

Uses the existing fixed training recipe; no held-out performance is claimed for
the refit. Writes a new checkpoint and provenance without changing app defaults.
"""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from train import ROOT, dump, sha


def main():
    source=ROOT/'fine_tuning/runs/loss_resolution5_20260923_135551'
    original=json.loads((source/'plan.json').read_text())
    data=Path(original['dataset'])
    assert sha(data/'manifest.json')==original['dataset_manifest_sha256']
    records=json.loads((data/'manifest.json').read_text())['records']
    assert len(records)==16
    name='native16_deployment_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'fine_tuning/runs'/name;out.mkdir()
    models=ROOT/'fine_tuning/models'/name;models.mkdir()
    plan={k:original[k] for k in ('dataset','dataset_manifest_sha256','base_checkpoint','base_checkpoint_sha256','seed','steps','batch','lr')}
    plan.update(model_directory=str(models),folds=[],prompted_evaluation=False,control_run=original['control_run'],loss='Native-resolution BCE+Dice; unchanged score supervision',
                protocol='Full corrected 16-image refit from original SAM; fixed 300 updates. No validation or checkpoint selection.',
                intended_selection='two_stage_nested',source_comparison_run=str(source))
    dump(out/'plan.json',plan)
    (out/'source').mkdir()
    for script in ('refit_native16_deployment.py','train.py'):
        shutil.copy2(Path(__file__).with_name(script),out/'source'/script)
    shutil.copy2(source/'source/worker.py',out/'source/worker.py')
    (ROOT/'outputs/native16_deployment_active.txt').write_text(str(out))
    torch.set_num_threads(4)
    print(f'RUN {out}',flush=True)
    subprocess.run([sys.executable,'-u',str(out/'source/worker.py'),str(out/'plan.json'),'0'],check=True)
    checkpoint=models/'all_images.pt'
    config=json.loads((out/'final_all_images/config.json').read_text())
    assert set(config['train_images'])=={r['name'] for r in records} and not config['validation_images']
    base=torch.load(plan['base_checkpoint'],map_location='cpu',weights_only=True)['model']
    fitted=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    assert base.keys()==fitted.keys()
    changed=[]
    for key,tensor in fitted.items():
        assert tensor.shape==base[key].shape and torch.isfinite(tensor).all(),key
        if not torch.equal(tensor,base[key]):
            assert key in config['trainable_names'],key
            changed.append(key)
    assert changed and sha(Path(plan['base_checkpoint']))==plan['base_checkpoint_sha256']
    verified=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                  train_images=config['train_images'],changed_tensors=changed,
                  frozen_tensors_unchanged=True,finite=True,selection='two_stage_nested',
                  independent_test_performance=None,app_model_replaced=False)
    dump(out/'model_verification.json',verified)
    dump(ROOT/'fine_tuning/models/latest_deployment.json',dict(run=str(out),**verified))
    print('COMPLETE '+str(checkpoint),flush=True)


if __name__=='__main__':
    main()
