"""Train/reuse all sixteen clean leave-one-image-out checkpoints."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from train import ROOT, dump, sha
from leave_one_out import verify_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    source=ROOT/'fine_tuning/runs/footer_clean_5_20260921_104127'
    if args.resume:
        out=args.resume.resolve();plan=json.loads((out/'plan.json').read_text())
    else:
        old=json.loads((source/'plan.json').read_text())
        assert json.loads((source/'verification.json').read_text())['passed']
        name='clean16_methods_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        out=ROOT/'fine_tuning/runs'/name;out.mkdir()
        models=ROOT/'fine_tuning/models'/name;models.mkdir()
        shutil.copytree(source/'dataset',out/'dataset')
        # Inputs and frozen encoder are identical; copied features cannot leak labels.
        shutil.copytree(source/'features',out/'features')
        plan=copy.deepcopy(old)
        plan.update(run=str(out),dataset=str(out/'dataset'),model_directory=str(models),
            selected_folds=list(range(1,17)),reused_folds=old['selected_folds'],reuse_source=str(source),
            protocol='16 clean images; 15 train / 1 held out independently; fixed 300 updates. Reuse five verified clean models, train eleven from original SAM. Compare fixed relaxed-only and two-stage methods without reference-dependent selection.',
            limitations='Exploratory image-wise leave-one-out on previously examined related captures. No independent external validation. Application unchanged.',
            timing_protocol='Measure shared SAM generation separately; time complete relaxed and two-stage postprocessing paths. Model loading, training, metrics, figures and disk I/O excluded from analysis timing. CPU-specific inference benchmark reported separately.')
        dump(out/'plan.json',plan)
        (out/'source').mkdir()
        for script in Path(__file__).parent.glob('*.py'):shutil.copy2(script,out/'source'/script.name)
        for fold in plan['reused_folds']:
            old_checkpoint=Path(old['model_directory'])/f'fold_{fold}.pt'
            expected=json.loads((source/f'fold_{fold}/model_verification.json').read_text())['checkpoint_sha256']
            assert sha(old_checkpoint)==expected
            shutil.copy2(old_checkpoint,models/old_checkpoint.name)
            shutil.copytree(source/f'fold_{fold}',out/f'fold_{fold}')
            result=json.loads((out/f'fold_{fold}/result.json').read_text())
            result.update(checkpoint=str(models/old_checkpoint.name),reused_from=str(old_checkpoint))
            dump(out/f'fold_{fold}/result.json',result)
    assert sha(Path(plan['dataset'])/'manifest.json')==plan['dataset_manifest_sha256']
    print(f'RUN {out}',flush=True)
    for fold in range(1,17):
        if not (out/f'fold_{fold}/result.json').exists():
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('cross_validate.py')),
                '--worker-plan',str(out/'plan.json'),'--fold',str(fold)],check=True)
        verify_model(plan,fold)
        print(f'CHECKPOINT {fold}/16 VERIFIED',flush=True)
    dump(out/'training_complete.json',dict(folds=16,reused=plan['reused_folds'],
        newly_trained=[f for f in range(1,17) if f not in plan['reused_folds']],app_model_replaced=False))
    print(f'TRAINING COMPLETE {out}',flush=True)


if __name__=='__main__':main()
