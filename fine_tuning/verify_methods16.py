"""Verify complete clean-data method comparison artifacts and timing arithmetic."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from train import dump,sha
from leave_one_out import area_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    plan=json.loads((run/'plan.json').read_text());data=Path(plan['dataset'])
    manifest=json.loads((data/'manifest.json').read_text())
    assert sha(data/'manifest.json')==plan['dataset_manifest_sha256']
    records=manifest['records'];assert len(records)==16 and sum(r['instances'] for r in records)==2018
    verified=[]
    for dirname,expected_folds in [('methods_gpu',list(range(1,17))),('methods_cpu',[1,7,15])]:
        folder=run/dirname;config=json.loads((folder/'plan.json').read_text())
        assert config['folds']==expected_folds
        assert json.loads((folder/'summary.json').read_text())['completed_images']==len(expected_folds)
        for fold in expected_folds:
            local=folder/f'fold_{fold}';result=json.loads((local/'result.json').read_text())
            name=plan['folds'][fold-1]['validation'][0]
            r=next(r for r in records if r['name']==name)
            assert result['held_out']==name and set(result['train_images'])=={r['name'] for r in records}-{name}
            assert not result['reference_used_for_selection']
            assert sha(result['checkpoint'])==result['checkpoint_sha256']
            assert sha(data/r['labels'])==r['labels_sha256']
            truth=np.array(Image.open(data/r['labels']))
            assert truth.shape==(768,1024)
            maps={v:np.array(Image.open(local/f'{v}_instances.tif')) for v in ['strict','relaxed','two_stage']}
            assert np.array_equal(maps['two_stage'][maps['strict']>0],maps['strict'][maps['strict']>0])
            for v,pred in maps.items():
                row=next(m for m in result['metrics'] if m['variant']==v)
                for key,value in area_metrics(truth,pred).items():assert np.isclose(row[key],value,rtol=0,atol=1e-10)
                error=np.array(Image.open(local/f'{v}_errors.png'))
                assert np.array_equal(np.all(error==[0,220,255],axis=-1),(truth>0)&(pred==0))
                assert np.array_equal(np.all(error==[255,0,170],axis=-1),(truth==0)&(pred>0))
            t=result['timing'];assert t['sam_generation_seconds']>0
            for v in ['relaxed','two_stage']:
                assert t[v+'_post_seconds']>0
                assert np.isclose(t[v+'_post_seconds'],np.median(t['raw_repetitions'][v]))
                assert np.isclose(t[v+'_analysis_seconds'],t['sam_generation_seconds']+t[v+'_post_seconds'])
            verified.append(dict(device=config['device'],fold=fold,checkpoint_verified=True,metric_rows=3,error_maps=3,timing_verified=True))
        page=folder/'index.html'
        for link in re.findall(r'(?:href|src)="([^"]+)"',page.read_text(encoding='utf-8')):
            assert (page.parent/link).exists(),link
    dump(run/'comparison_verification.json',dict(passed=True,gpu_images=16,cpu_images=3,
        metric_rows_recomputed=57,error_maps_verified=57,checks=verified,app_model_replaced=False))
    print('PASS: 16 GPU + 3 CPU held-out comparisons, 57 metric rows, 57 error maps and timing totals.')


if __name__=='__main__':main()
