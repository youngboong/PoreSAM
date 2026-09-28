"""Regenerate filled comparisons from saved masks without running selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

from evaluate_nested import ROOT, dump, plot, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Completed nested evaluation directory')
    args=parser.parse_args();run=args.run.resolve()
    plan=json.loads((run/'plan.json').read_text())
    source=Path(plan['source_run'])
    training=json.loads((source/'plan.json').read_text())
    data=Path(training['dataset'])
    records=json.loads((data/'manifest.json').read_text())['records']
    folders=[run/f'fold_{fold}' for fold in plan['folds']]
    results=[json.loads((folder/'result.json').read_text()) for folder in folders]
    protected=[run/'plan.json',run/'summary.json',run/'metrics.csv']
    protected += [p for folder in folders for p in folder.iterdir() if p.suffix in ('.tif','.json')]
    before={str(p.relative_to(run)):sha(p) for p in protected}
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    for folder,result in zip(folders,results):
        record=next(r for r in records if r['name']==result['image'])
        for path,digest in result['input_sha256'].items():
            if sha(Path(path)) != digest:
                raise ValueError(f'Saved evaluation input changed: {path}')
        gray=np.array(Image.open(data/record['image']).convert('L'))
        truth=np.array(Image.open(data/record['labels']))
        maps={v:np.array(Image.open(source/'methods_gpu'/folder.name/f'{v}_instances.tif')) for v in ('relaxed','two_stage')}
        maps.update({v:np.array(Image.open(folder/f'{v}_instances.tif')) for v in ('relaxed_nested','two_stage_nested')})
        plot(folder,gray,truth,maps,result['metrics'])
        print(f'Rendered {folder.name}: {result["image"]}',flush=True)
    after={str(p.relative_to(run)):sha(p) for p in protected}
    if before != after:
        raise AssertionError('Rendering changed masks, metrics, decisions or original plan.')
    # Preserve the original execution snapshots; record the later rendering code separately.
    snapshot=run/'source/filled_overlay_rendering';snapshot.mkdir(exist_ok=True)
    for name in ('render_nested.py','evaluate_nested.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    dump(run/'rendering.json',dict(rendered_at_utc=datetime.now(timezone.utc).isoformat(),
         completed_images=len(results),mask_fill=dict(rgb=[255,215,0],alpha=.4),
         unchanged_artifact_sha256=after,
         renderer_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},
         selection_rerun=False,metrics_recomputed=False))
    page=(run/'index.html').read_text(encoding='utf-8')
    page=re.sub(r'<p id="rendering-note">.*?</p>','',page,flags=re.DOTALL)
    note=f'<p id="rendering-note">{len(results)}장 모두 분석 완료. 노란색 반투명 채움은 실제 pore 마스크 영역입니다. 저장된 마스크로 비교 그림만 다시 생성했으며 분할 결과와 수치는 동일합니다.</p>'
    page=page.replace('</h1>','</h1>'+note,1)
    (run/'index.html').write_text(page,encoding='utf-8')
    print(f'COMPLETE: {len(results)} images; masks and metrics unchanged.',flush=True)


if __name__=='__main__':
    main()
