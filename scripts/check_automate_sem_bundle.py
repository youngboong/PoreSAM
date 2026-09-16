"""Run packaged Automate on an isolated copy of a saved SEM analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from check_desktop_windows import close


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--revision',type=int)
    parser.add_argument('--expect-merged',type=int,nargs='+')
    parser.add_argument('--compare-masks',type=Path,help='Require exact equality with a reference candidate NPZ')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=root/'outputs/desktop_checks'/('automate_sem_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    project_id=args.dataset.split('__run_')[0]
    source=args.data/'manual_edits'/args.dataset
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    original_latest=digest(source/'latest.json')
    shutil.copytree(args.data/'projects'/project_id,run/'data/projects'/project_id)
    shutil.copytree(source,run/'data/manual_edits'/args.dataset)
    if args.revision is not None:
        copied=run/'data/manual_edits'/args.dataset
        assert (copied/f'revision_{args.revision:04d}'/'entrance_candidates.npz').is_file()
        latest=json.loads((copied/'latest.json').read_text());latest['revision']=args.revision
        (copied/'latest.json').write_text(json.dumps(latest))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=os.environ.copy();env['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS']=f'--remote-debugging-port={port}'
    process=subprocess.Popen([str(args.bundle.resolve()/'PoreSAM.exe'),'--data-dir',str(run/'data')],env=env)
    errors=[]
    try:
      deadline=time.monotonic()+90
      while time.monotonic()<deadline:
        assert process.poll() is None
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version',timeout=1).close();break
        except OSError:time.sleep(.25)
      else:raise AssertionError('App did not start')
      with sync_playwright() as p:
        browser=p.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
        page=browser.contexts[0].pages[0];page.on('pageerror',lambda error:errors.append(str(error)))
        page.wait_for_function("typeof busy!=='undefined'&&!busy")
        page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click()
        page.wait_for_function('state&&!busy')
        before=page.evaluate('({revision:state.revision,count:state.candidates.length})')
        started=time.monotonic();page.locator('#automatePores').click()
        page.wait_for_function("!busy&&document.getElementById('status').textContent.includes('Automate complete')",timeout=240000)
        elapsed=time.monotonic()-started
        after=page.evaluate('({revision:state.revision,count:state.candidates.length,stats:state.stats})')
        assert after['revision']>before['revision'],page.locator('#status').inner_text()
        def masks_at(revision):
            with np.load(run/'data/manual_edits'/args.dataset/f'revision_{revision:04d}'/'entrance_candidates.npz') as archive:
                return {int(key.rsplit('_',1)[1]):archive[key] for key in archive.files}
        old=masks_at(before['revision']);new=masks_at(after['revision'])
        if args.compare_masks:
            with np.load(args.compare_masks) as archive:
                reference={int(k.rsplit('_',1)[1]):archive[k] for k in archive.files}
            assert set(new)==set(reference),'Candidate IDs differ from reference'
            assert all(np.array_equal(new[i],reference[i]) for i in new),'Candidate pixels differ from reference'
        old_union=np.logical_or.reduce(list(old.values()));new_union=np.logical_or.reduce(list(new.values()))
        gained=new_union&~old_union
        left_gain=int(gained[339:539,0:280].sum())
        if args.expect_merged:
            keep=min(args.expect_merged)
            assert keep in new and all(i not in new for i in args.expect_merged if i!=keep),sorted(new)
            metadata=json.loads((run/'data/manual_edits'/args.dataset/f"revision_{after['revision']:04d}"/'edit.json').read_text())
            assert set(args.expect_merged)<=set(metadata['annotations'][str(keep)]['merged_from'])
        else:assert left_gain>15000,left_gain
        assert after['stats']['overlap_pixels']==0
        assert page.evaluate('queuedRegions.length')==0,'Automate created manual review work'
        assert digest(source/'latest.json')==original_latest,'User data changed during isolated test'
        assert not errors,errors
        page.screenshot(path=str(run/'automate_result.png'))
        report=json.loads((run/'data/manual_edits'/args.dataset/f"revision_{after['revision']:04d}"/'report.json').read_text())
        gray=np.asarray(Image.open(report['image']).convert('L'))[:gained.shape[0]]
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        rgb[gained]=(rgb[gained]*.5+np.array([0,220,255])*.5).astype(np.uint8)
        Image.fromarray(rgb).save(run/'added_pores.png')
        if args.expect_merged:
            rgb=np.repeat(gray[:,:,None],3,axis=2);selected=new[min(args.expect_merged)]
            rgb[selected]=(rgb[selected]*.5+np.array([0,220,255])*.5).astype(np.uint8)
            Image.fromarray(rgb).save(run/'merged_pore.png')
        (run/'verification.json').write_text(json.dumps(dict(before=before,after=after,merged_ids=args.expect_merged,left_pore_added_pixels=left_gain,elapsed_seconds=elapsed,user_data_unchanged=True,reference_masks_identical=True if args.compare_masks else None,errors=errors),indent=2))
        # Review-only boxes are test drafts, not edits; clear them before closing.
        page.evaluate('window.clearRegionQueue()')
        close(process)
        print(run,dict(left_pore_added_pixels=left_gain,elapsed_seconds=elapsed),flush=True)
    finally:
        if process.poll() is None:close(process)


if __name__=='__main__':main()
