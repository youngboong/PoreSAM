"""Check preservation, exclusive ownership, replacement and browser review using cached SAM."""
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from playwright.sync_api import sync_playwright
from pore_editor import Editor,ROOT,make_handler
from pore_overlap import exclusive_existing,overlap_pixels,review_groups
from segment_first_pass import candidates_from_masks


def box(shape,x0,y0,x1,y1):
    mask=np.zeros(shape,bool);mask[y0:y1,x0:x1]=True;return mask


def main():
    run=ROOT/'outputs/editor_checks'/('exclusive_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'edits',ROOT/'outputs/projects');editor.large_search.root=ROOT/'outputs/large_pore_search'
    shape=(80,90);a=box(shape,5,5,25,25);b=box(shape,20,10,40,30)
    clean,info=exclusive_existing({1:a,2:b},{'1':{'review_status':'user_accepted'}})
    assert np.array_equal(clean[1],a) and overlap_pixels(clean.values())==0
    assert np.array_equal(clean[1]|clean[2],a|b)
    assert np.array_equal(clean[2]&~a,b&~a)
    fake=dict(dataset='fixture',gray=np.full(shape,120,np.uint8),report={'scale':{'um_per_pixel':1}},masks={1:a,2:b},annotations={},revision=0,history=[],next_id=3,preview=None)
    partial=box(shape,22,20,50,45)
    editor.preview_masks(fake,{'revision':0},[partial],[1.],'test')
    try:editor.mutate(fake,dict(revision=0,token=fake['preview']['token'],choice=0),'apply');raise AssertionError('Overlap accepted')
    except ValueError:pass
    assert fake['revision']==0 and not (run/'edits/fixture').exists()
    # Test exactly the masks the automatic pipeline exports, including touching pores.
    gray=np.full(shape,200,np.uint8);gray[a|b]=40
    raw=[dict(segmentation=m,predicted_iou=.99) for m in [a,b]]
    selected=candidates_from_masks(raw,gray,min_contrast=0,min_area=10)
    assert len(selected)==2 and overlap_pixels([r['mask'] for r in selected])==0
    entry=next(e for e in editor.image_library() if e['name']=='GF_1kx_1_BSE.tif')
    dataset=entry['latest_dataset'];state=editor.state(dataset)
    before={i:m.copy() for i,m in state['masks'].items()};key=editor.large_search.key(state['gray'])
    raw=editor.large_search.masks(state,key);groups,review=review_groups(raw,before,state['gray'])
    assert groups and overlap_pixels(list(before.values())+[g['mask'] for g in groups])==0
    assert any(len(g['alternatives'])>1 for g in groups)
    # A contained existing pore remains protected unless replacement review is requested.
    for g in groups:assert not g['replacement_ids']
    replacement_groups,_=review_groups(raw,before,state['gray'],True)
    assert any(g['replacement_ids'] for g in replacement_groups)
    user_files=list((ROOT/'outputs/manual_edits').glob('*/latest.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in user_files}
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Users\YoungJin\AppData\Local\Microsoft\Edge\Application\msedge.exe' if Path(r'C:\Users\YoungJin\AppData\Local\Microsoft\Edge\Application\msedge.exe').exists() else r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1500,height=1050));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('state!==null && !busy')
            page.evaluate("async d=>{await accept(await api('load',{dataset:d}));showPanel('editor')}",dataset)
            page.locator('#largeSearch').click();page.wait_for_function('!busy && !document.getElementById("largeReview").classList.contains("hidden")',timeout=20000)
            assert page.locator('#largeCandidate option').count()==len(groups)+1
            assert page.locator('#largeOutline').is_checked()
            page.screenshot(path=str(run/'combined_preview.png'),full_page=True)
            page.locator('#largeCandidate').select_option(str(groups[0]['id']));page.wait_for_function('!busy && preview!==null')
            assert not page.locator('#apply').is_disabled()
            page.locator('#apply').click();page.wait_for_function('!busy && state.revision===1',timeout=60000)
            assert overlap_pixels(state['masks'].values())==0
            assert all(np.array_equal(state['masks'][i],m) for i,m in before.items())
            assert page.locator('#largeReview').is_hidden()
            summary=json.loads((run/'edits'/dataset/'revision_0001/measurements/summary.json').read_text())
            assert summary['candidate_count']==len(before)+1
            page.locator('#status').click();page.keyboard.press('Control+z');page.wait_for_function('!busy && state.revision===2',timeout=60000)
            assert set(before)==set(state['masks']) and all(np.array_equal(state['masks'][i],m) for i,m in before.items())
            page.locator('#largeSearch').click();page.wait_for_function('!busy',timeout=20000)
            page.locator('#largeReplace').check();page.wait_for_function('!busy')
            replacement=next(g for g in replacement_groups if g['replacement_ids'])
            page.locator('#largeCandidate').select_option(str(replacement['id']));page.wait_for_function('!busy && preview!==null')
            page.locator('#apply').click();page.wait_for_function('!busy && state.revision===3',timeout=60000)
            assert overlap_pixels(state['masks'].values())==0
            assert all(np.array_equal(state['masks'][i],m) for i,m in before.items() if i not in replacement['replacement_ids'])
            page.locator('#undo').click();page.wait_for_function('!busy && state.revision===4',timeout=60000)
            # Old PI files may contain boundary overlap: cleanup saves a new revision, undo is exact.
            pi='PI35_2kx-4_BSE8';legacy=editor.state(pi);old={i:m.copy() for i,m in legacy['masks'].items()}
            assert overlap_pixels(old.values())>0
            page.evaluate("async d=>{await accept(await api('load',{dataset:d}))}",pi)
            assert page.locator('#legacyOverlap').is_visible()
            page.locator('#resolveOverlaps').click();page.wait_for_function('!busy && state.revision===1',timeout=60000)
            assert overlap_pixels(legacy['masks'].values())==0
            assert np.array_equal(np.logical_or.reduce(list(old.values())),np.logical_or.reduce(list(legacy['masks'].values())))
            page.locator('#undo').click();page.wait_for_function('!busy && state.revision===2',timeout=60000)
            assert all(np.array_equal(legacy['masks'][i],m) for i,m in old.items())
            assert not errors,errors
            browser.close()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        result=dict(passed=True,dataset=dataset,existing_count=len(before),raw_count=len(raw),addition_groups=len(groups),combined_overlap_pixels=0,errors=errors)
        (run/'result.json').write_text(json.dumps(result,indent=2));print(run,flush=True)
    finally:server.shutdown()


if __name__=='__main__':main()
