"""Real GPU discovery and browser editing; all revisions isolated from user data."""
import hashlib
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
import cv2
from playwright.sync_api import sync_playwright
from pore_editor import Editor, ROOT, make_handler
from result_paths import read_artifact


def main():
    run=ROOT/'outputs/editor_checks'/('large_'+time.strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True)
    editor=Editor(run/'edits',ROOT/'outputs/projects')
    dataset='image_dffc9b43ff372f12__run_0004'
    state=editor.state(dataset)
    original=read_artifact(state['baseline'],'entrance_candidates.npz')
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    refs_path=ROOT/'outputs/manual_edits/image_dffc9b43ff372f12__run_0002/revision_0002/entrance_candidates.npz'
    refs_digest=hashlib.sha256(refs_path.read_bytes()).hexdigest()
    errors=[]
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1500,height=1000))
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('state !== null && !busy')
            page.evaluate("async d=>{await accept(await api('load',{dataset:d}));showPanel('editor')}",dataset)
            page.locator('#largeSearch').click()
            deadline=time.time()+420
            while page.evaluate('busy'):
                print(page.locator('#largeStatus').inner_text(),flush=True)
                assert time.time()<deadline,'Discovery timed out'
                page.wait_for_timeout(10000)
            assert page.locator('#largeReview').is_visible(),page.locator('#status').inner_text()
            key=editor.large_search.key(state['gray'])
            masks=editor.large_search.masks(state,key)
            assert len(masks)==39,len(masks)
            assert page.locator('#largeStrength').input_value()=='medium'
            assert editor.large_search.key(state['gray'],'strong')!=key
            # References are read only AFTER inference and used solely for evaluation.
            with np.load(refs_path) as saved:refs=[saved[f'candidate_{i}'] for i in [30,31]]
            matches=[]
            for ref in refs:
                ious=[float((m&ref).sum()/(m|ref).sum()) for m in masks]
                idx=int(np.argmax(ious));matches.append(dict(id=idx+1,iou=ious[idx]))
                assert ious[idx]>.8,matches
            assert matches[0]['iou']>.94 and matches[1]['iou']>.84,matches
            print('MATCHES',matches,flush=True)
            # Click a displayed exclusive group rather than a hidden raw alternative.
            group_id=int(page.locator('#largeCandidate option').nth(1).get_attribute('value'))
            y,x=np.unravel_index(cv2.distanceTransform(masks[group_id-1].astype(np.uint8),cv2.DIST_L2,3).argmax(),masks[0].shape)
            page.locator('#zoom').evaluate("e=>e.value='75'")
            page.locator('#zoom').dispatch_event('input')
            rect=page.locator('#image').bounding_box()
            page.mouse.click(rect['x']+int(x)*.75,rect['y']+int(y)*.75)
            page.wait_for_function('!busy && preview !== null')
            assert page.evaluate('preview.choices.every(c=>c.pool_candidate_id>0)')
            page.locator('#largeCandidate').select_option(str(group_id))
            page.wait_for_function('!busy && preview !== null')
            assert len(state['masks'])==state['report']['entrance_candidate_count']
            count=len(state['masks'])
            page.locator('#apply').click()
            page.wait_for_function('!busy && state.revision===1',timeout=60000)
            assert (run/'edits'/dataset/'revision_0001/measurements/summary.json').is_file()
            assert (run/'edits'/dataset/'revision_0001/images/comparison.png').is_file()
            page.locator('#status').click()
            page.keyboard.press('Control+z')
            page.wait_for_function('!busy && state.revision===2',timeout=60000)
            assert len(state['masks'])==count
            # Reuse cache and reject keys for a different image or stale revisions.
            before=time.time();page.locator('#largeSearch').click()
            page.wait_for_function('!busy',timeout=15000)
            assert time.time()-before<15
            for data in [dict(dataset=dataset,revision=0,key=key),dict(dataset=dataset,revision=2,key='wrong')]:
                response=page.request.post(f'http://127.0.0.1:{server.server_port}/api/large-overview',data=data,headers={'X-Pore-Editor':'1'})
                assert response.status==400
            page.locator('[data-mode="ellipse"]').click()
            assert 'active' not in (page.locator('#largePick').get_attribute('class') or '')
            page.locator('#largeCandidate').select_option(str(group_id))
            page.wait_for_function('!busy && preview !== null')
            page.locator('#largeStrength').select_option('strong')
            assert page.evaluate('preview === null')
            assert not page.locator('#largeReview').is_visible()
            assert page.locator('#apply').is_disabled()
            response=page.request.post(f'http://127.0.0.1:{server.server_port}/api/large-search',data=dict(dataset=dataset,revision=2,strength='bad'),headers={'X-Pore-Editor':'1'})
            assert response.status==400 and not editor.workflow.running
            assert not errors,errors
            page.screenshot(path=str(run/'editor.png'),full_page=True)
            assert digest==hashlib.sha256(original.read_bytes()).hexdigest()
            assert refs_digest==hashlib.sha256(refs_path.read_bytes()).hexdigest()
            (run/'result.json').write_text(json.dumps(dict(passed=True,candidate_count=len(masks),reference_matches=matches,cache_key=key,errors=errors),indent=2))
            print(run,flush=True)
            browser.close()
    finally:server.shutdown()


if __name__=='__main__':main()
