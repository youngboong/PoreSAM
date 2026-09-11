"""Browser regression: two contained pores plus a partially overlapping neighbor."""
import io
import json
import threading
import time
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pore_editor import Editor,ROOT,make_handler,save_images
from analyze_candidates import export_folder
from test_editor_containment import rectangle


def main():
    run=ROOT/'outputs/ui_checks'/('replacement_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects')
    gray=np.full((80,80),180,np.uint8)
    masks={1:rectangle(10,10,5,5),2:rectangle(20,10,5,5),3:rectangle(28,10,10,10)}
    for mask in masks.values():gray[mask]=40
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('replacement-fixture.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=80,entrance_candidate_count=3,scale=dict(um_per_pixel=1,label_um=20,length_pixels=20),overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report));np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    save_images(folder,gray,masks);export_folder(folder)
    dataset=project['id']+'__run_0001'
    editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    state=editor.state(dataset)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1450,height=950));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#imageLibrary .image-card').click()
            page.wait_for_function('state && !busy');page.evaluate("showPanel('editor')")
            preview=editor.preview_masks(state,dict(revision=0),[rectangle(9,9,22,17)],[1.],'manual_polygon')
            page.evaluate('async data=>{await presentPreview(data)}',preview)
            assert page.locator('#apply').is_disabled()
            assert '2' in page.locator('#apply').inner_text()
            assert page.locator('#trimOverlap').is_visible()
            page.locator('#trimOverlap').click();page.wait_for_function('!busy && preview.trimmed_overlap_pixels===30')
            assert not page.locator('#apply').is_disabled()
            assert page.locator('#trimOverlap').is_hidden()
            assert state['revision']==0 and set(state['masks'])=={1,2,3}
            page.screenshot(path=str(run/'replacement_ready.png'))
            page.locator('#apply').click();page.wait_for_function('!busy && state.revision===1',timeout=60000)
            assert set(state['masks'])=={3,4}
            assert np.array_equal(state['masks'][3],masks[3])
            assert not (state['masks'][3]&state['masks'][4]).any()
            summary=json.loads((run/'edits'/dataset/'revision_0001/measurements/summary.json').read_text())
            assert summary['candidate_count']==2 and summary['overlap_pixels']==0
            page.wait_for_load_state('networkidle');assert not errors,errors
            browser.close()
        (run/'result.json').write_text(json.dumps(dict(passed=True,replaced_ids=[1,2],preserved_ids=[3],overlap_pixels=0,browser_errors=errors),indent=2))
        print(run,flush=True)
    finally:server.shutdown()


if __name__=='__main__':main()
