"""Browser + real SAM integration check; revisions are isolated from user edits."""
import hashlib
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from pore_editor import Editor, make_handler, ROOT
from result_paths import read_artifact


def main():
    run=ROOT/"outputs/editor_checks"/time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    original=read_artifact(ROOT/"outputs/PI35_5kx-4_bse_first_pass", "entrance_candidates.npz")
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    editor=Editor(run/"edits")
    server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(editor))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f"http://127.0.0.1:{server.server_port}"
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",headless=True)
            page=browser.new_page(viewport={"width":1480,"height":1040})
            page.on("pageerror",lambda error:(errors.append(str(error)),print('BROWSER ERROR:',error,flush=True)))
            page.goto(url)
            page.wait_for_function("document.getElementById('count').textContent === '76'")
            page.locator('[data-panel="editor"]').click()
            page.locator("#zoom").evaluate("e => e.value = '150'")
            page.locator("#zoom").dispatch_event("input")
            # Box at 1.5x zoom to verify CSS-to-image coordinate conversion.
            rect=page.locator("#image").bounding_box()
            page.mouse.move(rect["x"]+405*1.5,rect["y"]+95*1.5)
            page.mouse.down()
            page.mouse.move(rect["x"]+620*1.5,rect["y"]+305*1.5,steps=8)
            page.mouse.up()
            box=page.evaluate("box")
            assert np.allclose(box,[405,95,620,305],atol=1),box
            page.locator('[data-mode="positive"]').click()
            page.mouse.click(rect["x"]+520*1.5,rect["y"]+200*1.5)
            page.locator('[data-mode="negative"]').click()
            page.mouse.click(rect["x"]+640*1.5,rect["y"]+180*1.5)
            assert page.evaluate("labels")==[1,0]
            # Ctrl+Z removes mixed prompt inputs chronologically without saving a revision.
            page.keyboard.press('Control+z')
            assert page.evaluate('labels')==[1]
            assert np.allclose(page.evaluate('box'),box)
            assert page.evaluate('state.revision')==0
            page.mouse.click(rect["x"]+640*1.5,rect["y"]+180*1.5)
            page.locator("#predict").click()
            page.wait_for_function("!document.getElementById('apply').disabled",timeout=120000)
            assert page.locator("#choices button").count()>=1
            page.locator("#zoom").evaluate("e => e.value = '75'")
            page.locator("#zoom").dispatch_event("input")
            page.screenshot(path=str(run/"preview.png"),full_page=True)
            page.locator("#target").select_option("16")
            page.locator("#apply").click()
            page.wait_for_function("document.getElementById('revision').textContent.includes('수정본 1 ')",timeout=120000)
            assert page.locator("#count").inner_text()=="76"
            state=editor.state("PI35_5kx-4_bse")
            assert state["annotations"]["16"]["source"]=="prompted_sam"
            assert state["revision"]==1
            assert not (run/"edits/PI35_5kx-4_bse/revision_0001/measurements").exists()
            editor.generate_report(state,dict(revision=1))
            assert (run/"edits/PI35_5kx-4_bse/revision_0001/measurements/dashboard.pdf").is_file()
            revision_folder=run/"edits/PI35_5kx-4_bse/revision_0001"
            for filename in ["original.png","entrance_candidates_overlay.png","comparison.png","union_mask.png"]:
                assert (revision_folder/"images"/filename).is_file()
            html=(revision_folder/"measurements/index.html").read_text(encoding="utf-8")
            assert '../images/entrance_candidates_overlay.png' in html
            assert page.request.get(url+"/files/PI35_5kx-4_bse/revision_0001/images/comparison.png").status==200
            # Reload a separate engine to verify the published revision is durable.
            reloaded=Editor(run/"edits").state("PI35_5kx-4_bse")
            assert np.array_equal(reloaded["masks"][16],state["masks"][16])
            try:
                editor.mutate(state,dict(revision=0,target_id=16),"delete")
                raise AssertionError("stale revision accepted")
            except ValueError:
                pass
            # Form inputs retain native undo; the canvas shortcut restores a saved mask.
            page.locator('#target').focus()
            page.keyboard.press('Control+z')
            assert page.evaluate('state.revision')==1
            page.locator('#undo').focus()
            page.keyboard.press('Control+z')
            page.wait_for_function("document.getElementById('revision').textContent.includes('수정본 2 ')",timeout=120000)
            with np.load(original) as data:
                assert np.array_equal(state["masks"][16],data["candidate_16"])
            # Add a small manually drawn region on unmasked background, then delete it.
            union=np.logical_or.reduce(list(state["masks"].values()))
            dist=cv2.distanceTransform((~union).astype(np.uint8),cv2.DIST_L2,3)
            dist[:16]=0
            dist[-16:]=0
            dist[:,:16]=0
            dist[:,-16:]=0
            y,x=np.unravel_index(dist.argmax(),dist.shape)
            assert dist[y,x]>5
            page.locator('[data-mode="polygon"]').click()
            rect=page.locator("#image").bounding_box()
            for px,py in [(x-3,y-3),(x+3,y-3),(x+3,y+3),(x-3,y+3)]:
                page.mouse.click(rect["x"]+px*.75,rect["y"]+py*.75)
            assert len(page.evaluate("polygon"))==4
            page.locator("#predict").click()
            page.wait_for_function("!document.getElementById('apply').disabled")
            page.locator("#apply").click()
            page.wait_for_function("document.getElementById('count').textContent === '77'",timeout=120000)
            added=max(state["masks"])
            assert state["annotations"][str(added)]["source"]=="manual_polygon"
            # A larger new entrance replaces the contained candidate after preview.
            small=state["masks"][added].copy()
            for px,py in [(x-5,y-5),(x+5,y-5),(x+5,y+5),(x-5,y+5)]:
                page.mouse.click(rect["x"]+px*.75,rect["y"]+py*.75)
            page.locator("#predict").click()
            page.wait_for_function("!document.getElementById('apply').disabled")
            assert page.evaluate("preview.choices[choice].contained_ids")==[added]
            assert str(added)+'번' in page.locator('#overlap').inner_text()
            assert page.locator('#apply').inner_text()=='기존 1개 대체 및 저장'
            assert state['revision']==3 and added in state['masks']
            page.screenshot(path=str(run/'containment_preview.png'),full_page=True)
            page.locator('#apply').click()
            page.wait_for_function("state.revision === 4 && !busy",timeout=120000)
            assert added not in state['masks'] and len(state['masks'])==77
            larger=max(state['masks'])
            assert state['masks'][larger].sum()>small.sum()
            folder=editor.revision_folder(state['dataset'],4)
            meta=json.loads((folder/'edit.json').read_text(encoding='utf-8'))
            assert meta['action']['replaced_candidate_ids']==[added]
            editor.generate_report(state,dict(revision=state['revision']))
            summary=json.loads((folder/'measurements/summary.json').read_text(encoding='utf-8'))
            assert summary['candidate_count']==77
            expected_union=np.logical_or.reduce(list(state['masks'].values())).sum()*state['report']['scale']['um_per_pixel']**2
            assert np.isclose(summary['union_candidate_area_um2'],expected_union)
            assert str(added) not in meta['annotations']
            with np.load(folder/'entrance_candidates.npz') as masks:
                assert f'candidate_{added}' not in masks.files
            reloaded=Editor(run/'edits').state(state['dataset'])
            assert added not in reloaded['masks'] and larger in reloaded['masks']
            page.locator('#undo').click()
            page.wait_for_function("state.revision === 5 && !busy",timeout=120000)
            assert np.array_equal(state['masks'][added],small)
            assert larger not in state['masks']
            page.locator("#target").select_option(str(added))
            page.locator("#delete").click()
            page.wait_for_function("document.getElementById('revision').textContent.includes('수정본 6 ')",timeout=120000)
            assert page.locator("#count").inner_text()=="76"
            page.screenshot(path=str(run/"saved.png"),full_page=True)
            assert not errors,errors
            assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
            # Local server refuses foreign origins and path traversal.
            denied=page.request.post(url+"/api/load",headers={"X-Pore-Editor":"1","Origin":"http://foreign.invalid"},data={"dataset":"PI35_5kx-4_bse"})
            assert denied.status==403
            # One oval tool supports both circles and ellipses at any zoom/direction.
            assert page.locator('[data-mode="circle"]').count()==0
            page.locator('[data-mode="ellipse"]').click()
            rect=page.locator('#image').bounding_box()
            def drag_region(start,end):
                page.mouse.move(rect['x']+start[0]*.75,rect['y']+start[1]*.75)
                page.mouse.down()
                page.mouse.move(rect['x']+end[0]*.75,rect['y']+end[1]*.75,steps=5)
                page.mouse.up()
            drag_region((100,100),(140,140))
            circle=page.evaluate('shape')
            assert np.allclose([circle['cx'],circle['cy'],circle['rx'],circle['ry']],[120,120,20,20],atol=1)
            page.locator('#back').click()
            assert page.evaluate('shape') is None
            drag_region((620,305),(405,95))
            ellipse=page.evaluate('shape')
            assert np.allclose([ellipse['cx'],ellipse['cy'],ellipse['rx'],ellipse['ry']],[512.5,200,107.5,105],atol=1)
            page.locator('[data-mode="positive"]').click()
            page.mouse.click(rect['x']+520*.75,rect['y']+200*.75)
            assert page.evaluate('shape')==ellipse
            page.locator('#predict').click()
            page.wait_for_function("!document.getElementById('apply').disabled",timeout=120000)
            assert state['preview']['prompts']['shape']==ellipse
            effective=state['preview']['prompts']['effective_sam_prompts']
            assert np.allclose(effective['box'],[405,95,620,305],atol=1)
            assert effective['labels']==[1,0,0,0,0]
            expected=state['preview']['masks'][page.evaluate('choice')].copy()
            page.screenshot(path=str(run/'ellipse_preview.png'),full_page=True)
            page.locator('#target').select_option('16')
            page.locator('#apply').click()
            page.wait_for_function('state.revision === 7 && !busy',timeout=120000)
            assert np.array_equal(state['masks'][16],expected)
            assert state['annotations']['16']['prompts']['shape']==ellipse
            assert not (editor.revision_folder(state['dataset'],7)/'measurements').exists()
            editor.generate_report(state,dict(revision=7))
            assert (editor.revision_folder(state['dataset'],7)/'measurements/summary.json').is_file()
            page.locator('#undo').click()
            page.wait_for_function('state.revision === 8 && !busy',timeout=120000)
            with np.load(original) as data:
                assert set(state['masks'])=={int(k.split('_')[-1]) for k in data.files}
                assert all(np.array_equal(state['masks'][int(k.split('_')[-1])],data[k]) for k in data.files)
            bad=page.request.post(url+'/api/predict',headers={'X-Pore-Editor':'1'},data=dict(dataset=state['dataset'],revision=8,shape=dict(kind='ellipse',cx=100,cy=100,rx=0,ry=10)))
            assert bad.status==400
            assert not errors,errors
            browser.close()
        result=dict(status="passed",checks=["real SAM box prediction","zoom coordinate conversion","replace + report export","durable reload","stale revision rejection","undo restores mask","polygon add","containment preview + replacement + measurement export + reload + undo","delete","original preserved","foreign origin blocked","single ellipse tool: circular/reverse drag, +/- refinement, real SAM, save, undo, degenerate rejection"],screenshots=str(run))
        (run/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result),flush=True)
    finally:
        server.shutdown()


if __name__=="__main__":
    main()
