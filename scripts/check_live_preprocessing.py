"""Isolated checks for live preview ordering, real SAM input and pore splitting."""
import base64
import io
import json
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pore_editor import Editor, ROOT, make_handler, save_images
from pore_preprocessing import prepare_image, adjustable_config, kuwahara
from analyze_candidates import export_folder


def main():
    run=ROOT/'outputs/ui_checks'/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects')
    assert not editor.datasets()['datasets']
    rng=np.random.default_rng(9)
    sample=rng.integers(0,256,(9,11),dtype=np.uint8)
    padded=np.pad(sample,2,mode='symmetric').astype(float)
    expected=np.zeros_like(sample)
    for y in range(9):
        for x in range(11):
            blocks=[padded[y+dy:y+dy+3,x+dx:x+dx+3] for dy,dx in [(0,0),(0,2),(2,0),(2,2)]]
            expected[y,x]=round(min(blocks,key=lambda a:a.var()).mean())
    assert np.array_equal(kuwahara(sample,2),expected)
    for method in ['none','gaussian','box','median','bilateral','kuwahara']:
        off=adjustable_config(dict(normalize_enabled=False,blur_method=method,blur_strength=0))
        assert np.array_equal(prepare_image(sample,off)[0],sample)
        on=adjustable_config(dict(normalize_enabled=True,background_strength=3,blur_method=method,blur_strength=3))
        const=np.full((64,64),127,np.uint8)
        assert np.array_equal(prepare_image(const,on)[0],const)
    for value in [-1,31,True,float('nan')]:
        try:adjustable_config(dict(background_strength=value));raise AssertionError('Invalid strength accepted')
        except ValueError:pass
    # A delayed old response must never replace the latest preview.
    preview=editor.workflow.preview_preprocessing
    def delayed(project_id, config):
        if config.get('background_strength')==11:time.sleep(.4)
        return preview(project_id,config)
    editor.workflow.preview_preprocessing=delayed
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as pw:
            edge=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
            browser=pw.chromium.launch(executable_path=str(edge),headless=True)
            page=browser.new_page(viewport=dict(width=1600,height=1000))
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('!busy && typeof imageLibrary!=="undefined"')
            assert page.locator('#largeSearch').count()==0
            page.locator('#uploadFile').set_input_files(str(ROOT/'data/GF_1kx_1_BSE.tif'))
            page.wait_for_function('setupProject && !busy && document.getElementById("preprocessingImage").dataset.generation===String(previewGeneration)')
            project=page.evaluate('setupProject')
            gray=np.asarray(Image.open(editor.workflow.root/project['id']/'input/normalized.png'))[:project['config']['analysis_bottom']]
            for method in ['gaussian','box','median','bilateral','kuwahara','none']:
                page.locator('#blurMethod').select_option(method)
                page.wait_for_function('!previewInFlight && document.getElementById("preprocessingImage").dataset.generation===String(previewGeneration)')
                assert page.locator('#blurStrengthField').is_hidden()==(method=='none')
                payload=page.evaluate('preprocessingPayload()')
                url=page.locator('#preprocessingImage').get_attribute('src')
                shown=np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))))
                assert np.array_equal(shown,prepare_image(gray,adjustable_config(payload))[0])
            page.evaluate("$('backgroundStrength').value=11;$('backgroundStrength').dispatchEvent(new Event('input'))")
            page.wait_for_function('previewInFlight')
            page.evaluate("$('backgroundStrength').value=7;$('backgroundStrength').dispatchEvent(new Event('input'));$('blurMethod').value='kuwahara';$('blurMethod').dispatchEvent(new Event('input'));$('blurStrength').value=3;$('blurStrength').dispatchEvent(new Event('input'))")
            page.wait_for_function('!previewInFlight && document.getElementById("preprocessingImage").dataset.generation===String(previewGeneration)')
            expected_input=prepare_image(gray,adjustable_config(page.evaluate('preprocessingPayload()')))[0]
            shown=np.asarray(Image.open(io.BytesIO(base64.b64decode(page.locator('#preprocessingImage').get_attribute('src').split(',')[1]))))
            assert np.array_equal(shown,expected_input)
            page.locator('#newAnalysisSettings').scroll_into_view_if_needed()
            page.screenshot(path=str(run/'live_preview.png'))
            # Actual automatic SAM must consume precisely the previewed pixels.
            page.locator('#scaleUm').fill('50');page.locator('#scalePixels').fill('275')
            page.locator('#scaleConfirmed').check()
            page.locator('#advancedSettings summary').click();page.locator('#pointDensity').select_option('16')
            page.locator('#runAnalysis').click()
            page.wait_for_function('state!==null && !busy',timeout=240000)
            sam_state=editor.state(page.evaluate('state.dataset'))
            assert np.array_equal(sam_state['sam_gray'],expected_input)
            assert sam_state['report']['preprocessing_config']['blur_method']=='kuwahara'
            print('Real SAM and preview pixel equality passed',flush=True)
            # Deterministic connected pore: splitting and undo must change statistics exactly.
            scene=np.full((200,200),200,np.uint8);scene[30:170,30:170]=40
            buf=io.BytesIO();Image.fromarray(scene).save(buf,format='PNG')
            fixture=editor.workflow.upload('cut-fixture.png',buf.getvalue())
            folder=editor.workflow.root/fixture['id']/'runs/run_0001';folder.mkdir(parents=True)
            source=editor.workflow.root/fixture['id']/'input/normalized.png'
            mask=scene==40
            report=dict(image=str(source),analysis_bottom_exclusive=200,entrance_candidate_count=1,scale=dict(um_per_pixel=.5,label_um=50,length_pixels=100),overlay_relative_path='images/entrance_candidates_overlay.png')
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
            np.savez_compressed(folder/'entrance_candidates.npz',candidate_1=mask)
            save_images(folder,scene,{1:mask});export_folder(folder)
            dataset=fixture['id']+'__run_0001'
            editor.workflow.projects[fixture['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=fixture['config'])]
            page.evaluate("async d=>{await accept(await api('load',{dataset:d}));showPanel('editor')}",dataset)
            page.locator('[data-mode="cut"]').click()
            page.locator('#zoom').fill('150');page.locator('#zoom').dispatch_event('input')
            rect=page.locator('#image').bounding_box()
            page.mouse.move(rect['x']+150,rect['y']+15)
            page.mouse.down();page.mouse.move(rect['x']+150,rect['y']+285,steps=30);page.mouse.up()
            page.screenshot(path=str(run/'cut_path.png'))
            page.locator('#applyCut').click();page.wait_for_function('!busy && state.revision===1',timeout=60000)
            state=editor.state(dataset)
            assert len(state['masks'])==2 and all(m.any() for m in state['masks'].values())
            assert not (state['masks'][1]&state['masks'][2]).any()
            remaining=sum(int(m.sum()) for m in state['masks'].values())
            assert remaining==int(mask.sum())-3*140
            editor.generate_report(editor.state(page.evaluate('state.dataset')),dict(revision=page.evaluate('state.revision')))
            summary=json.loads((run/'edits'/dataset/'revision_0001/measurements/summary.json').read_text())
            assert summary['candidate_count']==2 and summary['union_candidate_area_um2']==remaining*.25
            page.screenshot(path=str(run/'cut_result.png'))
            page.locator('#status').click();page.keyboard.press('Control+z')
            page.wait_for_function('!busy && state.revision===2',timeout=60000)
            assert list(state['masks'])==[1] and np.array_equal(state['masks'][1],mask)
            try:editor.mutate(state,dict(revision=1,path=[[100,10],[100,190]],width=3),'cut');raise AssertionError('Stale cut accepted')
            except ValueError:pass
            assert not errors,errors
            page.wait_for_load_state('networkidle')
            browser.close()
        (run/'result.json').write_text(json.dumps(dict(passed=True,browser_errors=errors,real_sam_preview_equal=True,cut_count=2,undo_exact=True),indent=2))
        print(run,flush=True)
    finally:server.shutdown()


if __name__=='__main__':main()
