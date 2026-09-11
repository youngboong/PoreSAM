"""Isolated browser test of upload, calibration, real SAM, reuse and inline reports."""
import hashlib
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from pore_editor import Editor, ROOT, make_handler
from pore_preprocessing import preprocessing_config, prepare_image


def main():
    run=ROOT/'outputs/workspace_checks'/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}'
    original_source=ROOT/'data/PI35_5kx-4_bse.tif'
    # Same filename and pixels, different file bytes: it must not open the legacy analysis.
    source=run/original_source.name
    Image.open(original_source).save(source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1480,height=1120))
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url)
            page.wait_for_function('state !== null && !busy && imageLibrary.length === 3')
            assert page.title()=='Pore Editor'
            assert page.locator('#setupPanel').is_visible()
            assert not page.locator('#newAnalysisSettings').is_visible()
            assert page.locator('#imageLibrary .image-card').count()==3
            # Exact legacy source re-upload bypasses settings and does not create a project.
            page.locator('#uploadFile').set_input_files(str(original_source))
            page.wait_for_function('!busy && !document.getElementById("editorPanel").classList.contains("hidden")')
            assert page.evaluate('state.dataset')=='PI35_5kx-4_bse'
            assert not editor.workflow.projects
            page.locator('[data-panel="setup"]').click()
            page.locator('#uploadFile').set_input_files(str(source))
            page.wait_for_function('setupProject !== null && setupImage !== null && !busy',timeout=30000)
            project=page.evaluate('setupProject')
            assert page.evaluate('state') is None
            assert page.locator('#newAnalysisSettings').is_visible()
            assert page.locator('#pendingImages .image-card').count()==1
            assert project['width']==1024 and project['height']==946
            assert int(page.locator('#analysisBottom').input_value())==768
            assert not page.locator('#scaleConfirmed').is_checked()
            assert page.locator('#scaleUm').input_value()==''
            assert not page.locator('#advancedSettings').evaluate('(e)=>e.open')
            page.locator('#advancedSettings summary').click()
            assert page.locator('#normalizeEnabled').is_checked()
            assert not page.locator('#structureEnabled').is_checked()
            assert not page.locator('#coarseStrengthField').is_visible()
            # Preview works before calibration and does not persist analysis settings.
            initial_config=editor.workflow.inspect(project['id'])['config']
            page.locator('#structureEnabled').check()
            assert page.locator('#coarseStrengthField').is_visible()
            page.locator('#previewPreprocessing').click()
            page.wait_for_function('!busy && !document.getElementById("preprocessingImage").classList.contains("hidden")')
            assert page.locator('#preprocessingImage').evaluate('(img)=>img.naturalWidth')==1024
            assert editor.workflow.inspect(project['id'])['config']==initial_config
            assert not editor.workflow.jobs
            # Both switches are independent; all four combinations reach the server.
            for normalize,structure,mode in [(False,False,'none'),(True,False,'normalize'),(False,True,'structure'),(True,True,'coarse')]:
                page.locator('#normalizeEnabled').set_checked(normalize)
                page.locator('#structureEnabled').set_checked(structure)
                assert page.locator('#coarseStrengthField').is_visible()==structure
                payload=page.evaluate('preprocessingPayload()')
                assert payload['preprocessing_mode']==mode
                response=page.request.post(url+'/api/preprocess-preview',headers={'X-Pore-Editor':'1'},data=dict(project_id=project['id'],config=payload))
                assert response.status==200
                metadata=response.json()['metadata']
                assert ('percentiles' in metadata)==normalize
                assert ('radius_pixels' in metadata)==structure
            page.locator('#normalizeEnabled').uncheck()
            page.locator('#coarseStrength').select_option('weak')
            assert not page.locator('#preprocessingImage').is_visible()
            page.locator('#coarseStrength').select_option('medium')
            # Verify calibration clicks at CSS scaling.
            page.locator('#measureScale').click()
            page.locator('#setupCanvas').scroll_into_view_if_needed()
            rect=page.locator('#setupCanvas').bounding_box()
            factor=rect['width']/1024
            for x,y in [(765,808),(948,808)]:
                page.mouse.click(rect['x']+x*factor,rect['y']+y*factor)
            assert abs(float(page.locator('#scalePixels').input_value())-183)<1
            page.locator('#scaleUm').fill('10')
            page.locator('#pointDensity').select_option('16')
            page.locator('#minContrast').fill('-1')
            page.locator('#advancedSettings summary').click()
            page.locator('#runAnalysis').click()
            assert page.locator('#advancedSettings').evaluate('(e)=>e.open')
            assert page.evaluate('document.activeElement.id')=='minContrast'
            assert not editor.workflow.jobs
            page.locator('#minContrast').fill('8')
            page.locator('#advancedSettings summary').click()
            page.locator('#runAnalysis').click()
            page.wait_for_function('!busy')
            assert not editor.workflow.jobs  # unconfirmed scale rejected before work
            assert page.evaluate('document.activeElement.id')=='scaleConfirmed'
            page.locator('#scaleConfirmed').check()
            page.screenshot(path=str(run/'setup.png'),full_page=True)
            page.locator('#runAnalysis').click()
            page.wait_for_function('state?.dataset?.startsWith("image_") && !busy',timeout=240000)
            dataset=page.evaluate('state.dataset')
            assert page.locator('#analysisPanel').is_visible()
            assert page.locator('#analysisDownloads a').count()==6
            frame=page.frame_locator('#analysisFrame')
            frame.locator('h1').wait_for()
            assert page.request.get(url+page.locator('#analysisComparison').get_attribute('src')).status==200
            baseline=editor.baseline(dataset)
            report=json.loads((baseline/'report.json').read_text())
            editor.generate_report(editor.state(page.evaluate('state.dataset')),dict(revision=page.evaluate('state.revision')))
            summary=json.loads((baseline/'measurements/summary.json').read_text())
            assert report['selection_settings']==dict(min_contrast=8,min_area_pixels=100)
            assert report['settings']['points_per_side']==16
            assert report['preprocessing_config']==preprocessing_config('structure','medium')
            raw_gray=np.asarray(Image.open(source).convert('L'))[:768]
            expected,_=prepare_image(raw_gray,report['preprocessing_config'])
            assert np.array_equal(np.asarray(Image.open(report['sam_input_image'])),expected)
            assert np.array_equal(editor.state(dataset)['sam_gray'],expected)
            assert np.array_equal(editor.state(dataset)['gray'],raw_gray)
            assert summary['candidate_count']==report['entrance_candidate_count']
            with np.load(baseline/'masks/entrance_candidates.npz') as masks:
                assert len(masks.files)==summary['candidate_count']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
            page.screenshot(path=str(run/'analysis.png'),full_page=True)
            # Reusing the project must preserve its original result and persist a new threshold.
            first_hash=hashlib.sha256((baseline/'masks/entrance_candidates.npz').read_bytes()).hexdigest()
            page.locator('[data-panel="setup"]').click()
            page.locator('#advancedSettings summary').click()
            page.locator('#minContrast').fill('255')
            page.locator('#runAnalysis').click()
            page.wait_for_function('(old)=>state?.dataset !== old && !busy',arg=dataset,timeout=120000)
            second=page.evaluate('state.dataset')
            second_folder=editor.baseline(second)
            editor.generate_report(editor.state(page.evaluate('state.dataset')),dict(revision=page.evaluate('state.revision')))
            summary2=json.loads((second_folder/'measurements/summary.json').read_text())
            assert summary2['candidate_count']==0
            assert summary2['selection_settings']['min_contrast']==255
            assert hashlib.sha256((baseline/'masks/entrance_candidates.npz').read_bytes()).hexdigest()==first_hash
            assert json.loads((second_folder/'report.json').read_text())['raw_masks_reused_from']==str(baseline)
            # Restarted engine still lists both completed runs and loads an empty result.
            restored=Editor(run/'edits',run/'projects')
            assert dataset in restored.datasets()['datasets'] and second in restored.datasets()['datasets']
            assert restored.response(restored.state(second))['stats']['candidate_count']==0
            page.locator('#editFromAnalysis').click()
            assert page.locator('#editorPanel').is_visible()
            # Add a polygon to the empty analysis; the embedded report must follow the revision.
            page.locator('[data-mode="polygon"]').click()
            rect=page.locator('#image').bounding_box()
            for x,y in [(100,100),(120,100),(120,120),(100,120)]:
                page.mouse.click(rect['x']+x,rect['y']+y)
            page.locator('#predict').click()
            page.wait_for_function('!busy && preview !== null')
            page.locator('#apply').click()
            page.wait_for_function('state.revision === 1 && !busy',timeout=120000)
            assert page.evaluate('state.stats.candidate_count')==1
            page.locator('#report').click()
            assert page.locator('#analysisPanel').is_visible()
            assert '/revision_0001/' in page.locator('#analysisFrame').get_attribute('src')
            # Re-uploading the same bytes under a new name opens the latest saved edit.
            project_count=len(editor.workflow.projects);job_count=len(editor.workflow.jobs)
            page.locator('[data-panel="setup"]').click()
            page.locator('#uploadFile').set_input_files(dict(name='renamed.tif',mimeType='image/tiff',buffer=source.read_bytes()))
            page.wait_for_function('!busy && !document.getElementById("editorPanel").classList.contains("hidden")')
            assert page.evaluate('state.dataset')==second
            assert page.evaluate('state.revision')==1
            assert len(editor.workflow.projects)==project_count and len(editor.workflow.jobs)==job_count
            assert page.locator('#dataset option').count()==2
            page.locator('[data-panel="setup"]').click()
            page.locator(f'#imageLibrary [data-image-id="{digest}"]').click()
            page.wait_for_function('!busy && !document.getElementById("editorPanel").classList.contains("hidden")')
            assert page.evaluate('state.revision')==1
            # Explicit reanalysis keeps the same original image and analysis history.
            page.locator('#reuseImage').click()
            page.wait_for_function('!busy && setupProject !== null')
            assert page.evaluate('setupProject.id')==project['id']
            page.locator('#advancedSettings summary').click()
            page.locator('#scaleConfirmed').check()
            # Changing preprocessing must invalidate SAM cache, while older runs remain intact.
            page.locator('[data-panel="setup"]').click()
            page.locator('#structureEnabled').uncheck()
            page.locator('#normalizeEnabled').check()
            page.locator('#minContrast').fill('8')
            page.locator('#runAnalysis').click()
            page.wait_for_function('(old)=>state?.dataset !== old && !busy',arg=second,timeout=240000)
            third_folder=editor.baseline(page.evaluate('state.dataset'))
            third_report=json.loads((third_folder/'report.json').read_text())
            assert third_report['preprocessing_config']==preprocessing_config('normalize')
            assert third_report['raw_masks_reused_from'] is None
            assert hashlib.sha256((baseline/'masks/entrance_candidates.npz').read_bytes()).hexdigest()==first_hash
            # Importing legacy results keeps their original SAM input mode.
            seed=editor.workflow.upload(source.name,source.read_bytes(),ROOT/'outputs/PI35_5kx-4_bse_first_pass')
            assert seed['config']['preprocessing_mode']=='none'
            assert not errors,errors
            result=dict(status='passed',checks=['upload','preprocessing preview without calibration','preview invalidation','saved preprocessing and manual SAM input','preprocessing invalidates SAM cache','legacy input preserved','scaled calibration clicks','required scale confirmation','real SAM','inline reports','threshold reuse','zero candidates','immutable results','persistent projects','manual editing and report refresh'],run=str(run))
            (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result),flush=True)
            browser.close()
    finally:
        server.shutdown()


if __name__=='__main__': main()
