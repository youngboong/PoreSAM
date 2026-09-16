"""Exercise Stop controls and transactional cancellation on isolated project data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import io
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pore_editor import Editor, ROOT, make_handler, save_images


def main():
    run=ROOT/'outputs/ui_checks'/('stop_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    yy,xx=np.mgrid[:240,:240]
    masks=[(xx-x)**2+(yy-100)**2<20**2 for x in (45,120,195)]
    gray=np.full((240,240),190,np.uint8)
    for mask in masks:gray[mask]=30
    editor=Editor(run/'edits',run/'projects',device='cpu')
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('stop-test.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=240,
                entrance_candidate_count=1,scale=dict(um_per_pixel=.1,label_um=10,length_pixels=100),
                overlay_relative_path='images/entrance_candidates_overlay.png',selection_settings=dict(min_area_pixels=20,min_contrast=8),settings={})
    (folder/'report.json').write_text(json.dumps(report));save_images(folder,gray,{1:masks[0]})
    np.savez_compressed(folder/'entrance_candidates.npz',candidate_1=masks[0])
    dataset=project['id']+'__run_0001'
    editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    editor.workflow.persist(editor.workflow.projects[project['id']])
    entered=threading.Event();release=threading.Event();calls=[];hold=[True]
    def preview(state,payload,polygon=False):
        index=1 if payload['box'][0]<150 else 2;calls.append(index)
        if index==2 and hold[0]:entered.set();assert release.wait(15)
        state['preview']=dict(masks=[masks[index]])
        return dict(choices=[dict(score=.99)])
    editor.preview=preview
    boxes=dict(boxes=[dict(box=[90,70,150,130]),dict(box=[165,70,225,130])])
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(route,payload):
        request=urllib.request.Request(base+'/api/'+route,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Pore-Editor':'1'})
        with urllib.request.urlopen(request,timeout=10) as response:return json.load(response)
    errors=[]
    try:
      with patch('automate_pores.propose_boxes',return_value=boxes),sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport=dict(width=1600,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(base);page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
        baseline=page.evaluate('({revision:state.revision,stats:state.stats,ids:state.candidates.map(c=>c.candidate_id)})')
        page.locator('#automatePores').click();assert entered.wait(10)
        button=page.locator('#automatePores');assert button.inner_text()=='Stop' and button.is_enabled()
        page.wait_for_function("['rgb(220, 38, 38)','rgb(185, 28, 28)'].includes(getComputedStyle(document.getElementById('automatePores')).backgroundColor)")
        assert len(editor.state(dataset)['masks'])==1 # First discovered pore is still staged.
        button.click();page.wait_for_function("document.getElementById('automatePores').textContent==='Stopping…'")
        assert next(iter(editor.automate_stops.values()))['stop'].wait(2) # stop bypasses model lock
        page.screenshot(path=str(run/'automate_stopping.png'));release.set();page.wait_for_function('!busy')
        assert page.evaluate('({revision:state.revision,stats:state.stats,ids:state.candidates.map(c=>c.candidate_id)})')==baseline
        assert not list((run/'edits').glob('*/latest.json'))
        assert not editor.automate_stops and button.inner_text()=='Automate' and button.is_enabled()
        # A subsequent complete run publishes all discoveries in exactly one revision.
        hold[0]=False;button.click();page.wait_for_function('!busy && state.revision===1')
        assert len(editor.state(dataset)['masks'])==3
        assert len(list((run/'edits'/dataset).glob('revision_*')))==1
        assert editor.state(dataset)['history']==[0]
        # Cancellation during final file writing must not publish latest.json or state.
        current=editor.state(dataset);before=current['revision']
        payload=dict(dataset=dataset,revision=before)
        op=post('start-automate',payload);operation=editor.automate_stops[op['operation_id']]
        import copy
        working=copy.deepcopy(current);working['masks'][4]=(xx-120)**2+(yy-190)**2<10**2;working['next_id']=5
        operation['working']=working
        real_save=np.savez_compressed
        def stop_during_save(*args,**kwargs):
            result=real_save(*args,**kwargs);operation['stop'].set();return result
        with patch('pore_editor.np.savez_compressed',side_effect=stop_during_save):
            assert post('commit-automate',{**payload,**op})['cancelled']
        assert current['revision']==before and 4 not in current['masks']
        assert json.loads((run/'edits'/dataset/'latest.json').read_text())['revision']==before
        post('finish-automate',op)
        # Run Analysis stop uses the same button and never registers a partial run.
        model_entered=threading.Event();model_release=threading.Event()
        class Generator:
            def __init__(self,*args,**kwargs):pass
            def generate(self,*args):
                model_entered.set();assert model_release.wait(15)
                self.cancel_callback();return []
        with patch('project_workflow.build_sam2',return_value=object()),patch('project_workflow.ProgressGenerator',Generator):
            page.evaluate('async(id)=>{await showProject(await api("project",{project_id:id}));showPanel("setup");}',project['id'])
            page.locator('#scaleUm').fill('10');page.locator('#scalePixels').fill('100');page.locator('#scaleConfirmed').check()
            page.locator('#runAnalysis').click();assert model_entered.wait(10)
            button=page.locator('#runAnalysis');assert button.inner_text()=='Stop' and button.is_enabled()
            button.click();page.wait_for_function("document.getElementById('runAnalysis').textContent==='Stopping…'")
            job_id=page.evaluate("localStorage.getItem('poreActiveJob')")
            deadline=time.monotonic()+3
            while editor.workflow.status(job_id)['status']!='stopping' and time.monotonic()<deadline:time.sleep(.01)
            assert editor.workflow.status(job_id)['status']=='stopping'
            page.screenshot(path=str(run/'analysis_stopping.png'));model_release.set();page.wait_for_function('!busy')
            assert editor.workflow.status(job_id)['status']=='cancelled' and not editor.workflow.running
            assert len(editor.workflow.projects[project['id']]['runs'])==1
            assert not (folder.parent/'run_0002/report.json').exists()
            assert button.inner_text()=='Run Analysis' and button.is_enabled()
            assert page.evaluate("localStorage.getItem('poreActiveJob')") is None
        assert not errors,errors
        browser.close()
      (run/'verification.json').write_text(json.dumps(dict(red_stop_buttons=True,stop_bypasses_model_lock=True,automate_discard_all=True,automate_single_commit=True,stop_during_commit=True,analysis_no_partial_run=True,errors=errors),indent=2))
    finally:release.set();server.shutdown();server.server_close()
    print(run)


if __name__=='__main__':main()
