"""Legacy discovery diagnostic and single-target box UI regression."""
import argparse
import io
import json
import threading
import time
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image
from pore_editor import Editor,ROOT,LocalPoreServer,make_handler,save_images
from box_pores import preview_box,separate_candidates
from segment_first_pass import overlay


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ui',action='store_true');args=parser.parse_args()
    run=ROOT/'outputs/ui_checks'/('box_pores_'+('ui_' if args.ui else 'sam_')+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    yy,xx=np.mgrid[:320,:420];targets=[(xx-x)**2+(yy-90)**2<=25**2 for x in (90,210,330)]
    baseline=(xx-45)**2+(yy-260)**2<=20**2;gray=np.full((320,420),190,np.uint8)
    for mask in targets+[baseline]:gray[mask]=30
    # A broad mask can hide multiple separate pores from the old dark-residual gate.
    broad=np.zeros_like(gray,dtype=bool);broad[64:117,64:357]=True
    raw=[dict(segmentation=mask,predicted_iou=.99) for mask in [broad]+targets]
    selected=separate_candidates(raw,gray,8,100)
    assert len(selected)==3 and all(any(np.array_equal(item['mask'],target) for item in selected) for target in targets)
    editor=Editor(run/'edits',run/'projects',device='cpu');stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG');project=editor.workflow.upload('box-pores.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=320,entrance_candidate_count=1,scale=dict(um_per_pixel=.1,label_um=10,length_pixels=100),overlay_relative_path='images/entrance_candidates_overlay.png',selection_settings=dict(min_area_pixels=100,min_contrast=8))
    (folder/'report.json').write_text(json.dumps(report));save_images(folder,gray,{1:baseline});np.savez_compressed(folder/'entrance_candidates.npz',candidate_1=baseline)
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])];editor.workflow.persist(editor.workflow.projects[project['id']])
    state=editor.state(dataset)
    if not args.ui:
        counts=[];ious=[]
        for box,expected in [([52,52,128,128],targets[:1]),([50,50,370,130],targets)]:
            state['queued_previews']={}
            result=preview_box(editor,state,dict(dataset=dataset,revision=0,box=box,max_candidates=32))
            masks=[np.unpackbits(state['queued_previews'][r['token']]['masks'][0],count=gray.size).reshape(gray.shape).astype(bool) for r in result['regions']]
            scores=[max((mask&target).sum()/(mask|target).sum() for mask in masks) for target in expected]
            assert len(masks)==len(expected),(len(masks),len(expected),scores)
            assert min(scores)>.85,scores
            assert not (np.sum(masks,axis=0)>1).any()
            assert state['revision']==0 and len(state['masks'])==1
            overlay(gray,masks,numbered=True).save(run/f'preview_{len(expected)}_pores.png')
            counts.append(len(masks));ious.append(scores)
        (run/'verification.json').write_text(json.dumps(dict(real_cpu_sam=True,counts=counts,ious=ious,preview_only=True,no_overlap=True),indent=2))
    else:
        from playwright.sync_api import sync_playwright
        from scipy import ndimage as ndi
        calls={'search':0,'refine':0}
        class Generator:
            def __init__(self,*args,**kwargs):pass
            def generate(self,image):
                calls['search']+=1
                return [dict(segmentation=image[:,:,0]<100,predicted_iou=.99)]
        class Predictor:
            model=object()
            def predict(self,point_coords,point_labels,box,multimask_output):
                calls['refine']+=1
                index=0 if point_coords is None else min(range(3),key=lambda i:abs((90+120*i)-point_coords[0][0]))
                return np.array([broad if point_coords is None else targets[index]]),np.array([.99]),None
        editor.predictor=Predictor();editor.encoded_dataset=dataset;errors=[]
        server=LocalPoreServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
          with patch('box_pores.ProgressGenerator',Generator),sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1600,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}');page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state&&!busy')
            page.locator('[data-mode=box]').click()
            rect=page.locator('#image').bounding_box()
            def point(x,y):return rect['x']+x*rect['width']/420,rect['y']+y*rect['height']/320
            page.mouse.move(*point(50,50));page.mouse.down();page.mouse.move(*point(370,130));page.mouse.up()
            page.locator('#predict').click();page.wait_for_function("!busy&&queuedRegions.length===1&&queuedRegions.every(r=>r.status==='Ready')")
            assert page.evaluate('state.candidates.length')==1 and page.evaluate('state.revision')==0
            assert calls==dict(search=0,refine=1),calls
            assert page.evaluate("queuedRegions[0].kind")=='Box'
            page.locator('#apply').click();page.wait_for_function('!busy&&state.revision===1')
            assert len(state['masks'])==2 and any(np.array_equal(m,broad) for m in state['masks'].values())
            page.keyboard.press('Control+z');page.wait_for_function('!busy&&state.revision===2')
            assert len(state['masks'])==1
            page.locator('[data-mode=box]').click()
            page.mouse.move(*point(50,50));page.mouse.down();page.mouse.move(*point(370,130));page.mouse.up()
            page.locator('#predict').click();page.wait_for_function("!busy&&queuedRegions.length===1&&queuedRegions[0].status==='Ready'")
            page.locator('[data-mode=positive]').click();page.mouse.click(*point(91,91))
            page.locator('#updateRegionPreview').click();page.wait_for_function('!busy')
            assert calls==dict(search=0,refine=3),calls
            page.screenshot(path=str(run/'single_box_preview.png'))
            page.locator('#apply').click();page.wait_for_function('!busy&&state.revision===3')
            assert page.evaluate('state.candidates.length')==2
            assert any(np.array_equal(m,targets[0]) for m in state['masks'].values())
            assert not errors,errors
            # A full queue rejects expansion atomically, without modifying saved pores.
            before=set(state['queued_previews'])
            try:preview_box(editor,state,dict(dataset=dataset,revision=3,box=[50,50,370,130],max_candidates=1))
            except ValueError as error:assert 'Found 3 pores' in str(error)
            else:raise AssertionError('Queue overflow was silently accepted')
            assert set(state['queued_previews'])==before and state['revision']==3
            (run/'verification.json').write_text(json.dumps(dict(one_box_one_target=True,broad_sam_mask_preserved=True,no_local_search=True,individual_add=True,include_refinement=True,undo=True,measurements_updated=True,legacy_queue_limit_atomic=True,errors=errors),indent=2))
            browser.close()
        finally:server.shutdown();server.server_close()
    print(run,flush=True)


if __name__=='__main__':main()
