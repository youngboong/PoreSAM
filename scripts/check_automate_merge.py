"""Exercise conservative overlap decisions and transactional merge/undo in the UI."""
import copy
import io
import json
import threading
import time
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from automate_pores import reconcile,add_candidate,trim_candidate,consolidation_boxes,dark_shared_boundaries,propose_boxes
from pore_editor import Editor, ROOT, make_handler, save_images


def main():
    run=ROOT/'outputs/ui_checks'/('automate_merge_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    # More than 16 independent dark regions are now searched, up to 32.
    proposal_gray=np.full((300,400),200,np.uint8)
    for y in range(20,270,50):
        for x in range(20,370,50):proposal_gray[y:y+12,x:x+12]=20
    proposal_state=dict(gray=proposal_gray,masks={},report={})
    proposals=propose_boxes(proposal_state)['boxes']
    assert len(proposals)==32
    assert proposals[:16]==propose_boxes(proposal_state,limit=16)['boxes']
    assert len(propose_boxes(proposal_state,limit=40)['boxes'])==35
    proposal_state['masks']={1:proposal_gray<100}
    assert not propose_boxes(proposal_state)['boxes']
    yy,xx=np.mgrid[:240,:240]
    full=(xx-120)**2+(yy-120)**2<40**2
    masks={1:full&(xx<120),2:full&(xx>=120)}
    gray=np.full(full.shape,190,np.uint8);gray[full]=30
    merged,ids,review=reconcile(full,masks,gray,100,8)
    assert ids==[1,2] and not review and np.array_equal(merged,full)
    # The global shared-boundary threshold is 50%, independently of pore IDs.
    from scipy import ndimage as ndi
    seam=(ndi.binary_dilation(masks[1],iterations=2)&masks[2])|(ndi.binary_dilation(masks[2],iterations=2)&masks[1])
    ring=ndi.binary_dilation(full,iterations=4)&~full
    positions=np.flatnonzero(seam)
    needed=(len(positions)+1)//2
    half_dark=gray.copy();half_dark[seam]=190;half_dark.flat[positions[:needed]]=30
    assert dark_shared_boundaries(full,masks,[1,2],half_dark,ring,8)
    half_dark.flat[positions[needed-1]]=190
    assert not dark_shared_boundaries(full,masks,[1,2],half_dark,ring,8)
    # A bright separator prevents consolidation even if a SAM mask spans both.
    separated=gray.copy();separated[:,118:122]=190
    assert reconcile(full,masks,separated,100,8)[2]
    # An overlapping partial observation extends the same pore.
    partials={1:full&(xx<135)};candidate=full&(xx>105)
    partial,ids,review=reconcile(candidate,partials,gray,100,8)
    assert ids==[1] and np.array_equal(partial,full) and not review
    assert reconcile(full,{1:full},gray,100,8)==(None,[],False)
    # Merely touching objects remain distinct; no artificial bridge is drawn.
    touching=np.zeros_like(full);touching[80:110,80:100]=True
    other=np.zeros_like(full);other[80:110,100:120]=True
    touching_gray=np.full(full.shape,190,np.uint8);touching_gray[touching|other]=30
    result,ids,review=reconcile(other,{1:touching},touching_gray,100,8)
    assert result is not None and not ids and not (result&touching).any()
    # Low-coverage overlap is ambiguous and must enter review, not merge.
    a=np.zeros_like(full);a[60:100,60:100]=True
    b=np.zeros_like(full);b[60:100,85:125]=True
    ambiguous_gray=np.full(full.shape,190,np.uint8);ambiguous_gray[a|b]=30
    assert reconcile(b,{1:a},ambiguous_gray,100,8)[2]
    # A tiny overlap relative to the new pore should be trimmed, even if it is
    # a noticeable fraction of a much smaller existing pore.
    tiny=np.zeros_like(full);tiny[70:80,98:110]=True
    trim_gray=np.full(full.shape,190,np.uint8);trim_gray[a|tiny]=30
    trimmed,ids,review=reconcile(a,{1:tiny},trim_gray,100,8)
    assert trimmed is not None and not ids and not review
    assert np.array_equal(trimmed,a&~tiny)
    # Automate resolves ambiguous overlap itself and chooses the highest scoring
    # viable option, skipping a higher-scored blank/background mask.
    bad=np.zeros_like(full);bad[180:210,180:210]=True
    sandbox=dict(gray=ambiguous_gray,masks={1:a.copy()},report={},revision=0,next_id=2,
                 annotations={'1':dict(source='manual_polygon')})
    def fake_preview(state,payload):
        state['preview']=dict(masks=[bad,b])
        return dict(choices=[dict(score=.99),dict(score=.91)])
    fake=SimpleNamespace(check_revision=lambda *args:None,preview=fake_preview)
    added=add_candidate(fake,sandbox,dict(revision=0,box=[50,50,130,110]),staged=True)
    assert added['added'] and np.array_equal(sandbox['masks'][2],b&~a)
    assert np.array_equal(sandbox['masks'][1],a)
    assert sandbox['annotations']['2']['sam_score']==.91
    assert sandbox['annotations']['2']['overlap_resolution']=='trim'
    assert not sandbox.get('automate_review_boxes')
    assert not add_candidate(fake,sandbox,dict(revision=0,box=[50,50,130,110]),staged=True)['added']
    assert trim_candidate(full,{1:full},gray,100,8) is None
    # The shared SAM quality floor applies to both new pores and consolidation.
    for score,accepted in ((.699,False),(.7,True),(.772,True)):
        def score_preview(state,payload):
            state['preview']=dict(masks=[full.copy()])
            return dict(choices=[dict(score=score)])
        score_editor=SimpleNamespace(check_revision=lambda *args:None,preview=score_preview)
        for consolidation in (False,True):
            score_state=dict(gray=gray,masks=copy.deepcopy(masks) if consolidation else {},
                             report={},revision=0,next_id=3,
                             annotations={'1':dict(source='automatic'),'2':dict(source='automated_sam')} if consolidation else {})
            payload=dict(revision=0,box=[75,75,165,165])
            if consolidation:payload.update(consolidation_only=True,candidate_ids=[1,2])
            outcome=add_candidate(score_editor,score_state,payload,staged=True)
            assert bool(outcome.get('merged') or outcome.get('added'))==accepted,(score,consolidation,outcome)
            if accepted:assert np.array_equal(score_state['masks'][1 if consolidation else 3],full)
    grouping=dict(masks=masks,annotations={'2':dict(source='automated_sam',overlap_resolution='trim')})
    assert consolidation_boxes(grouping)[0]['candidate_ids']==[1,2]
    grouping['annotations']['2']['overlap_resolution']='add'
    assert consolidation_boxes(grouping)[0]['candidate_ids']==[1,2]
    grouping['annotations']['1']=dict(source='manual_cut')
    assert not consolidation_boxes(grouping)

    # A neighboring third pore must not block a valid pair in the same SAM mask.
    third=np.zeros_like(full);third[180:210,180:210]=True
    joint=full|third
    neighbor_gray=gray.copy();neighbor_gray[third]=30
    def joint_preview(state,payload):
        state['preview']=dict(masks=[joint.copy()]);return dict(choices=[dict(score=.95)])
    pair_state=dict(gray=neighbor_gray,masks={7:masks[1].copy(),52:masks[2].copy(),901:third.copy()},
                    report={},revision=0,next_id=902,
                    annotations={'7':dict(source='automatic'),'52':dict(source='automated_sam'),'901':dict(source='automatic')})
    pair_editor=SimpleNamespace(check_revision=lambda *args:None,preview=joint_preview)
    paired=add_candidate(pair_editor,pair_state,dict(revision=0,box=[75,75,215,215],consolidation_only=True,candidate_ids=[7,52,901]),staged=True)
    assert paired['merged'] and set(pair_state['masks'])=={7,901}
    assert np.array_equal(pair_state['masks'][7],full) and np.array_equal(pair_state['masks'][901],third)
    # A user Cut must remain intact in the normal proposal path too.
    cut_state=dict(gray=gray,masks={7:full&(xx<119),52:full&(xx>121)},report={},revision=0,next_id=53,
                   annotations={'7':dict(source='manual_cut'),'52':dict(source='manual_cut')})
    originals={i:m.copy() for i,m in cut_state['masks'].items()}
    def cut_preview(state,payload):
        state['preview']=dict(masks=[full.copy()]);return dict(choices=[dict(score=.99)])
    cut_editor=SimpleNamespace(check_revision=lambda *args:None,preview=cut_preview)
    assert not add_candidate(cut_editor,cut_state,dict(revision=0,box=[75,75,165,165]),staged=True).get('merged')
    assert all(np.array_equal(cut_state['masks'][i],m) for i,m in originals.items())

    editor=Editor(run/'edits',run/'projects',device='cpu')
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('merge-test.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=240,
                entrance_candidate_count=2,scale=dict(um_per_pixel=.1,label_um=10,length_pixels=100),
                overlay_relative_path='images/entrance_candidates_overlay.png',selection_settings=dict(min_area_pixels=100,min_contrast=8))
    (folder/'report.json').write_text(json.dumps(report));save_images(folder,gray,masks)
    np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':mask for i,mask in masks.items()})
    dataset=project['id']+'__run_0001'
    editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    editor.workflow.persist(editor.workflow.projects[project['id']])
    def preview(state,payload,polygon=False):
        state['preview']=dict(masks=[full.copy()]);return dict(choices=[dict(score=.99)])
    editor.preview=preview
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
      with patch('automate_pores.propose_boxes',return_value=dict(boxes=[dict(box=[75,75,165,165])])),sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}')
        page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
        before=copy.deepcopy(editor.state(dataset))
        page.locator('#automatePores').click();page.wait_for_function("!busy && document.getElementById('status').textContent.includes('Automate complete')")
        assert page.evaluate('state.candidates.length')==1 and page.evaluate('state.revision')==1
        assert '0 added, 1 merged' in page.locator('#status').inner_text()
        actual=editor.state(dataset)
        assert set(actual['masks'])=={1} and np.array_equal(actual['masks'][1],full)
        assert actual['annotations']['1']['merged_from']==[1,2] and actual['next_id']==before['next_id']
        assert editor.response(actual)['stats']['overlap_pixels']==0
        assert page.evaluate('Boolean(state.supplemental_overlay)')
        page.screenshot(path=str(run/'merged.png'))
        page.locator('[data-mode=select]').click();page.keyboard.press('Control+z')
        page.wait_for_function('!busy && state.candidates.length===2')
        assert all(np.array_equal(editor.state(dataset)['masks'][i],mask) for i,mask in masks.items())
        # Stop after merging into staging must preserve both originals.
        stopped=page.evaluate("""async()=>{
          const op=await api('start-automate',payload());
          await api('automate-add',{...payload(),...op,box:[75,75,165,165]});
          await api('stop-automate',op);
          const result=await api('commit-automate',{...payload(),...op});
          await api('finish-automate',op);return result;
        }""")
        assert stopped['cancelled'] and len(editor.state(dataset)['masks'])==2
        # Even when all dark pixels are covered, final consolidation revisits
        # a trimmed piece and its neighbor in a single undoable transaction.
        editor.state(dataset)['annotations']['2'].update(source='automated_sam',overlap_resolution='trim')
        with patch('automate_pores.propose_boxes',return_value=dict(boxes=[])):
            page.locator('#automatePores').click();page.wait_for_function('!busy&&state.candidates.length===1')
        assert editor.state(dataset)['annotations']['1']['merged_from']==[1,2]
        page.keyboard.press('Control+z');page.wait_for_function('!busy&&state.candidates.length===2')
        # Unusable overlap is skipped without creating review work or changing masks.
        with patch('automate_pores.reconcile',return_value=(None,[],True)):
            revision=page.evaluate('state.revision')
            page.locator('#automatePores').click();page.wait_for_function('!busy')
            assert page.evaluate('state.revision')==revision
            assert page.evaluate('queuedRegions.length')==0
        assert not errors,errors
        browser.close()
      (run/'verification.json').write_text(json.dumps(dict(proposal_limit=32,merge=True,partial_extension=True,bright_separator_protected=True,
            touching_not_merged=True,duplicate_ignored=True,automatic_trim=True,best_viable_option=True,no_review_queue=True,measurements=True,undo=True,stop_discards_merge=True,errors=errors),indent=2))
      print(run,flush=True)
    finally:server.shutdown()


if __name__=='__main__':main()
