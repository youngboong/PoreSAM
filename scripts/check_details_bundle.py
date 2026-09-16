"""Verify the native Details window in the packaged EXE via a test-only CDP port."""
import argparse
import ctypes
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pore_editor import Editor,ROOT,save_images
from check_desktop_windows import windows,close


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--check-box',action='store_true');parser.add_argument('--check-merge',action='store_true');args=parser.parse_args()
    run=ROOT/'outputs/desktop_checks'/('details_bundle_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'data/manual_edits',run/'data/projects',device='cpu')
    yy,xx=np.mgrid[:240,:240];masks={1:(xx-60)**2+(yy-100)**2<20**2,2:(xx-160)**2+(yy-100)**2<30**2};gray=np.full((240,240),190,np.uint8)
    for mask in masks.values():gray[mask]=30
    if args.check_merge:
        masks[3]=masks[1]&(xx>=60);masks[1]=masks[1]&(xx<60)
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG');project=editor.workflow.upload('native-details.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=240,entrance_candidate_count=2,scale=dict(um_per_pixel=.1,label_um=10,length_pixels=100),overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report));save_images(folder,gray,masks);np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])];editor.workflow.persist(editor.workflow.projects[project['id']])
    with socket.socket() as port_socket:port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
    env=os.environ.copy();env['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS']=f'--remote-debugging-port={port}'
    process=subprocess.Popen([str(args.bundle.resolve()/'PoreSAM.exe'),'--data-dir',str(run/'data')],env=env)
    errors=[]
    try:
      deadline=time.monotonic()+80
      while time.monotonic()<deadline:
        assert process.poll() is None
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version',timeout=1):break
        except OSError:time.sleep(.25)
      else:raise AssertionError('Test WebView debugging port did not open')
      with sync_playwright() as p:
        browser=p.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
        context=browser.contexts[0]
        deadline=time.monotonic()+30
        while not context.pages and time.monotonic()<deadline:time.sleep(.1)
        owner=context.pages[0];owner.on('pageerror',lambda error:errors.append(str(error)))
        owner.wait_for_function("typeof busy!=='undefined'&&!busy&&Boolean(window.pywebview?.api?.open_details)")
        assert owner.evaluate("analysisJobMessage({message:'Detecting pores…',status:'running',progress:40,elapsed_seconds:25,detection_remaining_seconds:130})")== 'Detecting pores… · Elapsed 25s · Detection: ~2m 10s remaining'
        assert 'Estimating time' in owner.evaluate("analysisJobMessage({message:'Loading SAM',status:'running',progress:5,elapsed_seconds:2,detection_remaining_seconds:null})")
        assert 'remaining' not in owner.evaluate("analysisJobMessage({message:'Saving masks',status:'running',progress:90,elapsed_seconds:30,detection_remaining_seconds:10})")
        assert 'remaining' not in owner.evaluate("analysisJobMessage({message:'Stopping',status:'stopping',progress:40,elapsed_seconds:30,detection_remaining_seconds:10})")
        owner.locator('#openAnalysisTab').click();owner.locator('#imageLibrary .image-card').first.click();owner.wait_for_function(f'state?.candidates.length==={len(masks)}&&!busy')
        revision=0
        if args.check_merge:
            merged=owner.evaluate("""async()=>{
              const op=await api('start-automate',payload());
              try {
                const preview=await api('automate-add',{...payload(),...op,box:[35,75,85,125]});
                const committed=await api('commit-automate',{...payload(),...op});
                if(committed.state)await accept(committed.state);
                return {preview,committed};
              }finally{await api('finish-automate',op)}
            }""")
            assert merged['preview']['merged'] and merged['committed']['merged_count']==1,merged
            assert owner.evaluate('state.candidates.length')==2
            revision=1
        if args.check_box:
            requests=[]
            owner.on('request',lambda request:requests.append(request.url))
            owner.locator('[data-mode=box]').click()
            owner.evaluate('box=[30,65,195,135];draw()')
            owner.locator('#predict').click()
            owner.wait_for_function("!busy&&queuedRegions.length===1&&queuedRegions[0].status==='Ready'",timeout=120000)
            assert owner.evaluate("queuedRegions[0].kind")=='Box'
            assert owner.evaluate('preview.choices.length')>=1
            assert any('/api/predict' in url for url in requests)
            assert not any('/api/box-pores' in url for url in requests)
            assert owner.evaluate('state.revision')==revision and owner.evaluate('state.candidates.length')==2
            owner.evaluate('clear();inputHistory=[];window.clearRegionQueue();draw()')
        with context.expect_page() as opened:owner.locator('#openPoreDetails').click()
        details=opened.value;details.on('pageerror',lambda error:errors.append(str(error)));details.wait_for_function("typeof state!=='undefined'&&state?.candidates.length===2")
        assert len([entry for entry in windows(process.pid) if entry[2]])==2
        handle=next(hwnd for hwnd,title,visible in windows(process.pid) if 'Pore Details' in title and visible)
        ctypes.windll.user32.SetWindowPos(ctypes.c_void_p(handle),None,1120,50,800,680,0x0004)
        details.locator('#poreListRows tr[data-id="2"]').click();owner.wait_for_function('getSelectedPores().includes(2)')
        details.locator('[data-sort-key=length_um]').click();details.locator('#detailsHistogramTab').click()
        assert details.evaluate('poreDetailsPlotData().type')=='histogram'
        details.locator('#detailsScatterTab').click();assert details.evaluate('poreDetailsPlotData().type')=='scatter'
        owner.locator('[data-mode=select]').click();owner.evaluate('selectPoreFromDetails(1)');details.wait_for_function('getSelectedPores()[0]===1')
        details.locator('#detailsTableTab').click();details.locator('#poreListRows tr[data-id="1"]').focus();details.keyboard.press('Delete')
        details.wait_for_function(f'state.candidates.length===1&&state.revision==={revision+1}')
        details.keyboard.press('Control+z');details.wait_for_function(f'state.candidates.length===2&&state.revision==={revision+2}')
        details.screenshot(path=str(run/'pore_details.png'))
        owner.locator('#openPoreDetails').click();assert len(context.pages)==2
        # Closing only the parent must also close its detached details window.
        parent=next(hwnd for hwnd,title,visible in windows(process.pid) if title=='PoreSAM' and visible)
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(parent),0x10,0,0)
        process.wait(timeout=20);assert process.returncode==0
        assert not errors,errors
      (run/'verification.json').write_text(json.dumps(dict(packaged_exe=True,single_target_box=args.check_box,automate_merge=args.check_merge,native_windows=2,selection_sync=True,edit_sync=True,keyboard_undo=True,plots=True,parent_closes_child=True,errors=errors),indent=2))
    finally:
      if process.poll() is None:close(process)
    print(run,flush=True)


if __name__=='__main__':main()
