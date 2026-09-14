"""Shape metrics and modeless details/plots, using isolated synthetic masks."""
import io
import json
import threading
import time
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from analyze_candidates import measure_masks,export_folder
from pore_editor import Editor,ROOT,make_handler,save_images


def main():
    run=ROOT/'outputs/ui_checks'/('incremental_preview_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    yy,xx=np.mgrid[:400,:400]
    circle=(xx-90)**2+(yy-90)**2<=40**2
    ellipse=((xx-260)/70)**2+((yy-90)/30)**2<=1
    left=np.zeros((400,400),bool);left[200:260,:30]=True
    bottom=np.zeros((400,400),bool);bottom[370:,200:260]=True
    masks={1:circle,2:ellipse,3:left,4:bottom}
    table,stats,*_=measure_masks(list(masks.items()),circle.shape,.5)
    assert abs(table.iloc[0].length_um-40)<.5 and abs(table.iloc[0].width_um-40)<.5
    assert abs(table.iloc[0].aspect_ratio-1)<.01 and table.iloc[0].roundness>.98
    assert abs(table.iloc[1].length_um-70)<.5 and abs(table.iloc[1].width_um-30)<.5
    assert abs(table.iloc[1].aspect_ratio-70/30)<.04 and abs(table.iloc[1].roundness-30/70)<.02
    rotated=measure_masks([(1,np.rot90(ellipse))],circle.shape,.5)[0].iloc[0]
    assert abs(rotated.length_um-table.iloc[1].length_um)<1e-9
    scaled=measure_masks([(1,ellipse)],circle.shape,2)[0].iloc[0]
    assert abs(scaled.length_um-table.iloc[1].length_um*4)<1e-9 and abs(scaled.roundness-table.iloc[1].roundness)<1e-9
    single=np.zeros((4,4),bool);single[1,1]=True
    degenerate=measure_masks([(1,single)],single.shape,1)[0].iloc[0]
    assert degenerate.aspect_ratio is None and degenerate.roundness is None
    editor=Editor(run/'edits',run/'projects');gray=np.full(circle.shape,180,np.uint8);gray[circle|ellipse|left|bottom]=40
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG');project=editor.workflow.upload('shapes.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=400,entrance_candidate_count=4,scale=dict(um_per_pixel=.5,label_um=50,length_pixels=100),overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report));np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()});save_images(folder,gray,masks);export_folder(folder)
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    # Deterministic SAM stand-in: exercise box/ellipse prompt plumbing without loading a model.
    class Predictor:
        calls=0
        def predict(self,point_coords,point_labels,box,multimask_output):
            self.calls+=1
            mask=np.zeros((400,400),bool)
            x0,y0,x1,y1=np.rint(box).astype(int);mask[y0:y1+1,x0:x1+1]=True
            return np.array([mask]),np.array([1.]),None
    editor.predictor=Predictor();editor.encoded_dataset=dataset
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
            def point(x,y):
                rect=page.locator('#image').bounding_box();return rect['x']+x*rect['width']/400,rect['y']+y*rect['height']/400
            def drag(x0,y0,x1,y1):
                page.mouse.move(*point(x0,y0));page.mouse.down();page.mouse.move(*point(x1,y1),steps=5);page.mouse.up()
            page.locator('[data-mode=box]').click();drag(50,200,100,250);drag(120,200,170,250)
            page.locator('#predict').click();page.wait_for_function("!busy && queuedRegions.every(r=>r.status==='Ready')")
            assert editor.predictor.calls==2
            tokens=page.evaluate('queuedRegions.map(r=>r.token)')
            page.locator('[data-mode=positive]').click();page.mouse.click(*point(70,220))
            assert page.locator('#apply').is_disabled()
            page.locator('#updateRegionPreview').click();page.wait_for_function("!busy && preview")
            assert editor.predictor.calls==3
            assert page.evaluate('queuedRegions[1].token')==tokens[1]
            assert page.evaluate('queuedRegions[0].input.labels')==[1]
            assert len(editor.state(dataset)['queued_previews'])==2
            page.locator('#predict').click();page.wait_for_function('!busy')
            assert editor.predictor.calls==3
            # Keep unsubmitted points with their own region when switching.
            page.locator('[data-mode=negative]').click();page.mouse.click(*point(40,190))
            page.locator('#regionQueueList .queued-region').nth(1).locator('button').first.click();page.wait_for_function('!busy')
            assert page.evaluate('queuedRegions[0].input.labels')==[1,0]
            assert page.evaluate('points.length')==0
            page.mouse.click(*point(110,190))
            page.locator('#updateRegionPreview').click();page.wait_for_function('!busy')
            assert editor.predictor.calls==4
            assert page.evaluate('queuedRegions[1].input.labels')==[0]
            page.locator('#predict').click();page.wait_for_function('!busy')
            assert editor.predictor.calls==5
            assert page.evaluate('state.revision')==0
            page.locator('#apply').click();page.wait_for_function('!busy && state.revision===1')
            assert page.evaluate('state.candidates.length')==5
            assert editor.predictor.calls==5
            page.locator('#apply').click();page.wait_for_function('!busy && state.revision===2')
            assert page.evaluate('state.candidates.length')==6
            assert editor.predictor.calls==5
            assert not errors,errors
            page.screenshot(path=str(run/'incremental_preview.png'))
            print('PASS: 2 initial predictions; one per edited region; unchanged previews reused; points stay with their region; individual Add; bounded cache')
            print(run)
            browser.close()
    finally:server.shutdown()

if __name__=='__main__':main()
