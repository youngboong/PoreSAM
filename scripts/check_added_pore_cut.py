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
    run=ROOT/'outputs/ui_checks'/('added_pore_cut_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
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
        def predict(self,point_coords,point_labels,box,multimask_output):
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
            page.locator('#openPoreDetails').click()
            page.locator('#detailsTargetPicker').click()
            page.locator('#detailsSelectNone').click()
            page.locator('#closePoreDetails').click()
            page.locator('[data-mode=box]').click();drag(50,200,100,250)
            page.locator('#predict').click();page.wait_for_function("!busy && preview")
            page.locator('#apply').click();page.wait_for_function('!busy && state.revision===1')
            new_id=page.evaluate('Math.max(...state.candidates.map(c=>c.candidate_id))')
            page.locator('#openPoreDetails').click()
            assert page.locator(f'#poreListRows tr[data-id="{new_id}"]').count()==1
            row=page.evaluate('(id)=>state.candidates.find(c=>c.candidate_id===id)',new_id)
            assert row['area_um2']>0 and row['equivalent_diameter_um']>0
            page.locator('#closePoreDetails').click()
            page.locator('[data-mode=cut]').click();drag(40,225,110,225)
            # Single cuts remain compatible with an older server response.
            page.evaluate('delete state.cut_paths_supported')
            page.locator('#applyCut').click();page.wait_for_function('!busy && state.revision===2')
            assert page.locator('#cutStatus').inner_text()=='Cuts applied.'
            assert page.evaluate('state.candidates.length')==6
            page.locator('#openPoreDetails').click()
            newest=page.evaluate('Math.max(...state.candidates.map(c=>c.candidate_id))')
            assert page.locator(f'#poreListRows tr[data-id="{newest}"]').count()==1
            page.screenshot(path=str(run/'added_pore_measurements.png'))
            page.locator('#closePoreDetails').click()
            page.keyboard.press('Control+z');page.wait_for_function('!busy && state.revision===3')
            assert page.evaluate('state.candidates.length')==5
            drag(40,225,110,225);drag(75,190,75,260)
            page.evaluate('delete state.cut_paths_supported')
            page.locator('#applyCut').click();page.wait_for_function('!busy')
            assert 'Restart' in page.locator('#cutStatus').inner_text()
            assert page.evaluate('state.revision')==3
            assert not errors,errors
            print('PASS: newly added pore measurements with filtered details, cut, undo, old-server compatibility and explicit multi-cut error')
            print(run)
            browser.close()
    finally:server.shutdown()

if __name__=='__main__':main()
