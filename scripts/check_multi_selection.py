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
    run=ROOT/'outputs/ui_checks'/('multiselect_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
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
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
            def click_pixel(x,y,shift=False):
                rect=page.locator('#image').bounding_box()
                if shift:page.keyboard.down('Control')
                page.mouse.click(rect['x']+x*rect['width']/400,rect['y']+y*rect['height']/400)
                if shift:page.keyboard.up('Control')
            def selected(ids):
                page.wait_for_function('(ids)=>!isPoreSelectionPending() && JSON.stringify(getSelectedPores())===JSON.stringify(ids)',arg=ids)
            page.locator('#selectBoundaryPores').click();selected([3,4])
            assert page.evaluate('state.revision')==0
            click_pixel(15,230,True);selected([4])
            page.keyboard.press('Delete');page.wait_for_function('state.revision===1 && !busy')
            assert [r['candidate_id'] for r in page.evaluate('state.candidates')]==[1,2,3]
            page.keyboard.press('Control+z');page.wait_for_function('state.revision===2 && !busy')
            click_pixel(90,90)
            click_pixel(260,90,True)
            click_pixel(15,230,True)
            selected([1,2,3])
            click_pixel(260,90,True);selected([1,3])
            page.locator('#openPoreDetails').click()
            assert page.locator('#poreListRows tr.selected').count()==2
            page.locator('#closePoreDetails').click()
            page.keyboard.press('Delete');page.wait_for_function('state.revision===3 && !busy')
            assert [r['candidate_id'] for r in page.evaluate('state.candidates')]==[2,4]
            folder=editor.revision_folder(dataset,3)
            assert json.loads((folder/'edit.json').read_text())['action']['candidate_ids']==[1,3]
            assert not (folder/'measurements').exists()
            page.keyboard.press('Control+z');page.wait_for_function('state.revision===4 && !busy')
            assert len(page.evaluate('state.candidates'))==4
            page.locator('#selectBoundaryPores').click();selected([3,4])
            page.screenshot(path=str(run/'boundary_selection.png'))
            page.locator('[data-mode=box]').click();selected([])
            click_pixel(90,90,True);selected([])
            page.keyboard.press('Delete');assert page.evaluate('state.revision')==4
            page.locator('[data-mode=select]').click()
            page.locator('#selectBoundaryPores').click();selected([3,4])
            assert page.evaluate('mode')=='select'
            state=editor.state(dataset)
            for ids in [[],[1,999],[True],None]:
                try:editor.mutate(state,dict(revision=4,target_ids=ids),'delete')
                except ValueError:pass
                else:raise AssertionError(ids)
                assert state['revision']==4 and len(state['masks'])==4
            assert not errors,errors
            print('PASS: boundary selection, exclusions, rapid Ctrl-clicks, multi-highlight, batch delete, one-step undo, drawing guard, atomic validation')
            print(run)
            browser.close()
    finally:server.shutdown()

if __name__=='__main__':main()
