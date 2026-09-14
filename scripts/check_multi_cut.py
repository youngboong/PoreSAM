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
    run=ROOT/'outputs/ui_checks'/('multi_cut_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
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
            page.locator('[data-mode=cut]').click()
            def drag(x0,y0,x1,y1):
                rect=page.locator('#image').bounding_box()
                def point(x,y):return rect['x']+x*rect['width']/400,rect['y']+y*rect['height']/400
                page.mouse.move(*point(x0,y0));page.mouse.down();page.mouse.move(*point(x1,y1),steps=10);page.mouse.up()
            drag(40,90,140,90);drag(90,40,90,140);drag(40,90,140,90)
            assert page.evaluate('allCutPaths().length')==3
            assert '3' in page.locator('#applyCut').inner_text()
            page.keyboard.press('Control+z');assert page.evaluate('allCutPaths().length')==2
            assert page.evaluate('state.revision')==0
            # Clear and undo restore the complete set of pending strokes.
            page.locator('#clear').click();assert page.locator('#applyCut').is_disabled()
            page.keyboard.press('Control+z');assert page.evaluate('allCutPaths().length')==2
            drag(40,90,140,90)
            page.screenshot(path=str(run/'pending_cuts.png'))
            page.locator('#applyCut').click();page.wait_for_function('!busy && state.revision===1')
            actual=editor.state(dataset)
            assert len(actual['masks'])==7
            assert all(np.array_equal(actual['masks'][i],masks[i]) for i in [2,3,4])
            assert not page.evaluate('state.report_ready')
            meta=json.loads((editor.revision_folder(dataset,1)/'edit.json').read_text())
            assert len(meta['action']['paths'])==3
            assert len(meta['action']['changed'])==1 and len(meta['action']['changed'][0]['resulting_ids'])==4
            union=np.logical_or.reduce(list(actual['masks'].values()))
            assert not union[90,60:120].any() and not union[60:120,90].any()
            page.screenshot(path=str(run/'applied_cuts.png'))
            page.keyboard.press('Control+z');page.wait_for_function('!busy && state.revision===2')
            assert all(np.array_equal(actual['masks'][i],masks[i]) for i in masks)
            assert len(actual['masks'])==4
            # Invalid batch is rejected atomically, including an invalid final path.
            for paths in [[],[[[40,90],[140,90]],[[90,40]]],[[[40,90],[140,90]],[[999,90],[999,140]]]]:
                try:editor.mutate(actual,dict(revision=2,paths=paths,width=3),'cut')
                except ValueError:pass
                else:raise AssertionError('Invalid batch accepted')
                assert actual['revision']==2
            # Legacy one-path callers remain supported.
            editor.mutate(actual,dict(revision=2,path=[[40,90],[140,90]],width=3),'cut')
            assert len(actual['masks'])==5
            assert not errors,errors
            print('PASS: intersecting and duplicate cuts, stroke undo, clear undo, atomic save/undo, unchanged other pores, legacy API')
            print(run)
            browser.close()
    finally:server.shutdown()

if __name__=='__main__':main()
