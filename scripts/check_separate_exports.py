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
    run=ROOT/'outputs/ui_checks'/('separate_exports_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
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
    stream=io.BytesIO();Image.fromarray(np.vstack([gray,np.full((40,400),91,np.uint8)])).save(stream,format='PNG');project=editor.workflow.upload('shapes.png',stream.getvalue())
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
            directory=run/'exported'
            page.route('**/api/choose-export-folder',lambda route:route.fulfill(json={'directory':str(directory)}))
            page.locator('#saveEditorImages').click();page.wait_for_function('!busy && document.getElementById("status").textContent.startsWith("3 images saved:")')
            exported=next(directory.glob('*_images_*'))
            assert {p.name for p in exported.iterdir()}=={'comparison.png','segmentation.png','pores_colored.png'}
            org=np.array(Image.open(editor.workflow.root/project['id']/'input/normalized.png').convert('RGB'))
            for name in ['segmentation.png','pores_colored.png']:
                pixels=np.array(Image.open(exported/name));assert pixels.shape==org.shape
                assert np.array_equal(pixels[400:],org[400:])
                assert not np.array_equal(pixels[:400],org[:400])
            comparison=np.array(Image.open(exported/'comparison.png'))
            assert comparison.shape==(440,800,3) and np.array_equal(comparison[:,:400],org)
            assert np.array_equal(comparison[:,400:],np.array(Image.open(exported/'segmentation.png')))
            assert not np.array_equal(np.array(Image.open(exported/'segmentation.png')),np.array(Image.open(exported/'pores_colored.png')))
            assert page.evaluate('state.revision')==0
            page.locator('[data-panel=analysis]').click()
            page.locator('#chooseExportDirectory').click();page.wait_for_function('!busy')
            page.locator('#generateReport').click();page.wait_for_function('!busy && document.getElementById("exportResult").textContent.startsWith("Exported to:")')
            reports=next(directory.glob('*_report_*'))
            assert {p.name for p in reports.iterdir()}=={'report.pdf','report.html'}
            html=(reports/'report.html').read_text(encoding='utf-8')
            assert html.count('src="data:image/png;base64,')==2
            assert 'href="candidates.csv"' not in html and 'href="report.pdf"' in html
            assert page.evaluate('state.revision')==0
            assert not errors,errors
            print('PASS: Save Images + native picker wiring, exactly three images, original footer, unchanged masks, report-only PDF/standalone HTML')
            print(exported);print(reports)
            browser.close()
    finally:server.shutdown()

if __name__=='__main__':main()
