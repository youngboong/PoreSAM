"""Direct four-file export and reports made from multiple external folders."""
import io
import json
import sys
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pore_editor import Editor,make_handler
from report_bundle import export_images
from report_workspace import handle
from test_report_hwpx import verify_hwpx

def main():
    run=ROOT/'outputs/ui_checks'/('folder_report_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects',device='cpu')
    pixels=np.arange(120*160,dtype=np.uint16).reshape(120,160)
    stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='TIFF')
    raw=stream.getvalue();project=editor.workflow.upload('sample.tif',raw)
    dataset=project['id']+'__run_0001';folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=100,entrance_candidate_count=1,scale=dict(um_per_pixel=.1),report_deferred=True)
    (folder/'report.json').write_text(json.dumps(report));mask=np.zeros((100,160),bool);mask[20:50,30:70]=True
    np.savez_compressed(folder/'entrance_candidates.npz',candidate_1=mask)
    editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])];editor.workflow.persist(editor.workflow.projects[project['id']])
    first=run/'first';second=run/'second';second.mkdir();Image.new('RGB',(160,120),'red').save(second/'other.png')
    state=editor.state(dataset)
    assert Path(export_images(editor,state,dict(directory=str(first))))==first
    expected={'sample_comparison.png','sample_segmentation.png','sample_pores_colored.png','sample_original.tif'}
    assert {f.name for f in first.iterdir()}==expected
    assert (first/'sample_original.tif').read_bytes()==raw
    original=np.asarray(Image.open(editor.workflow.root/project['id']/'input/normalized.png').convert('RGB'))
    assert np.array_equal(np.asarray(Image.open(first/'sample_segmentation.png'))[100:],original[100:])
    export_images(editor,state,dict(directory=str(first)))
    assert len(list(first.iterdir()))==8 and all(f.is_file() for f in first.iterdir())
    assert (first/'sample_original.tif').read_bytes()==raw
    # No folders or artifacts remain after either export.
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
      with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport=dict(width=1500,height=1050));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('!busy')
        page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state&&!busy');page.locator('[data-panel=analysis]').click();page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')")
        page.locator('#reportMore > summary').click()
        page.locator('#newReportWorkspace').click();page.wait_for_function('!busy')
        page.route('**/api/choose-report-folders',lambda route:route.fulfill(json={'directories':[str(first),str(second)]}))
        page.locator('#addReportFolders').click();page.wait_for_function('!busy')
        assert page.locator('#reportSourceChips .report-source-chip').count()==2
        identifier=page.locator('#reportWorkspaceSelect').input_value()
        saved=handle(editor,dict(action='load',report_id=identifier));assert len(saved['assets'])==9
        page.locator('#addReportFolders').click();page.wait_for_function('!busy')
        assert len(handle(editor,dict(action='load',report_id=identifier))['assets'])==9
        page.locator('#addReportAnalyses').click();page.wait_for_function('!busy')
        assert page.locator('#reportAnalysesChoices select').count()==0
        assert page.get_by_text('Analysis version',exact=True).count()==0
        page.locator('#cancelReportAnalyses').click()
        page.locator('#addReportFigure').click();page.get_by_role('button',name='Add Contents',exact=True).click()
        assert page.locator('#reportContentsChoices input').count()==5
        choices=page.locator('#reportContentsChoices input');choices.first.check();choices.last.check()
        page.screenshot(path=str(run/'folder_contents.png'))
        page.locator('#applyReportContents').click();page.wait_for_function("document.getElementById('reportDraftStatus').textContent==='Saved'")
        saved=handle(editor,dict(action='load',report_id=identifier));assert len(saved['options']['items'][0]['image_ids'])==2
        # Imported images are embedded; reports survive source-folder changes.
        (second/'other.png').unlink()
        page.locator('#previewReportPdf').click();page.wait_for_function('!busy',timeout=90000)
        assert page.locator('#reportPdfDialog').is_visible()
        response=page.request.get(page.evaluate("new URL(document.getElementById('reportPdfFrame').src,location.href).href"))
        assert response.headers['content-type']=='application/pdf' and response.body().startswith(b'%PDF-')
        page.locator('#closeReportPdf').click()
        page.locator('#exportDirectory').fill(str(run/'report_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
        assert len(list((run/'report_export').iterdir()))==2
        _,pictures,_=verify_hwpx(next((run/'report_export').glob('*.hwpx')));assert pictures==1
        page.reload();page.wait_for_function('!busy');page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state&&!busy');page.locator('[data-panel=analysis]').click();page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')")
        assert page.locator('#reportWorkspaceSelect').input_value()==''
        page.locator('#reportMore > summary').click();page.locator('#reportWorkspaceSelect').select_option(identifier);page.wait_for_function('!busy')
        assert page.locator('.report-panel-image').count()==2
        page.screenshot(path=str(run/'folder_report.png'))
        page.get_by_role('button',name='Remove folder second',exact=True).click();page.wait_for_function('!busy')
        assert page.locator('.report-panel-image').count()==1
        # The normal Analyze -> Generate Report flow adds the current measurements
        # even when the report was created from folders and not selected analyses.
        page.evaluate("async dataset=>{await accept(await api('load',{dataset}))}",dataset)
        page.locator('#addReportTable').click();page.wait_for_function('!busy')
        assert page.locator('.report-card table').count()==1
        assert page.locator('.report-card').count()==2
        assert not page.locator('#reportContentsDialog').is_visible()
        page.locator('#addReportTable').click();page.wait_for_function('!busy')
        assert page.locator('.report-card table').count()==1
        assert page.locator('.report-card').count()==2
        page.locator('#exportDirectory').fill(str(run/'table_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
        _,pictures,tables=verify_hwpx(next((run/'table_export').glob('*.hwpx')))
        assert pictures==1 and tables==2
        page.screenshot(path=str(run/'analysis_table.png'))
        page.locator('#addReportPlot').click();page.wait_for_function('!busy')
        page.locator('#reportPlotX').select_option('length_um');page.locator('#reportPlotBins').select_option('5')
        page.locator('#previewReportPlot').click();page.wait_for_function('!busy')
        assert page.locator('#reportPlotPreview').is_visible()
        page.screenshot(path=str(run/'histogram_options.png'))
        page.locator('#confirmReportPlot').click();page.wait_for_function('!busy')
        assert not page.locator('#reportPlotDialog').is_visible()
        assert page.locator('.report-card').count()==3
        page.locator('.report-card').first.get_by_role('button',name='Add Contents',exact=True).click()
        page.locator('#addContentsPlot').click();page.wait_for_function('!busy')
        page.locator('#reportPlotKind').select_option('scatter')
        page.locator('#reportPlotX').select_option('length_um');page.locator('#reportPlotY').select_option('width_um')
        page.locator('#previewReportPlot').click();page.wait_for_function('!busy')
        assert page.locator('#reportPlotPreview').is_visible()
        page.locator('#confirmReportPlot').click();page.wait_for_function('!busy')
        assert page.locator('.report-card').first.locator('.report-panel-image').count()==2
        saved=handle(editor,dict(action='load',report_id=identifier))
        assert len(saved['plots'])==2
        refreshed=handle(editor,dict(action='refresh',report_id=identifier))
        assert {p['id'] for p in refreshed['plots']} <= {a['id'] for a in refreshed['assets']}
        page.reload();page.wait_for_function('!busy');page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state&&!busy');page.locator('[data-panel=analysis]').click();page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')")
        assert page.locator('#reportWorkspaceSelect').input_value()==''
        page.locator('#reportMore > summary').click();page.locator('#reportWorkspaceSelect').select_option(identifier);page.wait_for_function('!busy')
        assert page.locator('.report-card').count()==3
        page.locator('#exportDirectory').fill(str(run/'plots_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
        _,pictures,tables=verify_hwpx(next((run/'plots_export').glob('*.hwpx')));assert pictures==2 and tables==2

        assert not errors,errors
        browser.close()
      print('VERIFIED '+str(run),flush=True)
    finally:server.shutdown()

if __name__=='__main__':main()
