"""End-to-end report composition, persistence, selection and PDF checks."""
import base64
import io
import json
from pathlib import Path
import sys
import threading
import time
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pore_editor import Editor,make_handler
from report_composer import content,defaults,validate,render_html,render_pdf


def open_more(page):
    if page.locator('#reportMore').get_attribute('open') is None:page.locator('#reportMore > summary').click()

def open_report(page):
    if page.locator('[data-panel=analysis]').is_disabled():
        page.locator('#openAnalysisTab').click()
        page.locator('#imageLibrary .image-card').first.click()
        page.wait_for_function('state&&!busy')
    page.locator('[data-panel=analysis]').click()

def main():
    run=ROOT/'outputs/ui_checks'/('multi_report_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects',device='cpu')
    gray=np.full((220,300),180,np.uint8);yy,xx=np.indices(gray.shape)
    masks={1:((xx-80)/30)**2+((yy-80)/20)**2<1,2:((xx-180)/23)**2+((yy-140)/35)**2<1,3:(xx<25)&(yy<30)}
    for mask in masks.values():gray[mask]=35
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('report-test.png',stream.getvalue());folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    (folder/'report.json').write_text(json.dumps(dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=220,entrance_candidate_count=3,scale=dict(um_per_pixel=.1),report_deferred=True)))
    np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    editor.workflow.persist(editor.workflow.projects[project['id']])
    stream=io.BytesIO();Image.fromarray(255-gray).save(stream,format='PNG')
    second=editor.workflow.upload('second-image.png',stream.getvalue());second_folder=editor.workflow.root/second['id']/'runs/run_0001';second_folder.mkdir(parents=True)
    (second_folder/'report.json').write_text(json.dumps(dict(image=str(editor.workflow.root/second['id']/'input/normalized.png'),analysis_bottom_exclusive=220,entrance_candidate_count=3,scale=dict(um_per_pixel=.2),report_deferred=True)))
    np.savez_compressed(second_folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    second_dataset=second['id']+'__run_0001';editor.workflow.projects[second['id']]['runs']=[dict(id='run_0001',number=1,dataset=second_dataset,config=second['config'])];editor.workflow.persist(editor.workflow.projects[second['id']])
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    try:
      with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('!busy')
        open_report(page);page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')")
        assert page.evaluate('state!==null')
        page.locator('#reportMore > summary').click()
        page.locator('#newReportWorkspace').click();page.wait_for_function('!busy')
        page.evaluate('state=null;controls()')
        page.locator('#addReportAnalyses').click();page.wait_for_function('!busy')
        assert page.locator('#reportAnalysesChoices input').count()==2
        for check in page.locator('#reportAnalysesChoices input').all():check.check()
        page.locator('#applyReportAnalyses').click();page.wait_for_function('!busy',timeout=90000)
        assert page.locator('.report-card').count()==2
        assert '2 images' in page.locator('#reportSourcesStatus').inner_text()
        page.locator('#reportTextTab').click();page.locator('#reportTitle').fill('Combined SEM report');page.locator('#reportResults').fill('Manually edited combined results.')
        page.wait_for_function("document.getElementById('reportDraftStatus').textContent==='Saved'",timeout=15000)
        identifier=page.locator('#reportWorkspaceSelect').input_value()
        page.locator('#reportContentTab').click();page.locator('#addReportTable').click()
        assert page.locator('#reportContentsChoices input').count()==2
        for check in page.locator('#reportContentsChoices input').all():check.check()
        page.locator('#applyReportContents').click()
        # Mix contents from two source images in one existing figure.
        figure=page.locator('.report-card').first;figure.get_by_role('button',name='Add Contents',exact=True).click()
        original=page.locator('#reportContentsChoices input[value$=_original]').last;original.check();page.locator('#applyReportContents').click()
        assert figure.locator('.report-panel-image').count()==2
        open_more(page)
        page.locator('#saveReportDraft').click();page.wait_for_function('!busy')
        saved=json.loads((run/'edits/report_workspaces'/f'{identifier}.json').read_text(encoding='utf-8'))
        from report_composer import METRICS
        expected_assets=2*(4+len(METRICS))
        assert len(saved['sources'])==2 and len(saved['assets'])==expected_assets
        assert len({a['id'] for a in saved['assets']})==expected_assets
        assert saved['options']['results']=='Manually edited combined results.'
        assert saved['options']['items'][0]['image_ids'][0].split('_comparison')[0]!=saved['options']['items'][0]['image_ids'][1].split('_original')[0]
        # Switching the active editor image must not replace the report.
        page.evaluate("async dataset=>{await accept(await api('load',{dataset}))}",dataset)
        open_report(page);page.wait_for_function('!busy')
        assert page.locator('#reportWorkspaceSelect').input_value()==identifier
        assert page.locator('.report-card').count()==4
        # New report and reopening keep independent content.
        open_more(page)
        page.locator('#newReportWorkspace').click();page.wait_for_function('!busy')
        assert page.locator('.report-card').count()==0
        open_more(page)
        page.locator('#reportWorkspaceSelect').select_option(identifier);page.wait_for_function('!busy')
        assert page.locator('.report-card').count()==4
        assert page.locator('#reportResults').input_value()=='Manually edited combined results.'
        page.locator('#exportDirectory').fill(str(run/'export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
        assert 'PDF and HWPX' in page.locator('#reportGenerationStatus').inner_text()
        assert len(list((run/'export').iterdir()))==2
        from test_report_hwpx import verify_hwpx
        text,pictures,tables=verify_hwpx(next((run/'export').glob('*.hwpx')));assert pictures==2 and tables>=2 and 'Manually edited' in text
        # New sessions start from the current analysis; saved reports reopen explicitly.
        page.close();page=browser.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('!busy');open_report(page);page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')")
        assert page.locator('#reportWorkspaceSelect').input_value()==''
        assert page.locator('#reportSourceChips .report-source-chip').count()==1
        open_more(page);page.locator('#reportWorkspaceSelect').select_option(identifier);page.wait_for_function('!busy')
        assert page.locator('#reportWorkspaceSelect').input_value()==identifier
        assert page.locator('.report-card').count()==4
        assert not page.locator('#reportWorkspaceSelect').is_visible()
        assert not page.locator('#saveReportDraft').is_visible()
        page.screenshot(path=str(run/'multi_report.png'),full_page=True)
        assert not errors,errors;browser.close()
      from report_workspace import handle
      state=editor.state(dataset);state['revision']+=1
      try:handle(editor,dict(action='preview',report_id=identifier,options=saved['options']))
      except ValueError as e:assert 'changed' in str(e)
      else:raise AssertionError('Stale report accepted')
      refreshed=handle(editor,dict(action='refresh',report_id=identifier));assert refreshed['options']['results']=='Manually edited combined results.'
      removed=handle(editor,dict(action='refresh',report_id=identifier,datasets=[dataset]));assert len(removed['sources'])==1
      allowed={a['id'] for a in removed['assets']}
      assert all(k in allowed for i in removed['options']['items'] for k in i.get('image_ids',[]))
      (run/'verification.json').write_text(json.dumps(dict(passed=True,errors=errors,two_sources=True,independent_drafts=True,reopened=True,mixed_sources=True,stale_guard=True,pdf_hwpx=True),indent=2))
      print('VERIFIED '+str(run),flush=True)
    finally:server.shutdown()

if __name__=='__main__':main()
