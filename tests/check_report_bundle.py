"""Report composer in the packaged WebView app; isolated synthetic analysis."""
import ctypes
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pore_editor import Editor
from check_desktop_windows import windows,close


def open_more(page):
    if page.locator('#reportMore').get_attribute('open') is None:page.locator('#reportMore > summary').click()

def main():
    run=ROOT/'outputs/desktop_checks'/('report_bundle_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'data/manual_edits',run/'data/projects',device='cpu')
    yy,xx=np.indices((240,320));masks={1:(xx-80)**2+(yy-90)**2<25**2,2:(xx-210)**2+(yy-140)**2<35**2}
    gray=np.full((240,320),190,np.uint8)
    for m in masks.values():gray[m]=30
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG');project=editor.workflow.upload('report-native.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=240,entrance_candidate_count=2,scale=dict(um_per_pixel=.1,label_um=10,length_pixels=100),report_deferred=True)
    (folder/'report.json').write_text(json.dumps(report));np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])];editor.workflow.persist(editor.workflow.projects[project['id']])
    other_stream=io.BytesIO();Image.fromarray(255-gray).save(other_stream,format='PNG')
    other=editor.workflow.upload('second-native.png',other_stream.getvalue());other_folder=editor.workflow.root/other['id']/'runs/run_0001';other_folder.mkdir(parents=True)
    other_report=dict(report,image=str(editor.workflow.root/other['id']/'input/normalized.png'),scale=dict(um_per_pixel=.2,label_um=20,length_pixels=100))
    (other_folder/'report.json').write_text(json.dumps(other_report));np.savez_compressed(other_folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    other_dataset=other['id']+'__run_0001';editor.workflow.projects[other['id']]['runs']=[dict(id='run_0001',number=1,dataset=other_dataset,config=other['config'])];editor.workflow.persist(editor.workflow.projects[other['id']])
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env=os.environ.copy();env['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS']=f'--remote-debugging-port={port}'
    process=subprocess.Popen([str(ROOT/'dist/PoreSAM/PoreSAM.exe'),'--data-dir',str(run/'data')],env=env)
    errors=[]
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            assert process.poll() is None
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version',timeout=1):break
            except OSError:time.sleep(.25)
        else:raise AssertionError('No test WebView')
        with sync_playwright() as p:
            browser=p.chromium.connect_over_cdp(f'http://127.0.0.1:{port}');context=browser.contexts[0]
            deadline=time.monotonic()+30
            while not context.pages and time.monotonic()<deadline:time.sleep(.1)
            page=context.pages[0];page.on('pageerror',lambda e:errors.append(str(e)))
            page.wait_for_function("typeof busy!=='undefined'&&!busy")
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state&&!busy')
            page.locator('[data-panel=analysis]').click();page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')",timeout=90000)
            assert page.locator('.report-card').count()==0
            assert page.locator('#reportTitle').input_value()=='주사전자현미경(SEM)'
            assert page.locator('#reportContentTab').inner_text()=='Figures & Tables'
            while page.get_by_role('button',name='Remove Figure',exact=True).count():page.get_by_role('button',name='Remove Figure',exact=True).first.click()
            page.locator('#addReportFigure').click()
            figure=page.locator('.report-card').first
            figure.get_by_role('button',name='Add Contents',exact=True).click()
            page.locator('#reportContentsChoices input[value=original]').check()
            page.locator('#reportContentsChoices input[value=hist_width]').check()
            page.locator('#applyReportContents').click()
            assert figure.locator('.report-panel-image').count()==2
            source=figure.locator('[data-asset-id="hist_width"]');target=figure.locator('[data-asset-id="original"]')
            transfer=page.evaluate_handle('new DataTransfer()');source.dispatch_event('dragstart',{'dataTransfer':transfer})
            rect=target.bounding_box();event={'dataTransfer':transfer,'clientX':rect['x']+rect['width']*.98,'clientY':rect['y']+rect['height']*.5}
            target.dispatch_event('dragover',event);assert target.get_attribute('data-drop-edge')=='right'
            target.dispatch_event('drop',event);transfer.dispose()
            assert figure.locator('.report-panel-row').count()==1
            page.locator('#loadReportContent').click();page.wait_for_function('!busy')
            expect(page.frame_locator('#composedReportPreview').locator('figure')).to_have_count(1)
            assert 'Revision' not in page.frame_locator('#composedReportPreview').locator('body').inner_text()
            page.locator('#reportTextTab').click();page.locator('#reportMethod').fill('사용자가 직접 입력한 SEM 분석 방법')
            page.locator('#reportResults').fill('사용자가 수정한 분석 결과 초안')
            open_more(page)
            page.locator('#saveReportDraft').click();page.wait_for_function('!busy')
            # Change pores, refresh report content, and retain manually edited text.
            page.evaluate("async()=>{await accept(await api('delete',{...payload(),target_ids:[1]}))}")
            page.locator('#loadReportContent').click();page.wait_for_function('!busy',timeout=90000)
            assert page.locator('#reportResults').input_value()=='사용자가 수정한 분석 결과 초안'
            assert 'Pores changed' in page.locator('#reportSummaryStatus').inner_text()
            page.locator('#reportContentTab').click();page.locator('#exportDirectory').fill(str(run/'export'))
            page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
            assert 'PDF and HWPX' in page.locator('#reportGenerationStatus').inner_text()
            export=run/'export';assert len(list(export.iterdir()))==2;assert next(export.glob('*.pdf')).stat().st_size>1000
            from test_report_hwpx import verify_hwpx
            hwpx_text,pics,tables=verify_hwpx(next(export.glob('*.hwpx')))
            assert '사용자가 수정한 분석 결과 초안' in hwpx_text and pics==1
            draft=json.loads(next((run/'data/manual_edits').glob('*/report_draft.json')).read_text(encoding='utf-8'))
            assert draft['items'][0]['image_ids']==['original','hist_width']
            assert draft['items'][0]['rows']==[['original','hist_width']]
            open_more(page)
            page.locator('#newReportWorkspace').click();page.wait_for_function('!busy')
            page.locator('#addReportAnalyses').click();page.wait_for_function('!busy')
            assert page.locator('#reportAnalysesChoices input').count()==2
            for check in page.locator('#reportAnalysesChoices input').all():check.check()
            page.locator('#applyReportAnalyses').click();page.wait_for_function('!busy',timeout=90000)
            assert page.locator('.report-card').count()==2
            assert '2 images' in page.locator('#reportSourcesStatus').inner_text()
            page.locator('#exportDirectory').fill(str(run/'combined_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
            assert 'PDF and HWPX' in page.locator('#reportGenerationStatus').inner_text()
            assert len(list((run/'combined_export').iterdir()))==2
            _,combined_pics,_=verify_hwpx(next((run/'combined_export').glob('*.hwpx')));assert combined_pics==2
            page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(run/'native_report.png'))
            assert not errors,errors
            handle=next(hwnd for hwnd,title,visible in windows(process.pid) if title=='PoreSAM' and visible)
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(handle),0x10,0,0);process.wait(timeout=20);assert process.returncode==0
        (run/'verification.json').write_text(json.dumps(dict(packaged_exe=True,errors=errors,export=str(export),manual_text_preserved=True),indent=2))
        print('VERIFIED '+str(run),flush=True)
    finally:
        if process.poll() is None:close(process)


if __name__=='__main__':main()
