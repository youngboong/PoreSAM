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
            assert page.locator('[data-panel=analysis]').is_disabled()
            page.evaluate("showPanel('analysis')")
            assert not page.locator('#analysisPanel').is_visible()
            assert page.locator('#savedImagePane h2').count()==0
            page.locator('#openAnalysisTab').click();assert page.locator('#imageLibrary .image-card .hint').count()==0
            page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state&&!busy')
            # A previous active report for another image must not replace this analysis.
            prior_report=page.evaluate("async datasets=>await api('report-workspace',{action:'create',datasets:[datasets.find(d=>d!==state.dataset)]})",[dataset,other_dataset])
            expected_name=page.evaluate("async()=>(await api('report-content',payload())).name")
            page.locator('[data-panel=analysis]').click();page.wait_for_function("!busy&&!document.getElementById('reportComposeBody').classList.contains('hidden')",timeout=90000)
            assert page.locator('#reportWorkspaceSelect').input_value()==''
            assert page.locator('#reportSourceChips .report-source-chip').count()==1
            assert page.locator('#reportSourceChips .report-source-chip').inner_text()==expected_name
            assert page.locator('.report-card').count()==0
            assert page.locator('#reportTitle').input_value()=='주사전자현미경(SEM)'
            assert page.locator('#reportContentTab').inner_text()=='Figures & Tables'
            while page.get_by_role('button',name='Remove Figure',exact=True).count():page.get_by_role('button',name='Remove Figure',exact=True).first.click()
            page.locator('#addReportFigure').click()
            page.get_by_role('button',name='Remove Figure',exact=True).last.click()
            assert page.locator('.report-card').count()==0
            page.locator('#addReportFigure').click()
            figure=page.locator('.report-card').first
            assert figure.locator('.report-item-menu').count()==0
            assert figure.get_by_role('button',name='Remove Figure',exact=True).locator('svg').count()==1
            figure.get_by_role('button',name='Add Contents',exact=True).click()
            assert page.locator('#reportContentsChoices input[value=hist_roundness]').count()==1
            assert page.locator('#reportContentsChoices input[value=hist_area_fraction]').count()==1
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
            with page.expect_response('**/api/report-pdf-preview') as preview_response:
                page.locator('#previewReportPdf').click()
            page.wait_for_function('!busy',timeout=90000)
            preview=preview_response.value.json();assert preview['opened']
            response=page.request.get(page.url.rstrip('/')+preview['url'])
            assert response.headers['content-type']=='application/pdf'
            assert response.body().startswith(b'%PDF-') and len(response.body())>1000
            assert not (run/'export').exists()
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                handle=next((hwnd for hwnd,title,visible in windows(process.pid) if title=='PoreSAM - PDF Preview' and visible),None)
                if handle:break
                time.sleep(.2)
            assert handle
            page.wait_for_timeout(4000)
            from PIL import ImageGrab
            from ctypes import wintypes
            ctypes.windll.user32.SetForegroundWindow(ctypes.c_void_p(handle))
            page.wait_for_timeout(500)
            rect=wintypes.RECT();ctypes.windll.user32.GetWindowRect(ctypes.c_void_p(handle),ctypes.byref(rect))
            ImageGrab.grab(bbox=(rect.left,rect.top,rect.right,rect.bottom)).save(run/'pdf_preview.png')
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(handle),0x10,0,0)
            deadline=time.monotonic()+10
            while any(hwnd==handle for hwnd,_,visible in windows(process.pid)) and time.monotonic()<deadline:time.sleep(.1)
            assert not any(hwnd==handle for hwnd,_,_ in windows(process.pid))
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
            # The packaged app can save four files directly and compose figures from multiple folders.
            first=run/'saved_images';second=run/'external_images';second.mkdir()
            Image.new('RGB',(120,80),'red').save(second/'external.png')
            Image.new('RGB',(120,80),'red').save(second/'external_copy.png')
            page.locator('#editFromAnalysis').click()
            page.route('**/api/choose-export-folder',lambda route:route.fulfill(json={'directory':str(first)}))
            page.locator('#saveEditorImages').click();page.wait_for_function("!busy&&document.getElementById('status').textContent.startsWith('4 images saved:')")
            assert len(list(first.iterdir()))==4 and all(f.is_file() for f in first.iterdir())
            page.locator('[data-panel=analysis]').click();page.wait_for_function('!busy')
            open_more(page);page.locator('#newReportWorkspace').click();page.wait_for_function('!busy')
            page.route('**/api/choose-report-folders',lambda route:route.fulfill(json={'directories':[str(first),str(second)]}))
            page.locator('#addReportFolders').click();page.wait_for_function('!busy')
            assert page.locator('#reportSourceChips .report-source-chip').count()==2
            page.locator('#addReportAnalyses').click();page.wait_for_function('!busy')
            assert page.locator('#reportAnalysesChoices select').count()==0
            page.locator('#cancelReportAnalyses').click()
            page.locator('#addReportFigure').click();page.get_by_role('button',name='Add Contents',exact=True).click()
            assert page.locator('#reportContentsChoices input').count()==5
            page.locator('#reportContentsChoices input').first.check();page.locator('#reportContentsChoices input').last.check()
            page.locator('#applyReportContents').click()
            page.locator('#exportDirectory').fill(str(run/'folders_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
            assert len(list((run/'folders_export').iterdir()))==2
            _,folder_pics,_=verify_hwpx(next((run/'folders_export').glob('*.hwpx')));assert folder_pics==1
            page.locator('#addReportTable').click();page.wait_for_function('!busy',timeout=90000)
            assert page.locator('.report-card table').count()==1
            assert page.locator('.report-card').count()==2
            page.locator('#addReportTable').click();page.wait_for_function('!busy',timeout=90000)
            assert page.locator('.report-card table').count()==1
            page.locator('#exportDirectory').fill(str(run/'folder_table_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
            _,table_pics,table_count=verify_hwpx(next((run/'folder_table_export').glob('*.hwpx')));assert table_pics==1 and table_count==2
            page.locator('#addReportPlot').click();page.wait_for_function('!busy')
            assert page.locator('#reportPlotX option[value="area_um2"]').inner_text()=='면적 (µm²)'
            assert page.locator('#reportPlotY option[value="width_um"]').inner_text()=='너비 (µm)'
            page.locator('#reportPlotX').select_option('area_um2')
            page.locator('#previewReportPlot').click();page.wait_for_function('!busy')
            assert page.locator('#reportPlotPreview').is_visible()
            page.screenshot(path=str(run/'plot_options.png'))
            page.locator('#confirmReportPlot').click();page.wait_for_function('!busy')
            assert not page.locator('#reportPlotDialog').is_visible()
            page.locator('#addReportPlot').click();page.wait_for_function('!busy')
            page.locator('#reportPlotKind').select_option('scatter')
            page.locator('#reportPlotX').select_option('length_um');page.locator('#reportPlotY').select_option('width_um')
            page.locator('#confirmReportPlot').click();page.wait_for_function('!busy')
            assert page.locator('.report-card').count()==4
            with page.expect_response('**/api/report-workspace') as preview_response:
                page.locator('#previewReportPdf').click()
            page.wait_for_function('!busy',timeout=90000)
            preview=preview_response.value.json();assert preview['opened']
            assert page.request.get(page.url.rstrip('/')+preview['url']).body().startswith(b'%PDF-')
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                handle=next((hwnd for hwnd,title,visible in windows(process.pid) if title=='PoreSAM - PDF Preview' and visible),None)
                if handle:break
                time.sleep(.2)
            assert handle
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(handle),0x10,0,0)
            deadline=time.monotonic()+10
            while any(hwnd==handle for hwnd,_,visible in windows(process.pid)) and time.monotonic()<deadline:time.sleep(.1)
            page.locator('#exportDirectory').fill(str(run/'plots_export'));page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
            _,plot_pics,plot_tables=verify_hwpx(next((run/'plots_export').glob('*.hwpx')));assert plot_pics==3 and plot_tables==2
            page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(run/'native_report.png'))
            assert not errors,errors
            handle=next(hwnd for hwnd,title,visible in windows(process.pid) if title=='PoreSAM' and visible)
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(handle),0x10,0,0);process.wait(timeout=20);assert process.returncode==0
        (run/'verification.json').write_text(json.dumps(dict(packaged_exe=True,errors=errors,export=str(export),manual_text_preserved=True),indent=2))
        print('VERIFIED '+str(run),flush=True)
    finally:
        if process.poll() is None:close(process)


if __name__=='__main__':main()
