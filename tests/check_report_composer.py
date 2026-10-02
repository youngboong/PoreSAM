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

def main():
    run=ROOT/'outputs/ui_checks'/('report_composer_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects',device='cpu')
    gray=np.full((220,300),180,np.uint8);yy,xx=np.indices(gray.shape)
    masks={1:((xx-80)/30)**2+((yy-80)/20)**2<1,2:((xx-180)/23)**2+((yy-140)/35)**2<1,3:(xx<25)&(yy<30)}
    for mask in masks.values():gray[mask]=35
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('report-test.png',stream.getvalue());folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    (folder/'report.json').write_text(json.dumps(dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=220,entrance_candidate_count=3,scale=dict(um_per_pixel=.1),report_deferred=True)))
    np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    state=editor.state(dataset);data=content(editor,state)
    assert data['stats']['candidate_count']==3 and data['stats']['complete_candidate_count']==2
    assert '3개' in data['summary'] and '2개' in data['summary']
    invalid=defaults(data);invalid['items']=[dict(id='../../anything',selected=True,caption='Invalid')]
    try:validate(invalid,data);raise AssertionError('Invalid asset accepted')
    except ValueError:pass
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
            page.locator('[data-panel=analysis]').click();page.wait_for_function("document.getElementById('reportComposeBody').classList.contains('hidden')===false && !busy")
            assert page.locator('.report-card').count()==0
            assert page.locator('#reportTitle').input_value()=='주사전자현미경(SEM)'
            assert page.locator('#reportContentTab').inner_text()=='Figures & Tables'
            assert page.locator('#reportMethod').input_value()==''
            page.locator('#reportTextTab').click()
            page.locator('#reportTitle').fill('SEM 기공 분석 보고서')
            page.locator('#reportConditions input').nth(1).fill('사용자 입력 기기')
            page.locator('#reportMethod').fill('시료를 준비하고 SEM 영상을 분석하였다.\n<직접 입력> & 수정 가능')
            page.locator('#reportResults').fill('사용자 수정 결과: 현재 측정값 검토 완료.')
            page.locator('#reportContentTab').click()
            page.locator('#addReportTable').click()
            assert page.locator('.report-card').first.locator('strong').inner_text()=='Table 1'
            page.locator('#addReportFigure').click()
            group=page.locator('.report-card').last
            group.get_by_role('button',name='Add Contents',exact=True).click()
            expect(page.locator('#reportContentsDialog')).to_be_visible()
            page.locator('#reportContentsChoices input[value=hist_diameter]').check()
            page.locator('#reportContentsChoices input[value=comparison]').check()
            page.locator('#applyReportContents').click()
            assert group.locator('.report-panel-image').count()==2
            assert page.locator('select[aria-label="Figure columns"]').count()==0
            group.get_by_role('button',name='Add Contents',exact=True).click()
            page.locator('#reportContentsChoices input[value=original]').check()
            page.locator('#cancelReportContents').click()
            assert group.locator('.report-panel-image').count()==2
            # Directional drops must preserve explicit rows through save/reload/export.
            def drag_panel(source_id,target_id,edge):
                source=group.locator(f'[data-asset-id="{source_id}"]');target=group.locator(f'[data-asset-id="{target_id}"]')
                transfer=page.evaluate_handle('new DataTransfer()');source.dispatch_event('dragstart',{'dataTransfer':transfer})
                rect=target.bounding_box();x=rect['x']+rect['width']*(.98 if edge=='right' else .5);y=rect['y']+rect['height']*(.98 if edge=='bottom' else .5)
                target.dispatch_event('dragover',{'dataTransfer':transfer,'clientX':x,'clientY':y})
                assert page.locator('[data-drop-edge]').count()==1
                target.dispatch_event('drop',{'dataTransfer':transfer,'clientX':x,'clientY':y})
                source.dispatch_event('dragend',{'dataTransfer':transfer});transfer.dispose()
            drag_panel('comparison','hist_diameter','right')
            assert group.locator('.report-panel-row').count()==1
            assert group.locator('.report-panel-row').first.locator('.report-panel-image').count()==2
            drag_panel('comparison','hist_diameter','bottom')
            assert group.locator('.report-panel-row').count()==2
            drag_panel('comparison','hist_diameter','right')
            group.get_by_role('button',name='Add Contents',exact=True).click()
            page.locator('#reportContentsChoices input[value=original]').check()
            page.locator('#applyReportContents').click()
            assert group.locator('.report-panel-row').count()==2
            assert group.locator('.report-panel-row').first.locator('.report-panel-image').count()==2
            target=page.locator('.report-card').first
            transfer=page.evaluate_handle('new DataTransfer()')
            group.locator('.report-drag-handle').dispatch_event('dragstart',{'dataTransfer':transfer})
            rect=target.bounding_box()
            target.dispatch_event('drop',{'dataTransfer':transfer,'clientY':rect['y']+1})
            transfer.dispose()
            assert page.locator('.report-card').first.locator('strong').inner_text()=='Figure 1'
            page.locator('#reportTextTab').click()
            page.locator('#setReportConditionsDefault').click();page.wait_for_function('!busy')
            preset=json.loads((run/'edits/report_conditions_default.json').read_text(encoding='utf-8'))
            assert preset[0][1]
            from report_composer import load
            assert load(editor,state)['options']['conditions']==preset
            page.locator('#reportContentTab').click()
            page.evaluate("window.reportTestClicks=[];for(const type of ['mousedown','mouseup','click'])document.addEventListener(type,e=>reportTestClicks.push({type,id:e.target.id,busy}),true)")
            page.locator('#loadReportContent').click();page.wait_for_function('!busy')
            frame=page.frame_locator('#composedReportPreview')
            try: expect(frame.locator('figure')).to_have_count(2)
            except Exception:
                print(page.evaluate("({busy,status:document.getElementById('status')?.textContent,preview:document.getElementById('reportPreviewStatus').textContent,clicks:reportTestClicks,handler:String(document.getElementById('refreshReportPreview').onclick)})"),errors,flush=True)
                page.screenshot(path=str(run/'failure.png'),full_page=True)
                raise
            assert frame.locator('body').inner_text().find('사용자 수정 결과')>=0
            assert frame.locator('body').inner_text().find('<직접 입력> & 수정 가능')>=0
            assert 'Revision' not in frame.locator('body').inner_text()
            assert 'Feret' not in frame.locator('body').inner_text()
            open_more(page)
            page.locator('#saveReportDraft').click();page.wait_for_function('!busy')
            saved=json.loads((run/'edits'/dataset/'report_draft.json').read_text(encoding='utf-8'))
            assert saved['results']=='사용자 수정 결과: 현재 측정값 검토 완료.'
            assert saved['items'][0]['image_ids']==['hist_diameter','comparison','original']
            assert saved['items'][0]['rows']==[['hist_diameter','comparison'],['original']]
            page.locator('#loadReportContent').click();page.wait_for_function('!busy')
            assert page.locator('#reportResults').input_value()==saved['results']
            assert page.locator('.report-panel-row').first.locator('.report-panel-image').count()==2
            page.locator('#exportDirectory').fill(str(run/'export'))
            page.locator('#generateReport').click();page.wait_for_function('!busy',timeout=90000)
            assert 'PDF and HWPX' in page.locator('#reportGenerationStatus').inner_text()
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(run/'composer.png'),full_page=True)
            export=run/'export';files=list(export.iterdir());assert len(files)==2 and all(f.is_file() for f in files)
            pdf_path=next(export.glob('*.pdf'));assert pdf_path.stat().st_size>1000
            from test_report_hwpx import verify_hwpx
            verify_hwpx(next(export.glob('*.hwpx')))
            assert not errors,errors
            browser.close()
        # No-pore case must still produce an editable report and empty distributions.
        empty=dict(state,masks={},report_content=None,measurements=None)
        no_pores=content(editor,empty);assert no_pores['stats']['candidate_count']==0
        assert '산출하지 않았다' in no_pores['summary']
        long=defaults(data);long['method']='긴 분석 방법 문장입니다. '*300;long['results']='측정값 설명입니다. '*100
        render_pdf(run/'long_text.pdf',data,long)
        sys.path.insert(0,str(ROOT/'outputs/report_test_deps'))
        import pymupdf as fitz
        for name,path in [('report',pdf_path),('long',run/'long_text.pdf')]:
            doc=fitz.open(path);text=''.join(p.get_text() for p in doc)
            assert '분석' in text
            if name=='report':
                assert '사용자 수정 결과' in text and '그림 1.' in text
                assert 'Revision' not in text and 'Feret' not in text
            for i,p in enumerate(doc):
                assert all(b[0]>=p.rect.width*.125-1 and b[1]>=0 and b[2]<=p.rect.width*.875+1 and b[3]<=p.rect.height+1 for b in p.get_text('blocks'))
                p.get_pixmap(matrix=fitz.Matrix(1,1)).save(run/f'{name}_page_{i+1}.png')
        (run/'verification.json').write_text(json.dumps(dict(success=True,errors=errors,export=str(export)),indent=2))
        print('VERIFIED '+str(run),flush=True)
    finally:server.shutdown()


if __name__=='__main__':main()
