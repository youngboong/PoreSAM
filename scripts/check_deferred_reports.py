"""Shape metrics and modeless details/plots, using isolated synthetic masks."""
import io
from pathlib import Path
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
    run=ROOT/'outputs/ui_checks'/('deferred_reports_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    yy,xx=np.mgrid[:400,:400]
    circle=(xx-90)**2+(yy-90)**2<=40**2
    ellipse=((xx-260)/70)**2+((yy-90)/30)**2<=1
    masks={1:circle,2:ellipse}
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
    editor=Editor(run/'edits',run/'projects');gray=np.full(circle.shape,180,np.uint8);gray[circle|ellipse]=40
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG');project=editor.workflow.upload('shapes.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=400,entrance_candidate_count=2,scale=dict(um_per_pixel=.5,label_um=50,length_pixels=100),overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report));np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()});save_images(folder,gray,masks);export_folder(folder)
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    from unittest.mock import patch
    editor.workflow.persist(editor.workflow.projects[project['id']])
    state=editor.state(dataset)
    # A legacy report is not reused as English; only explicit generation upgrades it.
    marker=folder/'measurements/report_format.json'
    marker.write_text(json.dumps(dict(language='ko',version=1)))
    with patch('pore_editor.export_folder',side_effect=AssertionError('Implicit report generation')):
        assert not editor.response(state)['report_ready']
    assert editor.generate_report(state,dict(revision=0))['report_ready']
    assert json.loads(marker.read_text())==dict(language='en',version=1)
    assert 'lang="en"' in (folder/'measurements/index.html').read_text(encoding='utf-8')
    assert state['revision']==0 and set(state['masks'])=={1,2}
    with patch('pore_editor.export_folder',side_effect=AssertionError('Unexpected report export')), patch('pore_editor.save_images',side_effect=AssertionError('Unexpected image export')):
        response=editor.mutate(state,dict(revision=0,target_id=1),'delete')
    revision=editor.revision_folder(dataset,1)
    assert not (revision/'measurements').exists() and not (revision/'images').exists()
    assert not response['report_ready'] and response['report_url'] is None and len(response['candidates'])==1
    reloaded=Editor(run/'edits',run/'projects').state(dataset)
    assert reloaded['revision']==1 and set(reloaded['masks'])=={2}
    with patch('pore_editor.export_folder',side_effect=RuntimeError('simulated export failure')):
        try:editor.generate_report(state,dict(revision=1))
        except RuntimeError:pass
        else:raise AssertionError('Expected export failure')
    assert not editor.response(state)['report_ready']
    ready=editor.generate_report(state,dict(revision=1))
    assert ready['report_ready'] and state['revision']==1
    assert (revision/'measurements/dashboard.pdf').exists() and (revision/'report_ready.json').exists()
    before=(revision/'measurements/dashboard.pdf').read_bytes()
    with patch('pore_editor.export_folder',side_effect=AssertionError('Report regenerated')):
        assert editor.generate_report(state,dict(revision=1))['report_ready']
        restored=editor.mutate(state,dict(revision=1),'undo')
    assert restored['revision']==2 and not restored['report_ready']
    assert (revision/'measurements/dashboard.pdf').read_bytes()==before
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
            page.locator('[data-panel=analysis]').click()
            assert page.locator('#generateReport').is_visible() and not page.locator('#analysisFrame').is_visible()
            assert page.locator('#analysisDownloads a').count()==0
            current=editor.revision_folder(dataset,2)
            assert not (current/'measurements').exists()
            page.locator('#exportDirectory').fill(str(run/'exports'))
            page.locator('#generateReport').click();page.wait_for_function('state.report_ready && !busy')
            bundle=Path(page.evaluate('state.exported_folder'))
            assert bundle.is_relative_to(run/'exports')
            assert (bundle/'original/shapes.png').read_bytes()==stream.getvalue()
            assert (bundle/'images/entrance_candidates_overlay.png').is_file()
            assert (bundle/'measurements/dashboard.pdf').is_file()
            assert (bundle/'images/comparison.png').is_file()
            first_bundle=bundle
            page.locator('#generateReport').click();page.wait_for_function('!busy')
            assert Path(page.evaluate('state.exported_folder'))!=first_bundle
            assert first_bundle.exists()
            assert page.locator('#analysisFrame').is_visible() and page.locator('#analysisDownloads a').count()==6
            assert page.evaluate('state.revision')==2
            assert (current/'measurements/dashboard.pdf').exists()
            assert page.request.get(f'http://127.0.0.1:{server.server_port}'+page.locator('#analysisFrame').get_attribute('src')).status==200
            page.screenshot(path=str(run/'generated_report.png'))
            page.locator('[data-panel=editor]').click()
            page.evaluate("async()=>{await accept(await api('delete',{...payload(),target_id:1}))}")
            assert page.evaluate('state.revision')==3 and not page.evaluate('state.report_ready')
            page.locator('[data-panel=analysis]').click()
            assert page.locator('#generateReport').is_visible() and not page.locator('#analysisFrame').is_visible()
            assert page.locator('#analysisDownloads a').count()==0
            page.screenshot(path=str(run/'report_pending.png'))
            assert not errors,errors
            print('PASS: deferred saves, reload, failed export retry, report reuse, undo, explicit UI generation, stale report hiding')
            print(run)
            browser.close()
    finally:server.shutdown()

if __name__=='__main__':main()
