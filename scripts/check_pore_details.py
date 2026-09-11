"""Shape metrics and modeless details/plots, using isolated synthetic masks."""
import io
import json
import re
import threading
import time
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from analyze_candidates import measure_masks,export_folder
from pore_editor import Editor,ROOT,make_handler,save_images


def main():
    run=ROOT/'outputs/ui_checks'/('details_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
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
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            assert page.locator('#newImagePane').is_visible() and not page.locator('#savedImagePane').is_visible()
            assert page.locator('#newImageTab').get_attribute('aria-selected')=='true'
            page.screenshot(path=str(run/'load.png'))
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
            assert not re.search('[\uac00-\ud7a3]',page.locator('body').inner_text())
            for removed in ['report','revision','saveLinks','fill','reload','reuseImage']:
                assert page.locator('#'+removed).count()==0
            assert not page.locator('#regionPanel').is_visible()
            page.screenshot(path=str(run/'editor.png'))
            assert not page.locator('#poreDetailsDialog').is_visible()
            page.locator('#openPoreDetails').click();assert page.locator('#poreDetailsDialog').is_visible()
            # Every edge and corner resizes in its own axis without moving the opposite edge.
            for direction in ['n','s','e','w','nw','ne','sw','se']:
                page.evaluate("Object.assign($('poreDetailsDialog').style,{left:'400px',top:'220px',right:'auto',width:'700px',height:'600px'})")
                before=page.locator('#poreDetailsDialog').bounding_box()
                grip=page.locator('[data-corner="'+direction+'"]').bounding_box()
                x,y=grip['x']+grip['width']/2,grip['y']+grip['height']/2
                dx=-40 if 'w' in direction else 40 if 'e' in direction else 0
                dy=-35 if 'n' in direction else 35 if 's' in direction else 0
                page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+dx,y+dy,steps=5);page.mouse.up()
                after=page.locator('#poreDetailsDialog').bounding_box()
                assert abs(after['width']-before['width']-(40 if dx else 0))<2,direction
                assert abs(after['height']-before['height']-(35 if dy else 0))<2,direction
                assert abs(after['x']-before['x']-(-40 if 'w' in direction else 0))<2,direction
                assert abs(after['y']-before['y']-(-35 if 'n' in direction else 0))<2,direction
            page.evaluate("Object.assign($('poreDetailsDialog').style,{left:'760px',top:'180px',right:'auto',width:'880px',height:'710px'})")
            assert page.locator('#poreListRows tr[data-id]').count()==2
            assert page.locator('#poreTable thead th').count()==6
            assert page.locator('#poreDetailsStats tr').count()==4
            observed=float(page.locator('#poreDetailsStats tr').nth(2).locator('td').first.inner_text())
            assert abs(observed-table.length_um.mean())<.001
            observed_std=float(page.locator('#poreDetailsStats tr').nth(3).locator('td').first.inner_text())
            assert abs(observed_std-table.length_um.std(ddof=1))<.001
            page.locator('#poreListRows tr[data-id="2"]').click();page.wait_for_function('window.getHighlightedPore()===2')
            page.screenshot(path=str(run/'details.png'))
            page.locator('#detailsHistogramTab').click();page.locator('#detailsXAxis').select_option('length_um')
            page.locator('#detailsBins').select_option('5')
            hist=page.evaluate('poreDetailsPlotData()');assert sum(hist['counts'])==2 and len(hist['counts'])==5
            page.locator('#detailsTargetPicker summary').click();page.locator('#detailsTargetOptions input[value="2"]').uncheck()
            hist=page.evaluate('poreDetailsPlotData()');assert hist['count']==1 and sum(hist['counts'])==1
            page.locator('#detailsSelectNone').click();assert page.evaluate('poreDetailsPlotData().count')==0
            page.locator('#detailsSelectAll').click();page.locator('#detailsTargetPicker summary').click()
            page.screenshot(path=str(run/'histogram.png'))
            page.locator('#detailsScatterTab').click();page.locator('#detailsXAxis').select_option('length_um');page.locator('#detailsYAxis').select_option('roundness')
            scatter=page.evaluate('poreDetailsPlotData()');assert scatter['count']==2 and scatter['yKey']=='roundness'
            # Known min/max values put the two points at opposite corners of the plot frame.
            rect=page.locator('#detailsPlot').bounding_box();page.mouse.click(rect['x']+100*rect['width']/1100,rect['y']+68*rect['height']/620)
            page.wait_for_function('window.getHighlightedPore()===1')
            page.screenshot(path=str(run/'scatter.png'))
            with page.expect_download() as plot_download:page.locator('#downloadPlot').click()
            plot_download.value.save_as(str(run/'export_scatter.png'))
            page.wait_for_function('!busy')
            assert Image.open(run/'export_scatter.png').width==1800
            with page.expect_download() as download:
                page.locator('#downloadDetails').click()
            download.value.save_as(str(run/'export.csv'));assert 'Roundness' in (run/'export.csv').read_text(encoding='utf-8-sig')
            # Moving the modeless popup leaves the editor usable; applying a polygon refreshes open details.
            title=page.locator('#poreDetailsHandle').bounding_box();old_x=title['x']
            page.mouse.move(title['x']+70,title['y']+15);page.mouse.down();page.mouse.move(title['x']+30,title['y']+45,steps=5);page.mouse.up()
            assert page.locator('#poreDetailsDialog').bounding_box()['x']<old_x
            page.locator('#detailsTableTab').click()
            preview=page.evaluate("async()=>await api('polygon',{...payload(),polygon:[[40,240],[140,240],[140,320],[40,320]],fill_holes:true})")
            page.evaluate("async p=>{await accept(await api('apply',{...payload(),token:p.token,choice:0}))}",preview)
            assert page.locator('#poreDetailsDialog').is_visible() and page.locator('#poreListRows tr[data-id]').count()==3
            assert page.evaluate('state.candidates.find(c=>c.candidate_id===3).roundness')>0
            page.locator('#detailsTargetPicker summary').click();page.locator('#detailsSelectNone').click();page.locator('#detailsTargetOptions input[value="3"]').check();page.locator('#detailsTargetPicker summary').click()
            assert page.locator('#poreDetailsStats tr').nth(3).locator('td').first.inner_text()=='—'
            page.keyboard.press('Escape');assert not page.locator('#poreDetailsDialog').is_visible()
            page.locator('#openPoreDetails').click();page.locator('[data-panel=analysis]').click();assert not page.locator('#poreDetailsDialog').is_visible()
            page.locator('[data-panel=load]').click()
            assert page.locator('#newImagePane').is_visible() and not page.locator('#savedImagePane').is_visible()
            page.locator('#openAnalysisTab').click();page.reload();page.wait_for_function('!busy')
            assert page.locator('#newImagePane').is_visible()
            # Even an already analyzed file enters calibration when imported through New Image.
            page.locator('#uploadFile').set_input_files(dict(name='shapes.png',mimeType='image/png',buffer=stream.getvalue()))
            page.wait_for_function("setupProject && !busy && !$('setupPanel').classList.contains('hidden')")
            assert not page.locator('#scaleConfirmed').is_checked()
            assert page.locator('#editorPanel').is_hidden()
            def wait_preview():
                page.wait_for_function("!previewPending && !previewInFlight && $('preprocessingImage').dataset.generation===String(previewGeneration)")
            wait_preview()
            original_preview=page.locator('#preprocessingImage').get_attribute('src')
            for method in ['gaussian','box','median','bilateral','kuwahara','none']:
                page.locator('#blurMethod').select_option(method);wait_preview()
                assert page.locator('#blurStrengthField').is_visible()==(method!='none')
                assert not re.search('[\uac00-\ud7a3]',page.locator('body').inner_text())
            page.locator('#normalizeEnabled').uncheck();wait_preview()
            page.locator('#backgroundStrength').fill('4');wait_preview()
            assert page.locator('#preprocessingImage').get_attribute('src')!=original_preview
            page.screenshot(path=str(run/'preprocess.png'))
            page.locator('#scaleUm').fill('50');page.locator('#scalePixels').fill('100')
            page.locator('#runAnalysis').click()
            assert 'Confirm calibration' in page.locator('#jobStatus').inner_text()
            assert not editor.workflow.running
            page.set_viewport_size(dict(width=1100,height=800))
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.wait_for_load_state('networkidle');assert not errors,errors;browser.close()
        (run/'result.json').write_text(json.dumps(dict(passed=True,metric_invariants=True,summary_stats=True,histogram_counts=True,scatter_selection=True,manual_updates=True,csv_export=True,browser_errors=errors),indent=2))
        print(run,flush=True)
    finally:server.shutdown()


if __name__=='__main__':main()
