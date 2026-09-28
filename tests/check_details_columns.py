"""Column selection, sorting, CSV, plots and the detached desktop page."""
import csv
import io
import json
from pathlib import Path
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pore_editor import Editor, ROOT, make_handler
from pore_details_export import LABELS, integer_histogram_edges


def main():
    run = ROOT/'outputs/ui_checks'/('details_columns_'+time.strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True)
    editor = Editor(run/'edits', run/'projects', device='cpu')
    gray = np.full((160, 200), 180, np.uint8)
    a = np.zeros_like(gray, dtype=bool); a[30:60, 30:80] = True
    b = np.zeros_like(a); b[:40, 130:160] = True
    gray[a] = 20; gray[b] = 60
    buffer = io.BytesIO(); Image.fromarray(gray).save(buffer, format='PNG')
    project = editor.workflow.upload('columns.png', buffer.getvalue())
    folder = editor.workflow.root/project['id']/'runs/run_0001'; folder.mkdir(parents=True)
    report = dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),
        analysis_bottom_exclusive=160, entrance_candidate_count=2,
        scale=dict(um_per_pixel=.5), report_deferred=True)
    (folder/'report.json').write_text(json.dumps(report))
    np.savez_compressed(folder/'entrance_candidates.npz', candidate_1=a, candidate_2=b)
    dataset = project['id']+'__run_0001'
    editor.workflow.projects[project['id']]['runs'] = [dict(id='run_0001', number=1, dataset=dataset, config=project['config'])]
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(editor))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', headless=True)
            context = browser.new_context(viewport=dict(width=1280, height=900), device_scale_factor=2, accept_downloads=True)
            page = context.new_page(); page.on('pageerror', lambda error: errors.append(str(error)))
            base = f'http://127.0.0.1:{server.server_port}'
            page.goto(base)
            page.locator('#openAnalysisTab').click(); page.locator('#imageLibrary .image-card').first.click()
            page.wait_for_function('state && !busy')
            assert page.locator('#refreshMeasurements').inner_text()==''
            assert page.locator('#refreshMeasurements svg').count()==1
            page.locator('#selectBoundaryPores').click()
            assert page.evaluate('getSelectedPores()')==[2]
            page.wait_for_function("document.getElementById('selectedPoreCount').textContent==='1 selected'")
            page.wait_for_function("""()=>{
                const layer=document.createElement('canvas');layer.width=200;layer.height=160;
                const c=layer.getContext('2d');window.drawSelectedPore(c);
                return c.getImageData(140,10,1,1).data[3]>0&&c.getImageData(40,40,1,1).data[3]===0;
            }""")
            assert page.locator('#selectBoundaryPores').get_attribute('aria-pressed')=='true'
            assert page.locator('#delete').is_enabled()
            page.locator('#openPoreDetails').click(); page.wait_for_function('!busy')
            page.locator('#detailsTargetPicker summary').click()
            page.locator('#detailsTargetOptions input[value="2"]').uncheck()
            page.locator('#closePoreDetails').click()
            page.locator('#selectBoundaryPores').click()
            assert page.evaluate('getSelectedPores()')==[2]
            page.locator('#openPoreDetails').click(); page.wait_for_function('!busy')
            assert page.evaluate('getSelectedPores()')==[2], 'Opening a filtered details table must not clear boundary selection'
            page.locator('#detailsTargetPicker summary').click()
            page.locator('#detailsSelectAll').click()
            page.locator('#detailsTargetPicker summary').click()
            assert page.evaluate('getSelectedPores()')==[2]
            assert set(page.evaluate('poreColumnCatalog.filter(c=>!c[3]).map(c=>c[0])'))==set(LABELS)
            page.locator('#openDetailsColumns').click()
            assert page.locator('#availableColumns option').count()+page.locator('#visibleColumns option').count()==28
            assert not any(word in page.locator('#availableColumns').inner_text() for word in ['Thread', 'Holes', 'Snapshot', 'Class', 'Sample'])
            page.screenshot(path=str(run/'columns.png'))
            page.locator('#clearColumns').click(); assert page.locator('#confirmColumns').is_disabled()
            page.locator('#availableColumns').select_option(['candidate_id','area_um2','brightness_mean','complete','solidity'])
            page.locator('#addColumns').click()
            page.locator('#visibleColumns').select_option('candidate_id')
            for _ in range(3): page.locator('#columnsUp').click()
            expected = page.locator('#visibleColumns option').evaluate_all('(items)=>items.map(i=>i.value)')
            assert expected[0]=='candidate_id'
            page.locator('#confirmColumns').click()
            assert page.locator('#poreTable thead button').evaluate_all('(items)=>items.map(i=>i.dataset.sortKey)')==expected
            assert page.locator('#poreListRows tr').first.locator('td').count()==5
            page.locator('[data-sort-key=brightness_mean]').click()
            assert page.locator('#poreListRows tr').first.get_attribute('data-id')=='1'
            page.locator('[data-sort-key=brightness_mean]').click()
            assert page.locator('#poreListRows tr').first.get_attribute('data-id')=='2'
            with page.expect_download() as download:
                page.locator('#downloadDetails').click()
            path = run/'columns.csv'; download.value.save_as(path)
            rows = list(csv.reader(path.read_text(encoding='utf-8-sig').splitlines()))
            assert rows[0]==['ID','Area (µm²)','Brightness mean','Complete','Solidity']
            assert rows[1][0]=='2' and rows[1][3]=='No'
            page.locator('#openDetailsColumns').click(); page.locator('#clearColumns').click(); page.locator('#cancelColumns').click()
            assert page.locator('#poreTable thead th').count()==5
            page.locator('#detailsHistogramTab').click(); page.locator('#detailsXAxis').select_option('brightness_mean')
            assert page.evaluate('poreDetailsPlotData().count')==2
            with page.expect_download() as download:
                page.locator('#downloadPlot').click()
            download.value.save_as(run/'brightness.png'); page.wait_for_function('!busy')
            assert (run/'brightness.png').read_bytes().startswith(b'\x89PNG')
            # The app serves the same picker in its independent native details page.
            detached = context.new_page(); detached.on('pageerror', lambda error: errors.append(str(error)))
            snapshot = editor.measurements(editor.state(dataset))
            detached.goto(base+'/pore-details-window')
            detached.evaluate('''snapshot=>{
              window.pywebview={api:{details_state:async known=>({key:'snapshot',data:known==='snapshot'?null:snapshot,busy:false,stale:false,selected:[],color:'#ff00ff'}),close_details:async()=>{}}};
              window.dispatchEvent(new Event('pywebviewready'));
            }''', snapshot)
            detached.wait_for_function('state && !busy')
            assert detached.locator('#poreTable thead button').evaluate_all('(items)=>items.map(i=>i.dataset.sortKey)')==expected
            detached.locator('#openDetailsColumns').click()
            detached.locator('#visibleColumns').select_option('solidity'); detached.locator('#removeColumns').click()
            detached.locator('#confirmColumns').click()
            assert detached.locator('#poreTable thead th').count()==4
            detached.locator('#openDetailsColumns').click(); detached.locator('#resetColumns').click()
            detached.screenshot(path=str(run/'desktop_columns.png'))
            detached.locator('#confirmColumns').click()
            assert detached.locator('#poreTable thead th').count()==8
            # Window height must be used by the list, not capped at 360px.
            detached.set_viewport_size(dict(width=1000, height=600))
            small = detached.locator('#detailsTableView .details-table-wrap').bounding_box()['height']
            detached.set_viewport_size(dict(width=1600, height=1000))
            large = detached.locator('#detailsTableView .details-table-wrap').bounding_box()['height']
            assert large > small + 350, (small, large)
            assert detached.locator('#poreDetailsStats tr').first.locator('td').first.inner_text().strip() == 'min'
            detached.locator('#detailsHistogramTab').click()
            detached.wait_for_function("Math.abs(detailsPlot.width-detailsPlot.getBoundingClientRect().width*devicePixelRatio)<=1")
            counts = detached.evaluate('poreDetailsPlotData().counts')
            edges = detached.evaluate('poreDetailsPlotData().edges')
            values = [r['equivalent_diameter_um'] for r in snapshot['candidates']]
            expected_edges=integer_histogram_edges(values, int(np.ceil(np.sqrt(len(values)))))
            assert edges==expected_edges.tolist() and all(e==int(e) for e in edges)
            assert counts==np.histogram(values,bins=expected_edges)[0].tolist()
            assert sum(counts)==len(values)

            axes = detached.evaluate('({x:poreDetailsPlotData().xAxis,y:poreDetailsPlotData().yAxis})')
            assert all(t['value']==int(t['value']) and '.' not in t['label'] for t in axes['y']['ticks'])
            assert all(len(t['label'].split('.')[-1])<=1 for t in axes['x']['ticks'] if '.' in t['label'])
            assert axes['y']['range'][1]>max(counts)

            detached.set_viewport_size(dict(width=1900, height=1100))
            detached.wait_for_function("Math.abs(detailsPlot.width-detailsPlot.getBoundingClientRect().width*devicePixelRatio)<=1 && Math.abs(detailsPlot.height-detailsPlot.getBoundingClientRect().height*devicePixelRatio)<=1")
            assert detached.evaluate('poreDetailsPlotData().counts') == counts
            assert detached.evaluate('detailsPlot.width') > 3000
            detached.screenshot(path=str(run/'resized_histogram.png'))
            detached.locator('#detailsScatterTab').click()
            # Selection coordinates must stay in logical plot space at 2x DPI.
            detached.evaluate('window.selectPoreFromDetails=id=>window.clickedPlotId=id')
            detached.locator('#detailsXAxis').select_option('area_um2')
            detached.locator('#detailsYAxis').select_option('area_um2')
            rect = detached.locator('#detailsPlot').bounding_box()
            scatter = detached.evaluate('poreDetailsPlotData()')
            for axis in [scatter['xAxis'],scatter['yAxis']]:
                assert len({tick['label'] for tick in axis['ticks']})==len(axis['ticks'])
                assert all('.' not in tick['label'] for tick in axis['ticks'])
            value=min(row['area_um2'] for row in snapshot['candidates'])
            xr,yr=scatter['xAxis']['range'],scatter['yAxis']['range']
            px=100+(value-xr[0])/(xr[1]-xr[0])*950
            py=523-(value-yr[0])/(yr[1]-yr[0])*455
            detached.mouse.click(rect['x']+px/1100*rect['width'], rect['y']+py/620*rect['height'])
            detached.screenshot(path=str(run/'readable_scatter.png'))
            assert detached.evaluate('window.clickedPlotId') is not None
            detached.locator('#detailsTableTab').click()
            detached.screenshot(path=str(run/'resized_details.png'))
            # Integer boundaries, constant/sub-unit data, negative values and exact upper edge.
            detached.locator('#detailsHistogramTab').click()
            detached.locator('#detailsXAxis').select_option('equivalent_diameter_um')
            for values in [[.1,.2,.9], [5,5], [-2.3,-1,0,1.2], [0,1,2,3,4], [19.54,21.85]]:
                for target in [5,10,20,30]:
                    detached.locator('#detailsBins').select_option(str(target))
                    detached.evaluate("values=>renderPoreDetails({dataset:'bins-test',candidates:values.map((v,i)=>({candidate_id:i+1,equivalent_diameter_um:v})),stats:{candidate_union_area_percent:0}})",values)
                    plotted=detached.evaluate('poreDetailsPlotData()')
                    edges=integer_histogram_edges(values,target)
                    assert plotted['edges']==edges.tolist()
                    assert plotted['counts']==np.histogram(values,bins=edges)[0].tolist()
                    assert sum(plotted['counts'])==len(values)
                    assert all(t['value'] in plotted['edges'] for t in plotted['xAxis']['ticks'])
            assert not errors, errors
            browser.close()
    finally: server.shutdown()
    print('PASS: 28 columns, selection/order/cancel, sorting, CSV, extra-metric plot and detached app page')
    print(run)


if __name__ == '__main__': main()
