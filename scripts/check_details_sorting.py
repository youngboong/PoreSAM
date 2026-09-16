"""Numeric Pore Details sorting, filtering, selection and CSV integration check."""
import csv
import io
import json
import math
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pore_editor import Editor, ROOT, make_handler, save_images


def main():
    run=ROOT/'outputs/ui_checks'/('details_sort_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    yy,xx=np.mgrid[:400,:400]
    masks={1:(xx-80)**2+(yy-80)**2<=30**2,2:((xx-240)/50)**2+((yy-80)/20)**2<=1,
           10:(xx-80)**2+(yy-240)**2<=30**2,11:(xx==240)&(yy==240)}
    gray=np.full((400,400),180,np.uint8)
    for mask in masks.values():gray[mask]=40
    editor=Editor(run/'edits',run/'projects',device='cpu')
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('sorting.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=400,
                entrance_candidate_count=len(masks),scale=dict(um_per_pixel=.5,label_um=50,length_pixels=100),
                overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report));save_images(folder,gray,masks)
    np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    dataset=project['id']+'__run_0001'
    editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click();page.wait_for_function('state && !busy')
            page.locator('#openPoreDetails').click()
            records=page.evaluate('state.candidates')
            original=[r['candidate_id'] for r in records]
            rows=lambda:[int(i) for i in page.locator('#poreListRows tr[data-id]').evaluate_all('(rows)=>rows.map(r=>r.dataset.id)')]
            footer=page.locator('#poreDetailsStats').inner_text()
            assert rows()==[1,2,10,11]
            keys=page.locator('.details-sort').evaluate_all('(buttons)=>buttons.map(b=>b.dataset.sortKey)')
            assert len(keys)==7
            for key in keys:
                button=page.locator(f'[data-sort-key="{key}"]')
                for _ in range(2):
                    button.click();direction=button.locator('xpath=..').get_attribute('aria-sort')
                    def order(r):
                        value=r.get(key);valid=isinstance(value,(float,int)) and math.isfinite(value)
                        return (not valid,value*(1 if direction=='ascending' else -1) if valid else 0,r['candidate_id'])
                    assert rows()==[r['candidate_id'] for r in sorted(records,key=order)],(key,direction,rows())
                    assert button.locator('.sort-direction').inner_text()==('▲' if direction=='ascending' else '▼')
                    assert page.locator('#poreTable thead th:not([aria-sort="none"])').count()==1
                    assert page.locator('#poreDetailsStats').inner_text()==footer
            assert page.evaluate('state.candidates.map(c=>c.candidate_id)')==original
            # Header buttons also work from the keyboard.
            button=page.locator('[data-sort-key="length_um"]');button.focus();page.keyboard.press('Enter')
            assert button.locator('xpath=..').get_attribute('aria-sort')=='ascending'
            page.keyboard.press('Space');assert button.locator('xpath=..').get_attribute('aria-sort')=='descending'
            page.locator('#poreListRows tr[data-id="10"]').click();page.wait_for_function('getSelectedPores().includes(10)')
            page.locator('#poreListRows tr[data-id="10"]').click();page.wait_for_function('getSelectedPores().length===0')
            page.locator('#poreListRows tr[data-id="1"]').click()
            page.locator('#poreListRows tr[data-id="10"]').click(modifiers=['Control'])
            page.wait_for_function('getSelectedPores().length===2')
            page.locator('#poreListRows tr[data-id="1"]').click()
            page.wait_for_function('JSON.stringify(getSelectedPores())==="[10]"')
            button.click();assert page.locator('#poreListRows tr[data-id="10"]').get_attribute('aria-selected')=='true'
            page.locator('#detailsTargetPicker summary').click();page.locator('#detailsTargetOptions input[value="2"]').uncheck()
            page.locator('#detailsTargetPicker summary').click()
            assert 2 not in rows() and button.locator('xpath=..').get_attribute('aria-sort')=='ascending'
            # New/updated measurements immediately respect the current ordering.
            page.evaluate('''() => {
              const extra={...state.candidates[0],candidate_id:12,length_um:1.0002};
              state={...state,candidates:state.candidates.map(c=>c.candidate_id===1?{...c,length_um:1.0001}:c).concat(extra)};
              renderPoreDetails(state);
            }''')
            assert rows()[:3]==[11,1,12],rows()
            assert 2 not in rows()
            expected=rows();page.locator('#closePoreDetails').click();page.locator('#openPoreDetails').click()
            assert rows()==expected and button.locator('xpath=..').get_attribute('aria-sort')=='ascending'
            with page.expect_download() as download:page.locator('#downloadDetails').click()
            downloaded=download.value;downloaded.save_as(run/'pore_details.csv')
            with (run/'pore_details.csv').open(encoding='utf-8-sig',newline='') as f:export=list(csv.reader(f))
            assert [int(row[0]) for row in export[1:-4]]==expected
            assert [row[0] for row in export[-4:]]==['min','max','mean','std']
            page.screenshot(path=str(run/'sorted_details.png'))
            assert not errors,errors
            (run/'verification.json').write_text(json.dumps(dict(columns=keys,numeric_sort=True,missing_last=True,selection_preserved=True,filtered_and_new_rows=True,csv_order=True,errors=errors),indent=2))
            browser.close()
    finally:server.shutdown();server.server_close()
    print(run)


if __name__=='__main__':main()
