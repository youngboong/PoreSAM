"""Check linked zoom/cursors and drawing coordinates using an isolated image."""
import io
import json
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pore_editor import Editor, ROOT, make_handler, save_images


def main():
    run=ROOT/'outputs/ui_checks'/('navigation_'+time.strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True)
    gray=np.full((768,1024),180,np.uint8)
    mask=np.zeros_like(gray,dtype=bool);mask[160:280,200:360]=True;gray[mask]=40
    editor=Editor(run/'edits',run/'projects',device='cpu')
    stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('navigation.png',stream.getvalue())
    folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=768,
                entrance_candidate_count=1,scale=dict(um_per_pixel=.5,label_um=50,length_pixels=100),
                overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report))
    np.savez_compressed(folder/'entrance_candidates.npz',candidate_1=mask);save_images(folder,gray,{1:mask})
    dataset=project['id']+'__run_0001'
    editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(editor))
    threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('#openAnalysisTab').click();page.locator('#imageLibrary .image-card').first.click()
            page.wait_for_function('state && !busy')

            def settle():page.evaluate('() => new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))')
            def zoom(value):
                page.evaluate('(z)=>{$("zoom").value=z;$("zoom").oninput()}',value);settle()
            def position(image,x,y):
                return page.evaluate('([id,x,y])=>{const c=$(id),r=c.getBoundingClientRect();return [r.left+x*r.width/c.width,r.top+y*r.height/c.height]}',[image,x,y])
            def move(image,x,y):page.mouse.move(*position(image,x,y));settle()
            def marker(image,x,y):
                expected=position(image,x,y)
                pointer=page.locator('#'+image).locator('xpath=../..').locator('.comparison-cursor-overlay')
                assert pointer.is_visible()
                box=pointer.locator('svg').bounding_box()
                assert abs(box['x']+15-expected[0])<1.5,(box,expected)
                assert abs(box['y']+15-expected[1])<1.5,(box,expected)
            zoom(100)
            pixels=page.evaluate('[canvas.toDataURL(),$("originalImage").toDataURL()]')
            move('originalImage',280,220);marker('image',280,220)
            assert page.locator('#originalViewport').locator('xpath=..').locator('.comparison-cursor-overlay').is_hidden()
            move('image',300,240);marker('originalImage',300,240)
            assert pixels==page.evaluate('[canvas.toDataURL(),$("originalImage").toDataURL()]')
            mouse=position('image',300,240)
            page.mouse.wheel(0,-120);settle()
            assert int(page.locator('#zoom').input_value())>100
            actual=page.evaluate('([x,y])=>eventPoint({clientX:x,clientY:y})',mouse)
            assert max(abs(actual[0]-300),abs(actual[1]-240))<1
            marker('originalImage',*actual)
            assert page.evaluate('canvas.style.width===$("originalImage").style.width && $("viewport").scrollLeft===$("originalViewport").scrollLeft')
            page.mouse.wheel(0,120);settle()
            assert int(page.locator('#zoom').input_value())<127
            move('originalImage',280,220)
            before=int(page.locator('#zoom').input_value());page.mouse.wheel(0,-120);settle()
            assert int(page.locator('#zoom').input_value())>before
            marker('image',280,220)
            # Palette tracks all three occupied colors, including user color changes.
            colors=[]
            for color in ['#ffd700','#ff7a00','#00ff66','#00a6ff','#ff00ff','#9654e8']:
                page.evaluate('(v)=>{$("poreColor").value=v;$("poreColor").oninput()}',color)
                move('originalImage',280,220);marker('image',280,220)
                values=page.evaluate('''() => {
                  const norm=c=>{const e=document.createElement('span');e.style.color=c;return e.style.color};
                  return [getComputedStyle(document.querySelector('.comparison-cursor-overlay:not(.hidden)')).color,
                    ...[$('poreColor').value,supplementalColor(),selectionColor()].map(norm)];
                }''')
                assert values[0] not in values[1:],values
                colors.append(values)
            assert len({v[0] for v in colors})>1
            # Linked scrolling updates the marker even with a stationary mouse.
            zoom(150);move('image',300,240)
            point=position('image',300,240)
            page.evaluate('$("viewport").scrollBy(45,30)');settle()
            actual=page.evaluate('([x,y])=>eventPoint({clientX:x,clientY:y})',point)
            marker('originalImage',*actual)
            # Selection and box drawing use original pixel coordinates after zoom.
            move('image',280,220);page.mouse.click(*position('image',280,220))
            page.wait_for_function('getSelectedPores().includes(1)')
            page.mouse.click(*position('image',280,220))
            page.wait_for_function('getSelectedPores().length===0&&!isPoreSelectionPending()')
            assert page.locator('#delete').is_disabled()
            page.locator('[data-mode=box]').click()
            page.mouse.move(*position('image',180,140));page.mouse.down()
            before=page.locator('#zoom').input_value();page.mouse.wheel(0,-120);settle()
            assert page.locator('#zoom').input_value()==before
            page.mouse.move(*position('image',380,300));page.mouse.up();settle()
            box=page.evaluate('box');assert max(abs(a-b) for a,b in zip(box,[180,140,380,300]))<1,box
            page.locator('[data-mode=select]').click()
            for value,delta in [(250,-120),(10,120)]:
                zoom(value);move('image',50,50);page.mouse.wheel(0,delta);settle()
                assert int(page.locator('#zoom').input_value())==value
            page.locator('#fitComparison').click();settle()
            move('originalImage',280,220);marker('image',280,220)
            page.screenshot(path=str(run/'linked_cursor.png'))
            page.mouse.move(5,5);settle()
            assert page.locator('.comparison-cursor-overlay:not(.hidden)').count()==0
            assert not errors,errors
            (run/'verification.json').write_text(json.dumps(dict(bidirectional=True,wheel_anchor=True,scroll_sync=True,drawing_coordinates=True,colors=colors,errors=errors),indent=2))
            browser.close()
    finally:server.shutdown();server.server_close()
    print(run)


if __name__=='__main__':main()
