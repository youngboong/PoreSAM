"""Native Details window movement, linked selection, editing and lifecycle check."""
import io
import json
import os
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from PIL import Image
import webview
from desktop_details import DesktopDetails
from pore_editor import Editor,ROOT,make_handler,LocalPoreServer,save_images


def main():
    run=ROOT/'outputs/desktop_checks'/('details_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    yy,xx=np.mgrid[:240,:240];masks={1:(xx-60)**2+(yy-100)**2<20**2,2:(xx-160)**2+(yy-100)**2<30**2}
    gray=np.full((240,240),190,np.uint8)
    for mask in masks.values():gray[mask]=30
    editor=Editor(run/'edits',run/'projects',device='cpu');stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
    project=editor.workflow.upload('native-details.png',stream.getvalue());folder=editor.workflow.root/project['id']/'runs/run_0001';folder.mkdir(parents=True)
    report=dict(image=str(editor.workflow.root/project['id']/'input/normalized.png'),analysis_bottom_exclusive=240,entrance_candidate_count=2,scale=dict(um_per_pixel=.1,label_um=10,length_pixels=100),overlay_relative_path='images/entrance_candidates_overlay.png')
    (folder/'report.json').write_text(json.dumps(report));save_images(folder,gray,masks);np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in masks.items()})
    dataset=project['id']+'__run_0001';editor.workflow.projects[project['id']]['runs']=[dict(id='run_0001',number=1,dataset=dataset,config=project['config'])]
    server=LocalPoreServer(('127.0.0.1',0),make_handler(editor));threading.Thread(target=server.serve_forever,daemon=True).start()
    bridge=DesktopDetails();url=f'http://127.0.0.1:{server.server_port}'
    main_window=webview.create_window('PoreSAM Details Test',url,width=1050,height=760,x=30,y=30,js_api=bridge)
    bridge._main=main_window;bridge._url=url
    main_window.events.closed+=bridge.close_details
    failed=[]
    def wait(window,expression):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:
                result=window.evaluate_js(expression)
                if result:return result
            except Exception:pass
            time.sleep(.15)
        raise AssertionError(expression)
    def check():
      try:
        wait(main_window,"typeof busy!=='undefined'&&!busy&&Boolean(window.pywebview?.api?.open_details)")
        main_window.evaluate_js("void work('Loading',async()=>{await accept(await api('load',{dataset:"+json.dumps(dataset)+"}));showPanel('editor')})")
        wait(main_window,"state?.candidates.length===2&&!busy")
        main_window.evaluate_js("document.getElementById('openPoreDetails').click()")
        deadline=time.monotonic()+15
        while bridge._details is None and time.monotonic()<deadline:time.sleep(.1)
        child=bridge._details;assert child is not None
        wait(child,"typeof state!=='undefined'&&state?.candidates.length===2")
        assert not main_window.evaluate_js("document.getElementById('poreDetailsDialog').open")
        child.move(1120,60);child.resize(780,650)
        assert child.x>=main_window.x+main_window.width
        child.evaluate_js("document.querySelector('[data-sort-key=length_um]').click()")
        assert child.evaluate_js("document.querySelector('[data-sort-key=length_um]').parentElement.getAttribute('aria-sort')")=='ascending'
        child.evaluate_js("document.querySelector('#poreListRows tr[data-id=\"2\"]').click()")
        wait(main_window,'getSelectedPores().includes(2)')
        wait(child,"document.querySelector('#poreListRows tr[data-id=\"2\"]').getAttribute('aria-selected')==='true'")
        main_window.evaluate_js('selectPoreFromDetails(1)');wait(child,"getSelectedPores()[0]===1")
        child.evaluate_js("document.getElementById('detailsHistogramTab').click()")
        wait(child,"poreDetailsPlotData()?.type==='histogram'")
        child.evaluate_js("document.getElementById('detailsScatterTab').click()")
        wait(child,"poreDetailsPlotData()?.type==='scatter'")
        # One real edit in the owner immediately updates the detached measurements.
        main_window.evaluate_js("document.querySelector('[data-mode=select]').click();selectPoreFromDetails(1)")
        wait(child,'getSelectedPores()[0]===1')
        child.evaluate_js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}))")
        wait(main_window,'state.candidates.length===1&&!busy');wait(child,'state.candidates.length===1&&state.revision===1')
        child.evaluate_js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true}))")
        wait(child,'state.candidates.length===2&&state.revision===2')
        main_window.evaluate_js("document.getElementById('openPoreDetails').click()")
        assert bridge._details is child
        bridge.close_details();wait(main_window,'true');assert bridge._details is None
        bridge.open_details();wait(bridge._details,"typeof state!=='undefined'&&state?.revision===2")
        (run/'verification.json').write_text(json.dumps(dict(native_window=True,moved_outside_parent=True,resized=True,selection_both_directions=True,edit_sync=True,delete_and_undo=True,sorting=True,histogram=True,scatter=True,reopened=True),indent=2))
      except Exception:
        failed.append(traceback.format_exc());(run/'error.txt').write_text(failed[-1],encoding='utf-8')
      finally:
        main_window.destroy()
    webview.start(check,gui='edgechromium',private_mode=True)
    server.shutdown();server.server_close()
    assert bridge._details is None,'Owner close left a detached window'
    print(run,flush=True)
    return 1 if failed else 0


if __name__=='__main__':os._exit(main())
