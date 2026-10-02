"""Exercise the packaged EXE on isolated data, including real CPU SAM inference."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import base64
import argparse
import ctypes
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=ROOT/'dist/PoreSAM');parser.add_argument('--check-stop',action='store_true');args=parser.parse_args()
    bundle=args.bundle.resolve()
    cuda_libraries=[p.name for p in bundle.rglob('*.dll') if any(term in p.name.lower() for term in ('cuda','cublas','cudnn','cufft','curand','cusparse','cusolver','nvrtc','nvjit'))]
    assert not cuda_libraries,cuda_libraries
    run=ROOT/'outputs/desktop_checks'/('bundle_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env['PATH']=os.pathsep.join([env.get('SystemRoot','C:/Windows')+'/System32',env.get('SystemRoot','C:/Windows')])
    process=subprocess.Popen([str(bundle/'PoreSAM.exe'),'--data-dir',str(run/'data')],env=env)
    started=time.monotonic()
    try:
        log=run/'logs/desktop.log'
        while time.monotonic()-started<120:
            if process.poll() is not None:raise RuntimeError('EXE exited: '+str(process.returncode))
            match=re.search(r'port=(\d+)',log.read_text(encoding='utf-8')) if log.is_file() else None
            if match:break
            time.sleep(.5)
        else:raise RuntimeError('EXE startup timeout')
        base='http://127.0.0.1:'+match[1]
        for route, expected in [('/pore-columns.js', 'Available columns'), ('/pore-details-window', '/pore-columns.js'), ('/', 'Refresh measurements')]:
            with urllib.request.urlopen(base+route) as response:
                assert expected in response.read().decode('utf-8')
        def post(route,payload):
            request=urllib.request.Request(base+'/api/'+route,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Pore-Editor':'1'})
            with urllib.request.urlopen(request,timeout=90) as response:return json.load(response)
        yy,xx=np.mgrid[:160,:160];gray=np.full((160,160),190,np.uint8);gray[(xx-80)**2+(yy-80)**2<25**2]=30
        buffer=io.BytesIO();Image.fromarray(gray).save(buffer,format='PNG')
        project=post('upload',dict(name='desktop-test.png',content=base64.b64encode(buffer.getvalue()).decode()))['project']
        config={**project['config'],'analysis_bottom':160,'scale_um':10,'scale_pixels':100,'scale_confirmed':True,'points_per_side':16,'min_area_pixels':20}
        if args.check_stop:
            cancelled=post('analyze',dict(project_id=project['id'],config={**config,'points_per_side':48}))
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                status=post('job',cancelled)
                if status['progress']>=10:break
                assert status['status']=='running',status
                time.sleep(.1)
            else:raise AssertionError('No SAM batch completed before timeout')
            requested=time.monotonic();post('stop-analysis',cancelled)
            while time.monotonic()-requested<30:
                status=post('job',cancelled)
                if status['status']=='cancelled':break
                assert status['status']=='stopping',status
                time.sleep(.1)
            else:raise AssertionError('SAM did not stop at the next batch')
            assert not post('project',dict(project_id=project['id']))['runs']
            print('Real CPU SAM stopped in',round(time.monotonic()-requested,3),'seconds',flush=True)
        job=post('analyze',dict(project_id=project['id'],config=config))
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            status=post('job',job)
            print(status.get('progress'),status.get('message'),flush=True)
            if status['status']=='failed':raise RuntimeError(status['message'])
            if status['status']=='complete':break
            time.sleep(2)
        else:raise RuntimeError('CPU analysis timeout')
        dataset=status['dataset'];state=post('load',dict(dataset=dataset))
        assert state['candidates'],'No pore detected in the synthetic test image'
        assert state['selection_settings']['method']=='two_stage_nested'
        model_manifest=bundle/'_internal/checkpoints/default_model.json'
        if model_manifest.is_file():
            deployment=json.loads(model_manifest.read_text(encoding='utf-8'))
            saved_report=next((run/'data/projects'/project['id']/'runs').glob('*/report.json'))
            assert json.loads(saved_report.read_text())['checkpoint_sha256']==deployment['sha256']
        assert all(key in state['candidates'][0] for key in ['angle_deg','solidity','brightness_mean','complete','mass_center_x_um'])
        measured=post('measurements',dict(dataset=dataset,revision=state['revision']))
        assert measured['candidates']==state['candidates']
        # Prompted SAM also loads bundled model/configs, independently of automatic masks.
        preview=post('predict',dict(dataset=dataset,revision=state['revision'],points=[],labels=[],box=[50,50,110,110]))
        assert preview['choices']
        images=post('export-images',dict(dataset=dataset,revision=state['revision'],directory=str(run/'exported'),labels=False))
        report=post('generate-report',dict(dataset=dataset,revision=state['revision'],export_directory=str(run/'exported')))
        assert len(list(Path(images['exported_folder']).glob('*.png')))==4
        assert {p.name for p in Path(report['exported_folder']).iterdir()}=={'report.pdf','report.html'}
        assert (Path(report['exported_folder'])/'report.pdf').read_bytes().startswith(b'%PDF-')
        result=dict(device='cpu',stop_checked=args.check_stop,automatic_candidates=len(state['candidates']),prompted_choices=len(preview['choices']),images=images['exported_folder'],report=report['exported_folder'],elapsed_seconds=time.monotonic()-started)
    finally:
        # Close only the test app's own native window.
        user=ctypes.windll.user32
        callback=ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
        def close(hwnd,param):
            pid=ctypes.c_ulong();user.GetWindowThreadProcessId(ctypes.c_void_p(hwnd),ctypes.byref(pid))
            if pid.value==process.pid and user.IsWindowVisible(ctypes.c_void_p(hwnd)):user.PostMessageW(ctypes.c_void_p(hwnd),0x0010,0,0)
            return True
        user.EnumWindows(callback(close),0)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate();process.wait(timeout=10)
            raise RuntimeError('Desktop did not close gracefully')
    assert process.returncode==0,process.returncode
    assert 'Desktop stopped' in log.read_text(encoding='utf-8')
    assert 'torch=2.7.1+cpu cuda=None' in log.read_text(encoding='utf-8')
    result['native_closed']=True
    result['cuda_libraries']=cuda_libraries
    result['bundle_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    (run/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)
    print(run,flush=True)

if __name__=='__main__':main()
