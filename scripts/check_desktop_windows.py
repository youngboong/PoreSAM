"""Open concurrent Windows apps and verify independent workspaces and slot reuse."""
import argparse
import base64
import ctypes
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
user=ctypes.windll.user32
callback=ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)


def windows(pid):
    found=[]
    def collect(hwnd,_):
        handle=ctypes.c_void_p(hwnd);owner=ctypes.c_ulong();user.GetWindowThreadProcessId(handle,ctypes.byref(owner))
        if owner.value==pid:
            title=ctypes.create_unicode_buffer(256);user.GetWindowTextW(handle,title,256)
            if title.value.startswith('PoreSAM'):found.append((hwnd,title.value,bool(user.IsWindowVisible(handle))))
        return True
    user.EnumWindows(callback(collect),0)
    return found


def close(process):
    for hwnd,_,_ in windows(process.pid):user.PostMessageW(ctypes.c_void_p(hwnd),0x10,0,0)
    process.wait(timeout=20)
    assert process.returncode==0,process.returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path);args=parser.parse_args()
    run=ROOT/'outputs/desktop_checks'/('windows_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
    base=run/'data';second=run/'data_workspaces/workspace_0002/outputs';processes=[]
    def launch(directory):
        command=[str(args.bundle.resolve()/'PoreSAM.exe')] if args.bundle else [sys.executable,str(ROOT/'scripts/pore_app.py')]
        process=subprocess.Popen(command+['--data-dir',str(base)],creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append(process);deadline=time.monotonic()+80;log=directory.parent/'logs/desktop.log'
        while time.monotonic()<deadline:
            assert process.poll() is None,'App exited early'
            matches=re.findall(r'port=(\d+)',log.read_text(encoding='utf-8')) if log.exists() else []
            if matches and any(visible for _,_,visible in windows(process.pid)):
                port=matches[-1]
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/images',timeout=1):pass
                    return process,port
                except OSError:pass
            time.sleep(.2)
        raise AssertionError('Visible window did not open')
    def images(port):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/images',timeout=5) as response:return json.load(response)['images']
    def upload(port):
        stream=io.BytesIO();Image.new('L',(160,160),140).save(stream,format='PNG')
        request=urllib.request.Request(f'http://127.0.0.1:{port}/api/upload',data=json.dumps(dict(name='window-test.png',content=base64.b64encode(stream.getvalue()).decode())).encode(),headers={'Content-Type':'application/json','X-Pore-Editor':'1'})
        with urllib.request.urlopen(request,timeout=10) as response:return json.load(response)
    try:
        one,port1=launch(base);upload(port1)
        two,port2=launch(second)
        assert port1!=port2 and len(images(port1))==1 and not images(port2)
        assert any(title=='PoreSAM - Workspace 2' and visible for _,title,visible in windows(two.pid))
        upload(port2);assert len(images(port1))==len(images(port2))==1
        assert (run/'webview').exists() and (second.parent/'webview').exists()
        close(one)
        three,port3=launch(base)
        assert port3!=port2 and len(images(port3))==1 and len(images(port2))==1
        assert any(title=='PoreSAM' and visible for _,title,visible in windows(three.pid))
        result=dict(visible_windows=True,independent_libraries=True,independent_profiles=True,workspace_reused=True,ports=[port1,port2,port3])
    finally:
        for process in processes:
            if process.poll() is None:close(process)
    (run/'verification.json').write_text(json.dumps(result,indent=2));print(run,flush=True)


if __name__=='__main__':main()
