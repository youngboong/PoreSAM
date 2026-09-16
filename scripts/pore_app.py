"""PoreSAM desktop entry point (Windows / WebView2)."""
import argparse
import ctypes
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import traceback


def default_data_dir():
    return Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData/Local'))/'PoreSAM/English/outputs'


def acquire_instance(directory):
    if os.name!='nt':return None
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p]
    api.CreateMutexW.restype=ctypes.c_void_p
    name='Local\\PoreSAMEnglish_'+hashlib.sha256(str(directory).lower().encode()).hexdigest()[:24]
    handle=api.CreateMutexW(None,False,name)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error()==183:
        api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle(handle)
        return False
    return handle


def acquire_workspace(base):
    """Reserve an independent persistent workspace for each concurrently open app."""
    number=1
    while True:
        directory=base if number==1 else base.parent/(base.name+'_workspaces')/f'workspace_{number:04d}'/'outputs'
        handle=acquire_instance(directory)
        if handle is not False:return directory,handle,number
        number+=1


def main():
    parser=argparse.ArgumentParser(description='PoreSAM for Windows')
    parser.add_argument('--device',choices=['cpu','auto','cuda'],default='cpu')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir())
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--smoke-test',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    directory,mutex,workspace_number=acquire_workspace(args.data_dir.expanduser().resolve())
    directory.mkdir(parents=True,exist_ok=True)
    os.environ['PORESAM_DATA_DIR']=str(directory)
    os.environ.setdefault('MPLCONFIGDIR',str(directory/'.matplotlib'))
    if args.checkpoint:os.environ['PORESAM_CHECKPOINT']=str(args.checkpoint.resolve())
    logs=directory.parent/'logs';logs.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[RotatingFileHandler(logs/'desktop.log',maxBytes=2_000_000,backupCount=3,encoding='utf-8')])
    # PyInstaller's windowed build has no console streams.
    if sys.stdout is None:sys.stdout=open(logs/'console.log','a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=sys.stdout
    server=None;worker=None
    try:
        import webview
        # CSV and plot exports use browser downloads; WebView2 opens Save As.
        webview.settings['ALLOW_DOWNLOADS']=True
        from pore_editor import Editor,LocalPoreServer,make_handler
        from app_paths import checkpoint_path
        import torch
        logging.info('Desktop runtime: torch=%s cuda=%s',torch.__version__,torch.version.cuda)
        if not checkpoint_path().is_file():raise FileNotFoundError('SAM model missing: '+str(checkpoint_path()))
        editor=Editor(device=args.device)
        server=LocalPoreServer(('127.0.0.1',0),make_handler(editor))
        worker=threading.Thread(target=server.serve_forever,name='PoreSAM server',daemon=True);worker.start()
        title='PoreSAM' if workspace_number==1 else f'PoreSAM - Workspace {workspace_number}'
        from desktop_details import DesktopDetails
        details=DesktopDetails()
        base_url=f'http://127.0.0.1:{server.server_port}'
        window=webview.create_window(title,base_url,width=1440,height=960,min_size=(1024,720),text_select=True,background_color='#edf2f3',js_api=details)
        details._main=window;details._url=base_url;details._title=title+' - Pore Details'
        window.events.closed+=details.close_details
        def picker():
            result=window.create_file_dialog(webview.FileDialog.FOLDER)
            return result[0] if result else ''
        editor.folder_picker=picker
        close_state={'approved':False,'checking':False}
        def check_close():
            # Run outside the native closing event: evaluate_js needs the UI thread.
            try:
                editing=window.evaluate_js("typeof busy!=='undefined' && busy")
                if editor.workflow.running or editing:
                    window.evaluate_js("setStatus('Wait for the current operation to finish before closing.',true)")
                    return
                draft=window.evaluate_js("typeof points!=='undefined' && Boolean(points.length||box||shape||polygon.length||cutPath.length||cutPaths.length||queuedRegions.some(r=>r.status!=='Added'))")
                if draft and not window.create_confirmation_dialog('Close PoreSAM','Discard pending drawing and previews? Saved edits are kept.'):return
            except Exception:logging.exception('Close check failed')
            finally:close_state['checking']=False
            close_state['approved']=True
            details.close_details()
            window.destroy()
        def closing():
            if args.smoke_test or close_state['approved']:return True
            if not close_state['checking']:
                close_state['checking']=True
                threading.Thread(target=check_close,name='PoreSAM close check',daemon=True).start()
            return False
        window.events.closing+=closing
        def smoke():
            try:
                import time
                for _ in range(120):
                    if window.evaluate_js("document.readyState==='complete' && typeof busy!=='undefined' && !busy && typeof imageLibrary!=='undefined'"):break
                    time.sleep(.25)
                else:raise RuntimeError('Desktop UI did not become ready')
                info=window.evaluate_js("({title:document.title,load:document.getElementById('newImageTab').getAttribute('aria-selected'),steps:document.querySelector('.steps').innerText})")
                assert info['load']=='true',info
                info.update(device=editor.device,data_dir=str(directory),port=server.server_port,workspace=workspace_number)
                args.smoke_test.parent.mkdir(parents=True,exist_ok=True)
                args.smoke_test.write_text(json.dumps(info,indent=2),encoding='utf-8')
            except Exception:
                logging.exception('Desktop smoke test failed')
                args.smoke_test.with_suffix('.error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            finally:window.destroy()
        if args.smoke_test:window.events.loaded+=smoke
        logging.info('Desktop started: data=%s device=%s port=%s',directory,editor.device,server.server_port)
        webview.start(gui='edgechromium',debug=args.debug,private_mode=False,storage_path=str(directory.parent/'webview'))
    except Exception:
        logging.exception('Desktop startup failed')
        if args.smoke_test:
            args.smoke_test.parent.mkdir(parents=True,exist_ok=True);args.smoke_test.with_suffix('.error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        elif os.name=='nt':ctypes.windll.user32.MessageBoxW(None,'PoreSAM could not start.\nInstall Microsoft Edge WebView2 Runtime and check the SAM model.\n\nDetails: '+str(logs/'desktop.log'),'PoreSAM',0x10)
        else:raise
        return 1
    finally:
        if server:server.shutdown();server.server_close()
        if worker:worker.join(timeout=5)
        if mutex:
            api=ctypes.WinDLL('kernel32',use_last_error=True);api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle(mutex)
        logging.info('Desktop stopped')
    return 0


if __name__=='__main__':
    multiprocessing.freeze_support()
    exit_code=main() or 0
    if os.name=='nt':
        # main() has closed the window/server and released the instance mutex.
        # Python.NET CLR finalization hangs (or crashes if its unload is skipped)
        # on this Windows stack. Flush our files, then let Windows reclaim it.
        logging.shutdown()
        for stream in (sys.stdout,sys.stderr):
            if stream:stream.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
