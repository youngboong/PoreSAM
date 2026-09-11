"""Browser regression: failed image-list responses cannot poison editor state."""
import json
import threading
import time
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright
from pore_editor import Editor, ROOT, make_handler


def main():
    run=ROOT/'outputs/workspace_checks'/('library_recovery_'+time.strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(Editor(run/'edits',run/'projects')))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    fault={'kind':'http'}
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1480,height=1080))
            page.on('pageerror',lambda error:errors.append(str(error)))
            def respond(route):
                kind=fault['kind']
                if kind=='ok': return route.continue_()
                body={'http':'{"error":"Not found"}','missing':'{}','entry':'{"images":[{}]}','json':'not JSON'}[kind]
                route.fulfill(status=404 if kind=='http' else 200,content_type='application/json',body=body)
            page.route('**/api/images',respond)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('state !== null && !busy && document.getElementById("libraryStatus").textContent.length > 0')
            assert page.evaluate('Array.isArray(imageLibrary) && imageLibrary.length === 0')
            assert page.locator('#retryLibrary').is_visible()
            fault['kind']='ok'
            page.locator('#retryLibrary').click()
            page.wait_for_function('!busy && imageLibrary.length === 3 && document.getElementById("libraryStatus").textContent === ""')
            original=page.evaluate('JSON.stringify(imageLibrary)')
            for kind in ['missing','entry','json','http']:
                fault['kind']=kind
                page.evaluate('window.onEditorAccepted(state)')
                page.wait_for_function('document.getElementById("libraryStatus").textContent.length > 0')
                assert page.evaluate('JSON.stringify(imageLibrary)')==original
                assert page.locator('#imageLibrary .image-card').count()==3
                # The exact find() caller remains usable after a failed background request.
                page.evaluate('updateHistory(state.dataset)')
                fault['kind']='ok'
                page.locator('#retryLibrary').click()
                page.wait_for_function('!busy && document.getElementById("libraryStatus").textContent === ""')
            assert not errors,errors
            page.screenshot(path=str(run/'recovered.png'),full_page=True)
            result=dict(status='passed',checks=['HTTP failure on startup','malformed JSON','missing image list','invalid list entry','preserve existing list','retry recovery','history remains usable'],javascript_errors=errors)
            (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(dict(result,output=str(run))),flush=True)
            browser.close()
    finally: server.shutdown()


if __name__=='__main__': main()
