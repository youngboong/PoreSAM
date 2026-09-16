"""Audit repeated packaged Automate on isolated PI/GF snapshots; no accuracy claims."""
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from check_desktop_windows import close

ROOT = Path(__file__).resolve().parents[1]


def audit_manual_cut():
    """Isolate postprocessing from SAM with a deterministic joint-mask fixture."""
    from automate_pores import add_candidate, consolidation_boxes
    yy, xx = np.mgrid[:240, :240]
    full = (xx-120)**2 + (yy-120)**2 < 40**2
    masks = {1: full & (xx < 119), 2: full & (xx > 121)}
    gray = np.full(full.shape, 190, np.uint8)
    gray[full] = 30
    state = dict(gray=gray, masks=masks, report={}, revision=0, next_id=3,
                 annotations={str(i): dict(source='manual_cut') for i in masks})
    def preview(state, payload):
        state['preview'] = dict(masks=[full.copy()])
        return dict(choices=[dict(score=.99)])
    editor = SimpleNamespace(check_revision=lambda *args: None, preview=preview)
    groups = consolidation_boxes(state)
    response = add_candidate(editor, state, dict(revision=0, box=[75, 75, 165, 165]), staged=True)
    return dict(test='Manual cut preservation in normal Automate proposal', synthetic_sam_mask=True,
                consolidation_groups=groups, before_count=2, after_count=len(state['masks']),
                response=response, manual_cut_preserved=len(state['masks']) == 2)


def fingerprint(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def load_masks(folder, revision):
    with np.load(folder / f'revision_{revision:04d}/entrance_candidates.npz') as archive:
        return {int(k.rsplit('_', 1)[1]): archive[k].astype(bool) for k in archive.files}


def draw(gray, masks, path, changed=()):
    canvas = np.repeat(gray[:, :, None], 3, axis=2)
    for key, mask in masks.items():
        color = np.array([0, 220, 255] if key in changed else [255, 215, 0])
        canvas[mask] = (canvas[mask] * .55 + color * .45).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(map(int, color)), 1)
        distance = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 3)
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        cv2.putText(canvas, str(key), (x-7, y+4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3)
        cv2.putText(canvas, str(key), (x-7, y+4), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    Image.fromarray(canvas).save(path)


def run_case(run, data, dataset, name):
    case = run / name
    case.mkdir()
    project_id = dataset.split('__run_')[0]
    source = data / 'manual_edits' / dataset
    project = data / 'projects' / project_id
    original = fingerprint(source), fingerprint(project)
    shutil.copytree(project, case / 'data/projects' / project_id)
    copied = case / 'data/manual_edits' / dataset
    shutil.copytree(source, copied)
    project_file = case / 'data/projects' / project_id / 'project.json'
    project_info = json.loads(project_file.read_text(encoding='utf-8'))
    project_info['runs'] = [r for r in project_info['runs'] if r['dataset'] == dataset]
    project_file.write_text(json.dumps(project_info), encoding='utf-8')
    revision = json.loads((source / 'latest.json').read_text())['revision']
    report = json.loads((source / f'revision_{revision:04d}/report.json').read_text())
    old = load_masks(source, revision)
    gray = np.asarray(Image.open(report['image']).convert('L'))[:next(iter(old.values())).shape[0]]
    Image.fromarray(gray).save(case / 'original.png')
    draw(gray, old, case / 'before.png')
    result = dict(name=name, dataset=dataset, source_revision=revision,
                  initial_count=len(old), preprocessing=project_info['runs'][0]['config'],
                  selection_settings=report.get('selection_settings'), passes=[], errors=[])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = f'--remote-debugging-port={port}'
    process = subprocess.Popen([str(ROOT / 'dist/PoreSAM/PoreSAM.exe'), '--data-dir', str(case / 'data')], env=env)
    try:
        deadline = time.monotonic()+90
        while time.monotonic() < deadline:
            assert process.poll() is None, 'App exited'
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version', timeout=1).close()
                break
            except OSError:
                time.sleep(.25)
        else:
            raise AssertionError('App startup timeout')
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
            page = browser.contexts[0].pages[0]
            page.on('pageerror', lambda e: result['errors'].append(str(e)))
            page.wait_for_function("typeof busy!=='undefined'&&!busy")
            page.locator('#openAnalysisTab').click()
            page.locator('#imageLibrary .image-card').first.click()
            page.wait_for_function('state&&!busy')
            assert page.evaluate('state.dataset') == dataset
            for number in range(1, 4):
                started = time.monotonic()
                page.locator('#automatePores').click()
                page.wait_for_function("!busy&&document.getElementById('status').textContent.includes('Automate complete')", timeout=300000)
                elapsed = time.monotonic()-started
                current = page.evaluate('({revision:state.revision,count:state.candidates.length,stats:state.stats})')
                new = load_masks(copied, current['revision'])
                shape = gray.shape
                union = lambda masks: np.logical_or.reduce(list(masks.values())) if masks else np.zeros(shape, bool)
                old_union, new_union = union(old), union(new)
                added = sorted(set(new)-set(old))
                removed = sorted(set(old)-set(new))
                changed = sorted(i for i in new if i in old and not np.array_equal(new[i], old[i]))
                overlap = int((np.sum(list(new.values()), axis=0)>1).sum()) if new else 0
                stats = current['stats']
                checks = dict(no_overlap=overlap == 0,
                              count_matches=current['count'] == len(new) == stats['candidate_count'],
                              area_matches=bool(abs(stats['candidate_union_area_percent']-new_union.mean()*100)<1e-7),
                              no_review_queue=page.evaluate('queuedRegions.length') == 0)
                metadata = json.loads((copied / f"revision_{current['revision']:04d}/edit.json").read_text())
                mergers = {str(i): metadata['annotations'].get(str(i), {}) for i in changed}
                record = dict(pass_number=number, revision=current['revision'], count=len(new),
                              added=added, removed=removed, changed=changed,
                              gained_pixels=int((new_union & ~old_union).sum()),
                              lost_pixels=int((old_union & ~new_union).sum()),
                              overlap_pixels=overlap, elapsed_seconds=round(elapsed, 2),
                              unchanged=not (added or removed or changed), checks=checks, mergers=mergers)
                result['passes'].append(record)
                draw(gray, new, case / f'pass_{number}.png', added+changed)
                diff = np.repeat(gray[:, :, None], 3, axis=2)
                diff[new_union & ~old_union] = [0, 220, 255]
                diff[old_union & ~new_union] = [255, 50, 100]
                Image.fromarray(diff).save(case / f'changes_{number}.png')
                old = new
                (case / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(name, number, {k: v for k, v in record.items() if k!='mergers'}, flush=True)
            page.evaluate('window.clearRegionQueue()')
            close(process)
    except Exception as exc:
        result['errors'].append(repr(exc))
    finally:
        if process.poll() is None:
            close(process)
        result['user_data_unchanged'] = original == (fingerprint(source), fingerprint(project))
        result['structural_checks_passed'] = (len(result['passes']) == 3 and not result['errors'] and
            result['user_data_unchanged'] and all(all(p['checks'].values()) for p in result['passes']))
        (case / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    run = ROOT / 'outputs/automate_validation' / time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    (run / 'manual_cut_check.json').write_text(json.dumps(audit_manual_cut(), indent=2), encoding='utf-8')
    local = Path(os.environ['LOCALAPPDATA']) / 'PoreSAM/English/outputs'
    cases = [(local, 'image_f0db2febd56b3364__run_0001', 'PI35_5kx_BSE'),
             (local, 'image_92d06ce0d584971c__run_0003', 'PI35_2kx_BSE8'),
             (local, 'image_0325034194c74826__run_0001', 'PI100_2kx_BSE'),
             (local, 'image_70c824c01df7df1b__run_0002', 'GF_1kx_BSE'),
             (ROOT / 'outputs', 'image_5d8e6a257fb13137__run_0002', 'GF_1kx_SE')]
    results = []
    print('RUN', run, flush=True)
    for data, dataset, name in cases:
        results.append(run_case(run, data, dataset, name))
        (run / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    body = ['<h1>Automate validation</h1><p>Yellow: existing masks. Cyan: added or changed masks in each pass. '
            'Changes images: cyan = gained pixels, pink = lost pixels. No ground-truth masks; structural checks are not accuracy scores.</p>']
    for result in results:
        name = result['name']
        body.append(f'<h2>{html.escape(name)}</h2><p>Initial count: {result["initial_count"]}; '
                    f'structural checks: {result["structural_checks_passed"]}</p>')
        for file, title in [('original.png', 'Original'), ('before.png', 'Before')] + [
            (f'pass_{i}.png', f'Automate {i}') for i in range(1, 4)]:
            body.append(f'<figure><a href="{name}/{file}"><img src="{name}/{file}"></a><figcaption>{title}</figcaption></figure>')
        body.append('<pre>'+html.escape(json.dumps([{k:v for k,v in p.items() if k!='mergers'} for p in result['passes']], indent=2))+'</pre>')
    (run / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Automate validation</title>'
        '<style>body{font:15px Segoe UI,sans-serif;background:#f1f5f9;color:#172033;margin:24px}figure{display:inline-block;width:46%;margin:1%}img{width:100%}pre{background:white;padding:16px;overflow:auto}</style>'+''.join(body), encoding='utf-8')
    print('COMPLETE', run, flush=True)


if __name__ == '__main__':
    main()
