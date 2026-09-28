"""Paired five-image preprocessing ablation; fixed held-out models and selection."""
import csv
import gc
import json
from datetime import datetime
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from sam2.build_sam import build_sam2
from train import ROOT, CONFIG, sha, dump
from evaluate_nested import metrics, mask_overlay, errors
sys.path.insert(0, str(ROOT/'scripts'))
from pore_preprocessing import prepare_adjustable
from pore_selection import TraceGenerator, select, label_map


def main():
    source = ROOT/'fine_tuning/runs/clean16_methods_20260921_110028'
    training = json.loads((source/'plan.json').read_text())
    data = Path(training['dataset'])
    assert sha(data/'manifest.json') == training['dataset_manifest_sha256']
    records = json.loads((data/'manifest.json').read_text())['records']
    out = ROOT/'fine_tuning/runs'/('preprocessing5_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    variants = {'normalize': (0, 'none'), 'background': (2, 'none'),
                'blur': (0, 'gaussian'), 'background_blur': (2, 'gaussian')}
    folds = [1, 7, 10, 15, 16]
    rng = random.Random(260921)
    schedule = {}
    for fold in folds:
        order = list(variants)
        rng.shuffle(order)
        schedule[fold] = order
    plan = dict(folds=folds, variants=variants, blur_strength=2, seed=260921,
                schedule=schedule, selection='two_stage_nested', source=str(source),
                normalized_input=True, normalization_reapplied=False, retrained=False,
                points_per_side=48, points_per_batch=4, dtype='bfloat16',
                primary_metric='image-wise area IoU', secondary_metric='instance F1 at IoU 0.5',
                limitations='Five purposively selected previously inspected images; related acquisitions can be in training. Fixed normalization-trained LOO models. One background strength and Gaussian blur strength only; not independent validation or a filter/strength optimization.',
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'scripts/pore_preprocessing.py', ROOT/'scripts/pore_selection.py', ROOT/'scripts/pore_nested.py', ROOT/'scripts/pore_conservative.py']})
    dump(out/'plan.json', plan)
    print('OUTPUT '+str(out), flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    rows = []
    for fold in folds:
        name = training['folds'][fold-1]['validation'][0]
        r = next(r for r in records if r['name'] == name)
        ep = r['evaluation_preprocessing']
        assert ep['normalize_enabled'] and ep['background_strength'] == 0 and ep['blur_method'] == 'none'
        config = json.loads((source/f'fold_{fold}/config.json').read_text())
        assert name not in config['train_images'] and config['validation_images'] == [name]
        checkpoint = Path(training['model_directory'])/f'fold_{fold}.pt'
        expected = json.loads((source/f'fold_{fold}/model_verification.json').read_text())['checkpoint_sha256']
        assert sha(checkpoint) == expected and sha(data/r['image']) == r['image_sha256']
        gray = np.array(Image.open(data/r['image']).convert('L'))
        folder = out/f'fold_{fold}'
        folder.mkdir()
        dump(folder/'provenance.json', dict(image=name, checkpoint=str(checkpoint), checkpoint_sha256=expected,
             image_sha256=r['image_sha256'], labels_sha256=r['labels_sha256'], train_images=config['train_images']))
        model = build_sam2(CONFIG, str(checkpoint), device='cuda', apply_postprocessing=False)
        predictions, inputs, durations = {}, {}, {}
        for variant in schedule[fold]:
            background, method = variants[variant]
            pixels, meta = prepare_adjustable(gray, dict(normalize_enabled=False, background_strength=background,
                                                        blur_method=method, blur_strength=2))
            inputs[variant] = pixels
            Image.fromarray(pixels).save(folder/f'{variant}_input.png')
            print(f'START {name} {variant}', flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generator = TraceGenerator(model, points_per_side=48, points_per_batch=4,
                    pred_iou_thresh=.7, stability_score_thresh=.85, crop_n_layers=0,
                    min_mask_region_area=0, output_mode='uncompressed_rle')
                generator.generate(np.repeat(pixels[:,:,None], 3, axis=2))
            result = select(generator.pool, pixels, r['source_preprocessing'].get('min_area_pixels',100),
                            r['source_preprocessing'].get('min_contrast',8), 'cuda', 'two_stage_nested')
            predictions[variant] = label_map(result['masks'], gray.shape)
            torch.cuda.synchronize()
            durations[variant] = time.perf_counter()-start
            Image.fromarray(predictions[variant]).save(folder/f'{variant}_instances.tif')
            del generator, result
            print(f'DONE {name} {variant} {durations[variant]:.1f}s', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        assert sha(data/r['labels']) == r['labels_sha256']
        truth = np.array(Image.open(data/r['labels']))
        local = [dict(fold=fold, image=name, variant=v, seconds=durations[v], **metrics(truth,predictions[v])) for v in variants]
        rows.extend(local)
        dump(folder/'result.json', local)
        sheet = Image.new('RGB', (4*512, 3*414), 'white')
        draw = ImageDraw.Draw(sheet)
        for col, v in enumerate(variants):
            m = next(m for m in local if m['variant']==v)
            panels = [np.repeat(inputs[v][:,:,None],3,axis=2), mask_overlay(gray,predictions[v]), errors(gray,truth,predictions[v])]
            for row, panel in enumerate(panels):
                sheet.paste(Image.fromarray(panel).resize((512,384)),(col*512,row*414+30))
            draw.text((col*512+8,8),f'{v} | IoU {m["iou"]:.4f} | F1 {m["instance_f1"]:.4f}',fill='black')
        sheet.save(folder/'comparison.jpg', quality=94)
        print('METRICS '+json.dumps(local), flush=True)
    fields = ['iou','instance_f1','missed_area_pct','extra_area_pct','relative_area_error_pct']
    means = {v:{k:float(np.mean([r[k] for r in rows if r['variant']==v])) for k in fields} for v in variants}
    deltas = {}
    for v in variants:
        ds = [next(r['iou'] for r in rows if r['fold']==f and r['variant']==v)-next(r['iou'] for r in rows if r['fold']==f and r['variant']=='normalize') for f in folds]
        deltas[v] = dict(per_image=ds, mean=float(np.mean(ds)), improved=sum(d>1e-10 for d in ds), worse=sum(d< -1e-10 for d in ds))
    dump(out/'summary.json',dict(image_count=5, means=means, area_iou_deltas=deltas, limitations=plan['limitations']))
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    page = '<!doctype html><meta charset="utf-8"><title>5-image preprocessing test</title><style>body{font:16px system-ui;margin:30px}td,th{padding:10px;border-bottom:1px solid #aaa}img{width:100%}</style><h1>5-image preprocessing test</h1><p>All conditions include the same saved percentile normalization. Background: morphological opening radius 2 px. Blur: Gaussian strength 2 (sigma 1 px). Fixed two_stage_nested and held-out model per image.</p><p>'+plan['limitations']+'</p><table><tr><th>Condition</th>'+''.join('<th>'+k+'</th>' for k in fields)+'</tr>'
    for v in variants:
        page += '<tr><td>'+v+'</td>'+''.join(f'<td>{means[v][k]:.4f}</td>' for k in fields)+'</tr>'
    page += '</table><p>IoU and F1: higher is better. Error percentages: lower is better. Equal image weights. Cyan: missed reference; magenta: extra prediction. Rows: processed input, prediction overlay, error map.</p><p><a href="metrics.csv">Per-image CSV</a> | <a href="summary.json">Summary</a> | <a href="plan.json">Plan</a></p>'
    for fold in folds:
        name = next(r['image'] for r in rows if r['fold']==fold)
        page += f'<h2>{name}</h2><a href="fold_{fold}/comparison.jpg"><img src="fold_{fold}/comparison.jpg"></a>'
    (out/'index.html').write_text(page,encoding='utf-8')
    print('COMPLETE '+str(out/'index.html'),flush=True)


if __name__ == '__main__':
    main()
