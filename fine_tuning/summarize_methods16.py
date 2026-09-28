"""Combine full accuracy comparison and measured CPU-subset timing."""
import argparse
import json
from pathlib import Path
from train import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    gpu=json.loads((run/'methods_gpu/summary.json').read_text())
    cpu=json.loads((run/'methods_cpu/summary.json').read_text())
    assert gpu['completed_images']==16 and cpu['completed_images']==3
    verified=json.loads((run/'comparison_verification.json').read_text());assert verified['passed']
    comparison=dict(full_16_image_evaluation=gpu,cpu_3_image_measurement=cpu,
        gpu_two_stage_extra_seconds=gpu['timing']['extra_two_stage_seconds']['mean'],
        cpu_two_stage_extra_seconds=cpu['timing']['extra_two_stage_seconds']['mean'],
        cpu_two_stage_to_relaxed_time_ratio=cpu['timing']['two_stage_analysis_seconds']['mean']/cpu['timing']['relaxed_analysis_seconds']['mean'],
        verification=verified,app_model_replaced=False)
    dump(run/'summary.json',comparison)
    def accuracy_table(summary):
        table='<table><tr><th>Method</th><th>Area IoU ↑</th><th>Dice ↑</th><th>Missed area % ↓</th><th>Extra area % ↓</th><th>Total-area error % ↓</th></tr>'
        for v in ['relaxed','two_stage']:
            m=summary['macro_image_means'][v]
            table+=f'<tr><td>{v}</td><td>{m["iou"]:.4f}</td><td>{m["dice"]:.4f}</td><td>{m["missed_area_pct"]:.2f}</td><td>{m["extra_area_pct"]:.2f}</td><td>{m["relative_area_error_pct"]:.2f}</td></tr>'
        return table+'</table>'
    html='''<!doctype html><meta charset="utf-8"><title>PoreSAM — 16-image method comparison</title>
<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:20px;color:#233044}td,th{padding:10px;border-bottom:1px solid #ddd}a{color:#1262a4}li{margin:12px 0}</style>
<h1>Relaxed only vs Two-stage — 16-image comparison</h1>
<p>16 footer-corrected images, 2,018 reference pores. Each model trains on the other 15 images for 300 updates; five verified clean models reused and eleven trained from original SAM. No test image is used to optimize its corresponding model. Threshold rules were fixed before this comparison.</p>
<h2>Area accuracy: all 16 images</h2>'''+accuracy_table(gpu)
    w=gpu['paired_image_wins']['iou']
    html+=f'<p>Image-wise IoU wins: Relaxed only {w["relaxed"]}; Two-stage {w["two_stage"]}; ties {w["ties"]}. Means weight each image equally.</p>'
    html+='<p><a href="methods_gpu/index.html">All 16 image comparisons, family means, native error maps and GPU timings</a></p>'
    html+='<h2>CPU timing: three preselected images</h2><p>Folds 1, 7 and 15: dense PI100, difficult PI300 and complex top. Actual CPU float32 inference, 8 threads, batch 8, Detailed 48×48 grid. No concurrent training or benchmark. These are three-image measurements, not a 16-image CPU mean.</p><table><tr><th>CPU stage</th><th>Mean seconds/image</th></tr>'
    for k in ['sam_generation_seconds','relaxed_post_seconds','two_stage_post_seconds','relaxed_analysis_seconds','two_stage_analysis_seconds','extra_two_stage_seconds']:
        html+=f'<tr><td>{k}</td><td>{cpu["timing"][k]["mean"]:.2f}</td></tr>'
    html+='</table><p>Shared SAM is run once per image. Each complete method-specific postprocessing path is measured twice, alternating order. Reported post time is the median; analysis time adds shared SAM generation. Training, model loading, preprocessing, report rendering and disk I/O are excluded. This measures inference/selection, not total GUI wall time. Generation timings have one repetition per image.</p>'
    html+=f'<p>CPU Two-stage / Relaxed time ratio: {comparison["cpu_two_stage_to_relaxed_time_ratio"]:.3f}×. Shared generation is not charged twice to Two-stage.</p>'
    html+='<p><a href="methods_cpu/index.html">CPU per-image timings and separate accuracy/error maps</a></p>'
    html+='<h2>Interpretation and limits</h2><p>Area IoU compares the spatial union of predicted and reference pore pixels. Missed area = reference pixels not found / reference area. Extra area = predicted pixels outside the reference / reference area. Total-area error = absolute predicted-minus-reference area / reference area; it may hide cancellation of misses and extras.</p><p>Relaxed only filters a larger low-threshold candidate pool with the original pore rules. Two-stage retains strict detections and adds candidates with boundary evidence and limited overlap. Both use identical saved per-image minimum-area and contrast settings. These are previously examined related acquisitions, not independent external specimens. CPU float32 and GPU bfloat16 can differ, so CPU subset metrics are not pooled with the 16-image GPU evaluation. No application default or checkpoint has been replaced.</p>'
    exceptions=[];strict_better=[]
    for fold in range(1,17):
        rows=json.loads((run/f'methods_gpu/fold_{fold}/result.json').read_text())['metrics']
        scores={r['variant']:r['iou'] for r in rows};name=rows[0]['image']
        if scores['relaxed']>scores['two_stage']:exceptions.append(name)
        if scores['strict']>max(scores['relaxed'],scores['two_stage']):strict_better.append(name)
    html+='<p><strong>Exceptions:</strong> Relaxed-only has higher IoU on '+', '.join(exceptions)+'. Both expanded-candidate methods have lower IoU than strict selection on '+', '.join(strict_better)+'. In particular, top_2_B still has substantial excess predicted area; footer correction has not solved its pore-definition/selection problem.</p>'
    html+='<ul><li><a href="dataset_verification.json">Footer-corrected dataset audit</a></li><li><a href="comparison_verification.json">Model, area metric, error-map and timing verification</a></li><li><a href="methods_gpu/metrics.csv">16-image area metrics CSV</a></li><li><a href="methods_gpu/timing.csv">GPU timing CSV</a></li><li><a href="methods_cpu/timing.csv">CPU timing CSV</a></li></ul>'
    (run/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(gpu_mean=gpu['macro_image_means'],gpu_wins=w,cpu_times=cpu['timing'],cpu_time_ratio=comparison['cpu_two_stage_to_relaxed_time_ratio']),indent=2))


if __name__=='__main__':main()
