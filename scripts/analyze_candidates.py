"""Descriptive 2D candidate measurements. Not validated porosity or pore counts."""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import perimeter_crofton
from result_paths import read_artifact


def measure_masks(masks, shape, um_per_pixel, grid=(3, 4)):
    h, w = shape
    union = np.zeros(shape, dtype=bool)
    shared = np.zeros(shape,dtype=bool)
    rows = []
    counts = np.zeros(grid, dtype=int)
    complete_counts = np.zeros(grid, dtype=int)
    for candidate_id, mask in masks:
        if mask.shape != shape or mask.dtype != np.bool_ or not mask.any():
            raise ValueError(f"Invalid candidate mask {candidate_id}")
        shared |= union & mask
        union |= mask
        yy, xx = np.nonzero(mask)
        area = int(mask.sum())
        # Crop before estimating perimeter; Crofton pads the crop with background.
        cropped=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
        perimeter=float(perimeter_crofton(cropped,directions=4))
        circularity_raw=float(4*np.pi*area/perimeter**2) if perimeter>0 else 0.
        circularity=float(np.clip(circularity_raw,0,1))
        edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        cy, cx = float(yy.mean()), float(xx.mean())
        dx,dy=xx-cx,yy-cy
        eigenvalues=np.linalg.eigvalsh([[np.mean(dx*dx),np.mean(dx*dy)],[np.mean(dx*dy),np.mean(dy*dy)]])
        minor,major=4*np.sqrt(np.maximum(eigenvalues,0))
        length_um,width_um=float(major*um_per_pixel),float(minor*um_per_pixel)
        aspect_ratio=float(major/minor) if minor>0 else None
        roundness_raw=float(4*area/(np.pi*major**2)) if major>0 else None
        roundness=float(np.clip(roundness_raw,0,1)) if roundness_raw is not None else None
        gy, gx = min(int(cy / h * grid[0]), grid[0]-1), min(int(cx / w * grid[1]), grid[1]-1)
        counts[gy, gx] += 1
        complete_counts[gy, gx] += int(not edge)
        rows.append(dict(candidate_id=candidate_id, area_pixels=area, area_um2=area*um_per_pixel**2,
                         equivalent_diameter_um=2*np.sqrt(area*um_per_pixel**2/np.pi),
                         perimeter_um=perimeter*um_per_pixel,circularity=circularity,circularity_raw=circularity_raw,
                         length_um=length_um,width_um=width_um,aspect_ratio=aspect_ratio,roundness=roundness,roundness_raw=roundness_raw,
                         touches_image_edge=edge, included_in_size_distribution=not edge,
                         centroid_x_pixels=cx, centroid_y_pixels=cy,
                         centroid_x_um=cx*um_per_pixel, centroid_y_um=cy*um_per_pixel,
                         grid_row=gy+1, grid_column=gx+1, review_status="unreviewed"))
    columns = ["candidate_id", "area_pixels", "area_um2", "equivalent_diameter_um", "perimeter_um", "circularity", "circularity_raw", "touches_image_edge",
               "length_um", "width_um", "aspect_ratio", "roundness", "roundness_raw",
               "included_in_size_distribution", "centroid_x_pixels", "centroid_y_pixels", "centroid_x_um",
               "centroid_y_um", "grid_row", "grid_column", "review_status"]
    frame = pd.DataFrame(rows, columns=columns)
    interior = frame[frame["included_in_size_distribution"].astype(bool)]
    analyzed_area = h*w*um_per_pixel**2
    union_area = int(union.sum())*um_per_pixel**2
    stats = dict(status="UNREVIEWED AUTOMATIC CANDIDATES; not validated pores or 3D porosity",
                 image_width_pixels=w, image_height_pixels=h, um_per_pixel=um_per_pixel,
                 field_width_um=w*um_per_pixel, field_height_um=h*um_per_pixel,
                 analyzed_area_um2=analyzed_area, candidate_count=len(frame),
                 complete_candidate_count=len(interior), edge_candidate_count=len(frame)-len(interior),
                 union_candidate_area_um2=union_area, candidate_union_area_percent=100*union_area/analyzed_area,
                 sum_individual_candidate_area_um2=float(frame["area_um2"].sum()),
                 overlap_pixels=int(shared.sum()),
                 overlap_excess_area_um2=(int(frame['area_pixels'].sum())-int(union.sum()))*um_per_pixel**2,
                 observed_candidates_per_10000_um2=len(frame)/analyzed_area*10000,
                 minimum_filter_area_pixels=100, minimum_filter_area_um2=100*um_per_pixel**2,
                 grid_rows=grid[0], grid_columns=grid[1], grid_count_rule="each centroid assigned to one cell",
                 size_distribution_rule="number weighted; image-edge candidates excluded; no stereological correction")
    for column in ["equivalent_diameter_um", "area_um2", "circularity", "length_um", "width_um", "aspect_ratio", "roundness"]:
        values = interior[column].dropna().to_numpy(dtype=float)
        stats[f"complete_{column}_mean"] = float(values.mean()) if len(values) else None
        stats[f"complete_{column}_median"] = float(np.median(values)) if len(values) else None
        stats[f"complete_{column}_std"] = float(values.std(ddof=1)) if len(values)>1 else None
    stats["diameter_definition"] = "equivalent circle diameter = 2*sqrt(projected area/pi), not a measured circular width"
    stats['circularity_definition']='4*pi*area/perimeter^2; 2D circularity, not 3D sphericity; clipped to [0,1] for pixel-estimation overshoot; unbounded value in circularity_raw'
    stats['perimeter_method']='skimage.measure.perimeter_crofton, 4 directions, including hole boundaries; visible mask only for image-boundary pores'
    stats['length_width_definition']='major/minor full axis lengths of ellipse with the same normalized second central moments; 4*sqrt(covariance eigenvalues); not bounding box or Feret diameters'
    stats['aspect_ratio_definition']='Length / Width; null when Width is zero'
    stats['roundness_definition']='4*area/(pi*major_axis_length^2), clipped to [0,1] for pixel-estimation overshoot; raw in roundness_raw; null when Length is zero; distinct from perimeter-based circularity'
    return frame, stats, union, counts, complete_counts


def histogram_table(values):
    if not len(values):
        return pd.DataFrame(columns=["bin_left", "bin_right", "count"])
    counts, edges = np.histogram(values, bins="auto")
    return pd.DataFrame(dict(bin_left=edges[:-1], bin_right=edges[1:], count=counts))


def export_folder(folder):
    cache = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    source = Image.open(report["image"]).convert("L")
    gray = np.asarray(source)[:report["analysis_bottom_exclusive"]]
    with np.load(read_artifact(folder, "entrance_candidates.npz"), allow_pickle=False) as data:
        masks = [(int(k.split("_")[-1]), data[k]) for k in sorted(data.files, key=lambda k:int(k.split("_")[-1]))]
    table, stats, union, counts, complete_counts = measure_masks(masks, gray.shape, report["scale"]["um_per_pixel"])
    if len(table) != report["entrance_candidate_count"]:
        raise ValueError("Mask count differs from segmentation report")
    stats["image"] = report["image"]
    stats["scale"] = report["scale"]
    selection = report.get("selection_settings", {})
    if selection:
        stats["selection_settings"] = selection
        stats["minimum_filter_area_pixels"] = selection.get("min_area_pixels", 100)
        stats["minimum_filter_area_um2"] = stats["minimum_filter_area_pixels"] * stats["um_per_pixel"]**2
    if "revision" in report:
        stats["revision"] = report["revision"]
        stats["status"] = "user edited candidates; not all objects reviewed"
        notes = report.get("candidate_annotations", {})
        table["review_status"] = table["candidate_id"].map(lambda i: notes.get(str(i), {}).get("review_status", "unreviewed"))
        table["source"] = table["candidate_id"].map(lambda i: notes.get(str(i), {}).get("source", "automatic"))
        table["revision"] = report["revision"]
    quality_path = folder / "quality_review.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    if quality:
        stats["visual_quality_review"] = quality
    out = folder / "measurements"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "candidates.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8")
    Image.fromarray(union.astype(np.uint8)*255).save(out / "union_mask.png")
    inner = table[table["included_in_size_distribution"].astype(bool)]
    d_hist = histogram_table(inner["equivalent_diameter_um"].to_numpy(dtype=float))
    a_hist = histogram_table(inner["area_um2"].to_numpy(dtype=float))
    d_hist.to_csv(out / "diameter_histogram.csv", index=False)
    a_hist.to_csv(out / "area_histogram.csv", index=False)
    yedges = np.linspace(0, gray.shape[0], counts.shape[0]+1)
    xedges = np.linspace(0, gray.shape[1], counts.shape[1]+1)
    spatial = []
    for r in range(counts.shape[0]):
        for c in range(counts.shape[1]):
            y0,y1 = int(round(yedges[r])), int(round(yedges[r+1]))
            x0,x1 = int(round(xedges[c])), int(round(xedges[c+1]))
            spatial.append(dict(row=r+1, column=c+1, candidate_count=int(counts[r,c]),
                                complete_candidate_count=int(complete_counts[r,c]),
                                cell_area_um2=(y1-y0)*(x1-x0)*stats["um_per_pixel"]**2,
                                union_area_percent=float(union[y0:y1,x0:x1].mean()*100)))
    pd.DataFrame(spatial).to_csv(out / "spatial_grid.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    result_label = f"수정본 {report['revision']} · 전체 검토 미완료" if "revision" in report else "자동 후보 통계 · 수동 검토 전"
    image_label = Path(report.get('source_name', report['image'])).stem
    fig.suptitle(f"{image_label} — {result_label}", fontsize=17)
    axes[0,0].axis("off")
    median = stats["complete_equivalent_diameter_um_median"]
    text = (f"분석 영역: {stats['field_width_um']:.2f} × {stats['field_height_um']:.2f} µm\n"
            f"분석 면적: {stats['analyzed_area_um2']:,.1f} µm²\n"
            f"후보 합집합 면적: {stats['union_candidate_area_um2']:,.1f} µm²\n"
            f"후보 면적률: {stats['candidate_union_area_percent']:.2f}%\n"
            f"관측 후보: {stats['candidate_count']}개 (이미지 경계 접촉 {stats['edge_candidate_count']}개)\n"
            f"크기 분포 대상: {stats['complete_candidate_count']}개\n"
            f"관측 개수 밀도: {stats['observed_candidates_per_10000_um2']:.2f}개 / 10,000 µm²\n")
    text += f"등가원직경 중앙값: {median:.2f} µm" if median is not None else "크기 분포 대상 없음"
    circularity_median=stats['complete_circularity_median']
    if circularity_median is not None:text += f"\n원형도 중앙값: {circularity_median:.3f} (1에 가까울수록 원형)"
    axes[0,0].text(.02, .98, text, va="top", fontsize=12, linespacing=1.8)
    if quality:
        axes[0,0].text(.02, .02, quality["chart_note"], color="#b02525", fontsize=11,
                       va="bottom", wrap=True)
    for ax, hist, xlabel, title in [(axes[0,1], d_hist, "등가원직경 (µm)", "직경 분포"),
                                   (axes[1,0], a_hist, "투영 면적 (µm²)", "면적 분포")]:
        if len(hist):
            ax.bar(hist.bin_left, hist["count"], width=hist.bin_right-hist.bin_left, align="edge", color="#27878e", edgecolor="white")
        ax.set(xlabel=xlabel, ylabel="후보 개수", title=title)
        ax.grid(axis="y", alpha=.2)
    ax = axes[1,1]
    ax.imshow(gray, cmap="gray")
    ax.imshow(np.ma.masked_where(~union, union), cmap="summer", alpha=.22, vmin=0,vmax=1)
    for y in yedges: ax.axhline(y, color="white", lw=.8)
    for x in xedges: ax.axvline(x, color="white", lw=.8)
    for r in range(counts.shape[0]):
        for c in range(counts.shape[1]):
            ax.text((xedges[c]+xedges[c+1])/2,(yedges[r]+yedges[r+1])/2,str(counts[r,c]),
                    ha="center",va="center",color="white",fontsize=13,bbox=dict(facecolor="black",alpha=.65,pad=3))
    ax.set(title="영역별 후보 개수 (중심점 기준, 이미지 경계 접촉 후보 포함)", xticks=[],yticks=[])
    fig.savefig(out / "dashboard.png", dpi=170)
    fig.savefig(out / "dashboard.pdf")
    plt.close(fig)
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k,v in stats.items() if not isinstance(v,dict))
    quality_html = f"<p><strong>{html.escape(quality['summary_ko'])}</strong></p>" if quality else ""
    overlay_url = "../" + report.get("overlay_relative_path", "entrance_candidates_overlay.png")
    shape_table=table[['candidate_id','length_um','width_um','aspect_ratio','equivalent_diameter_um','roundness','circularity']].rename(columns={'candidate_id':'ID','length_um':'Length (µm)','width_um':'Width (µm)','aspect_ratio':'Aspect ratio','equivalent_diameter_um':'Equivalent diameter (µm)','roundness':'Roundness','circularity':'Circularity'}).to_html(index=False,float_format=lambda v:f'{v:.3f}',na_rep='—',border=0)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>{html.escape(folder.name)} 분석</title>
<style>body{{font:16px 'Malgun Gothic',sans-serif;max-width:1150px;margin:35px auto;padding:0 20px;color:#25313b}}img{{max-width:100%}}aside{{background:#fff2d6;padding:18px;border-left:5px solid #bc8120}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:9px;overflow-wrap:anywhere}}a{{color:#006a80}}</style>
<h1>{html.escape(image_label)} 분석 결과</h1>
<aside><b>{html.escape(result_label)}</b>{quality_html}<br>마스크 오류가 포함될 수 있어 실제 pore 개수나 공극률로 확정할 수 없습니다.
면적률은 2D pore 후보 마스크의 합집합입니다. 개별 크기 분포는 이미지 경계에 닿는 후보를 제외하며,
큰 pore가 제외될 가능성에 대한 통계적 보정은 적용하지 않았습니다. 겹치는 후보는 개별 크기에 중복 기여할 수 있습니다.</aside>
<p><a href="candidates.csv">후보별 CSV</a> · <a href="summary.json">요약 JSON</a> · <a href="dashboard.pdf">PDF</a> · <a href="diameter_histogram.csv">직경 분포 CSV</a> · <a href="area_histogram.csv">면적 분포 CSV</a> · <a href="spatial_grid.csv">영역별 개수 CSV</a></p>
<img src="dashboard.png" alt="면적, 개수, 직경 및 공간 분포"><h2>분할 결과</h2><img src="{html.escape(overlay_url, quote=True)}" alt="후보 윤곽">
<h2>Pore details</h2><p>Length·Width는 동일한 2차 모멘트를 갖는 타원의 장축·단축 길이입니다. Aspect ratio = Length / Width. Roundness = 4 × 면적 / (π × Length²). Circularity = 4π × 면적 / 둘레². Roundness와 Circularity는 1에 가까울수록 2D에서 원형에 가깝습니다. 실제 3D 구형도를 측정한 값은 아닙니다. 이미지 경계에 닿은 pore는 보이는 부분만 계산하며, 요약 평균·중앙값에서는 제외합니다.</p>{shape_table}
<h2>계산 기준</h2><p>등가원직경 = 2√(투영 면적/π). 스케일바 문자는 사람이 확인한 값이며 OCR이 아닙니다.
자동 분할의 최소 후보 면적 {stats['minimum_filter_area_pixels']}px는 배율에 따라 다른 실제 면적에 해당합니다. 사용자 추가 마스크에는 이 자동 필터를 적용하지 않습니다. 서로 다른 이미지의 개수만 직접 비교하지 마세요.</p><table>{rows}</table></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({k:stats[k] for k in ["image","candidate_count","complete_candidate_count","analyzed_area_um2","union_candidate_area_um2","candidate_union_area_percent","complete_equivalent_diameter_um_median"]}), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.folders:
        export_folder(folder)


if __name__ == "__main__":
    main()
