"""Build an index of PI measurements, excluding GF and crop experiments."""
import html
import json
from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[1]
out = root / "outputs/PI_analysis"
out.mkdir(exist_ok=True)
names = ["PI35_5kx-4_bse", "PI35_2kx-4_BSE8", "PI100_2kx_bse"]
rows = []
cards = []
for name in names:
    folder = root / "outputs" / f"{name}_first_pass" / "measurements"
    s = json.loads((folder / "summary.json").read_text())
    rows.append(dict(image=name, analyzed_area_um2=s["analyzed_area_um2"],
                     union_candidate_area_um2=s["union_candidate_area_um2"],
                     candidate_count=s["candidate_count"], complete_candidate_count=s["complete_candidate_count"],
                     edge_candidate_count=s["edge_candidate_count"],
                     candidate_area_percent=s["candidate_union_area_percent"],
                     median_equivalent_diameter_um=s["complete_equivalent_diameter_um_median"],
                     mean_equivalent_diameter_um=s["complete_equivalent_diameter_um_mean"],
                     candidates_per_10000_um2=s["observed_candidates_per_10000_um2"],
                     minimum_filter_area_um2=s["minimum_filter_area_um2"], status="unreviewed automatic candidates"))
    # Confirm that exported distributions and spatial grid account for the expected objects.
    for filename in ["diameter_histogram.csv", "area_histogram.csv"]:
        assert pd.read_csv(folder / filename)["count"].sum() == s["complete_candidate_count"]
    assert pd.read_csv(folder / "spatial_grid.csv")["candidate_count"].sum() == s["candidate_count"]
    assert len(pd.read_csv(folder / "candidates.csv")) == s["candidate_count"]
    url = f"../{name}_first_pass/measurements/"
    cards.append(f'<section><h2>{html.escape(name)}</h2><p><a href="{url}index.html">상세 보고서</a> · <a href="{url}candidates.csv">후보별 CSV</a> · <a href="{url}dashboard.pdf">PDF</a></p><a href="{url}dashboard.png"><img src="{url}dashboard.png" alt="{html.escape(name)} 통계"></a></section>')
frame = pd.DataFrame(rows)
frame.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
display = frame[["image","analyzed_area_um2","union_candidate_area_um2","candidate_count","complete_candidate_count","candidate_area_percent","median_equivalent_diameter_um"]].copy()
display.columns = ["이미지","분석 면적 (µm²)","후보 합집합 면적 (µm²)","관측 후보 수","크기 분포 대상 수","후보 면적률 (%)","등가원직경 중앙값 (µm)"]
table = display.to_html(index=False, float_format=lambda v:f"{v:,.2f}")
page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PI 입구 후보 분석</title>
<style>body{{font:16px 'Malgun Gothic',sans-serif;background:#f4f6f7;color:#23353c;max-width:1200px;margin:36px auto;padding:0 20px}}section{{background:white;padding:20px;margin:25px 0;border-radius:12px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:14px;width:100%;background:white}}th,td{{padding:12px;border:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}aside{{padding:18px;background:#fff0cf;border-left:5px solid #b27c1c;line-height:1.8}}a{{color:#076b79}}.scroll{{overflow-x:auto}}</style>
<h1>PI 입구 후보 분석</h1><p>기존 자동 분할 결과 3개 · GF 제외 · 확대 실험 결과 제외</p>
<aside><b>검증 전 자동 후보의 측정값입니다.</b> 실제 pore 개수·공극률로 확정할 수 없습니다.
면적률은 겹침을 한 번만 세는 마스크 합집합 기준입니다. 크기 분포는 이미지 경계에 닿는 후보를 제외합니다.
직경은 투영 면적과 같은 면적의 원 지름이며, 실제 3D pore 직경이 아닙니다.<br>
5kx와 2kx는 분석 범위·해상도와 최소 검출 면적이 다릅니다. PI35의 두 영상이 독립 시야인지도 확인되지 않아 합쳐 집계하지 않았습니다.</aside>
<p><a href="summary.csv">전체 요약 CSV</a> · <a href="http://127.0.0.1:8765">입구 보완 화면</a> · <a href="../../EDITOR.md">수정 화면 사용법</a></p>
<div class="scroll">{table}</div>{''.join(cards)}</html>'''
(out / "index.html").write_text(page, encoding="utf-8")
(out / "README.md").write_text("""# PI 분석 결과

`index.html`에서 PI 세 이미지의 보고서를 볼 수 있습니다. GF와 확대 실험은 제외했습니다.
`summary.csv`에는 이미지별 면적, 후보 수, 이미지 경계 접촉 후보 수, 등가원직경, 관측 개수 밀도를 기록했습니다.
각 원본 결과 폴더의 `measurements`에 그래프 PNG/PDF, 후보별 CSV, 면적/직경 분포 CSV,
3행×4열 공간별 개수 CSV, 요약 JSON을 저장했습니다.

자동 후보는 아직 정답 검증을 거치지 않았습니다. 투영 입구 면적률을 3D 공극률로 해석하지 않습니다.
이미지 경계 접촉 후보를 제외한 크기 분포에는 큰 객체가 빠질 가능성에 대한 보정이 없으며,
개별 후보가 겹치는 경우 그 후보들의 개별 크기는 중복될 수 있습니다.
PI35 5kx와 2kx는 독립 시야인지 확인되지 않았으므로 합산하지 않았습니다.

재실행:

```powershell
python scripts/analyze_candidates.py outputs/PI35_5kx-4_bse_first_pass outputs/PI35_2kx-4_BSE8_first_pass outputs/PI100_2kx_bse_first_pass
python scripts/summarize_pi.py
python scripts/test_measurements.py
```
""", encoding="utf-8")
print(frame.to_string(index=False))
print("Export consistency checks passed.")
