# 주요 Python 기능 실행 및 출력 경로

명령은 **프로젝트 루트**에서 PowerShell로 실행한다. 아래 예시는 이 컴퓨터의 경로를 사용한다. `scripts` 안의 모든 `.py` 파일이 독립 실행 프로그램인 것은 아니다. `pore_preprocessing.py`, `automate_pores.py` 등은 다른 코드에서 함수를 불러 사용하는 모듈이다.

## 1. 실행 준비

```powershell
Set-Location 'C:\Users\YoungJin\Documents\pore_seg'
conda activate pore-app-cpu
```

환경을 활성화하지 않을 때는 아래처럼 Python 경로를 직접 지정할 수 있다.

```powershell
& 'C:\Users\YoungJin\anaconda3\envs\pore-app-cpu\python.exe' scripts/pore_app.py
```

SAM 실행에는 `checkpoints/sam2.1_hiera_small.pt`가 필요하다. CPU 환경 설치와 빌드 방법은 [DESKTOP.md](DESKTOP.md)를 참고한다.

## 2. 자주 쓰는 실행 명령

### Windows 앱

```powershell
python scripts/pore_app.py
```

- 기본 장치: CPU.
- 기본 데이터 폴더는 아래 명령으로 확인한다. 이후 이 문서에서는 그 위치를 `<데이터 루트>`라고 쓴다.
- 설치된 `C:\Users\YoungJin\Documents\PoreSAM\PoreSAM.exe`도 같은 기본 데이터 위치를 사용한다.

```powershell
python -c "import sys; sys.path.insert(0, 'scripts'); from pore_app import default_data_dir; print(default_data_dir())"
```

프로젝트의 `outputs`를 사용하려면 명시한다.

```powershell
python scripts/pore_app.py --device cpu --data-dir .\outputs
```

이미 같은 데이터 위치를 사용하는 앱이 열려 있으면 새 창에는 별도 workspace가 배정된다. 두 번째 창은 `<데이터 루트의 상위 폴더>/outputs_workspaces/workspace_0002/outputs`를 사용한다.

### 브라우저 UI

```powershell
python scripts/pore_editor.py --device cpu --port 8765
```

브라우저에서 `http://127.0.0.1:8765`를 연다. 기본 저장 위치는 **프로젝트의 `outputs`**다. 포트를 이미 사용 중이면 `--port 8766`처럼 바꾼다. 같은 데이터 폴더에 데스크톱 앱과 브라우저 서버를 동시에 쓰지 않는다.

### 터미널에서 처음 SAM 분석

```powershell
python scripts/segment_first_pass.py --image 'data/GF_1kx_1_BSE.tif' --scale-um 50 --device cpu --points-per-side 48 --min-area 100 --min-contrast 8 --output 'outputs/cli/GF_BSE_first_pass'
```

`--scale-um 50`은 예시 이미지의 스케일 표시값이다. 다른 이미지에서는 이미지 하단의 실제 숫자로 바꾼다.

| 옵션 | 의미 |
|---|---|
| `--image` | 입력 TIF 경로 |
| `--scale-um` | 스케일 바가 나타내는 길이(μm), 필수 |
| `--device cpu` | CPU 사용 |
| `--points-per-side 48` | 48×48개 점으로 탐색 |
| `--min-area 100` | 최소 후보 면적 100 px² |
| `--min-contrast 8` | 주변 평균 밝기 − 내부 평균 밝기 최소 8 |
| `--output` | 결과 폴더. 생략하면 `outputs/<이미지 이름>_first_pass` |
| `--reuse` | 같은 출력 폴더의 저장된 SAM 후보를 재사용해 선별만 다시 수행 |

이 CLI는 기존의 **흑백 SEM TIF용 첫 분석 도구**다. 하단 영역과 스케일 바가 자동 검출되어야 하며, UI의 밝기 정규화·background removal·smoothing 설정을 자동으로 읽거나 적용하지 않는다. UI와 같은 전처리·영역 설정을 사용하려면 앱의 `Run Analysis`를 이용한다.

동일한 `--output`으로 실행하면 해당 폴더의 결과를 갱신한다. 비교 실험은 출력 폴더를 다르게 지정한다. `--reuse`는 저장된 SAM 설정을 그대로 사용하므로 점 밀도를 바꾸는 재분석 용도가 아니다.

출력 예:

```text
outputs/cli/GF_BSE_first_pass/
  report.json                         분석 설정, 스케일, 개수
  sam_raw/
    raw_masks.npz                     SAM 품질·안정성·중복 검사 이후 후보
    raw_metadata.json                 SAM 후보별 점수와 메타데이터
  masks/
    entrance_candidates.npz           pore 선별 이후 마스크
  images/
    analysis_region_and_scale.png
    raw_sam_overlay.png
    entrance_candidates_overlay.png
    comparison.png
```

파일명의 `entrance_candidates`는 기존 저장 형식의 이름이고, 현재 UI의 pore 마스크에 해당한다.

### 저장된 CLI 분석에서 측정·보고서 생성

위 명령으로 첫 분석을 만든 뒤 실행한다. SAM을 다시 돌리지 않는다.

```powershell
python scripts/analyze_candidates.py 'outputs/cli/GF_BSE_first_pass'
```

출력은 입력 결과 폴더 안의 `measurements/`다.

```text
measurements/
  candidates.csv              pore별 면적·등가직경·형상
  summary.json                전체 통계와 측정 정의
  diameter_histogram.csv
  area_histogram.csv
  spatial_grid.csv
  union_mask.png
  dashboard.png
  dashboard.pdf
  index.html
  report_format.json
```

데스크톱의 수정본은 보고서 이미지가 아직 생성되지 않았을 수 있다. 그 경우 앱의 `Generate Report` 또는 아래 표의 `Editor.generate_report()`를 사용한다. 이 함수는 필요한 이미지도 준비한 뒤 측정 보고서를 만든다.

### 환경 점검

```powershell
python scripts/check_environment.py --device cpu
```

패키지·CPU 연산·SAM 구조 생성·TIF 읽기를 확인하고 **터미널에 JSON을 출력**한다. 학습 가중치를 읽어 실제 segmentation을 검증하는 명령은 아니다. 파일이 필요하면 직접 저장한다.

```powershell
New-Item -ItemType Directory -Force outputs/environment_setup | Out-Null
python scripts/check_environment.py --device cpu | Out-File -Encoding utf8 outputs/environment_setup/cpu_check.json
```

## 3. 주요 함수와 저장 여부

| 파일 / 함수 | 역할 | 반환값 또는 저장 위치 |
|---|---|---|
| `pore_preprocessing.prepare_image(gray, config)` | 정규화 → 밝은 가는 구조 제거 → smoothing | 처리 이미지 배열, 처리 메타데이터. 자동 저장 없음 |
| `segment_first_pass.detect_footer(gray)` | SEM 하단 정보 영역 경계 추정 | 경계 y 좌표. 자동 저장 없음 |
| `segment_first_pass.detect_scale_bar(gray, footer_y, label_um)` | 스케일 바 픽셀 길이와 μm/px 계산 | 스케일 딕셔너리. 자동 저장 없음 |
| `segment_first_pass.candidates_from_masks(raw, gray, min_contrast=8, min_area=100)` | SAM 후보에 연결성·면적·대비·포함 관계 검사 | 선별된 후보 목록. 자동 저장 없음 |
| `segment_first_pass.overlay(gray, masks, numbered=False)` | 이미지에 마스크 표시 | PIL 이미지. `.save(path)`로 저장 |
| `project_workflow.ProjectWorkflow.upload(name, content)` | 입력 파일 등록, 분석용 흑백 이미지 생성 | 데이터 루트의 `projects/<image_id>/input/`, `project.json` |
| `ProjectWorkflow.start(project_id, payload)` | UI와 같은 분석을 비동기로 시작 | job ID. 완료 시 `projects/<image_id>/runs/run_NNNN/` |
| `ProjectWorkflow.status(job_id)` | 분석 진행률·완료 상태 확인 | 상태 딕셔너리 |
| `pore_editor.Editor.state(dataset)` | 분석과 최신 수정본 불러오기 | 이미지·마스크·설정이 담긴 state. SAM 재실행 없음 |
| `Editor.preview(state, payload)` | 박스·타원·점으로 SAM 미리보기 | 후보는 `state['preview']`, 응답에는 점수·미리보기. 수정본 저장 없음 |
| `automate_pores.propose_boxes(state, limit=32)` | 미검출 어두운 영역에서 탐색 박스 제안 | `boxes`, `threshold`. SAM 실행·마스크 저장 없음 |
| `automate_pores.add_candidate(editor, state, payload, staged=True)` | 한 박스의 SAM 후보를 추가·병합·trim | `staged=True`이면 메모리의 state만 변경. 기본값 `False`는 수정본 저장 |
| `automate_pores.consolidation_boxes(state, limit=16)` | Automate 추가 pore와 접한 주변 pore의 병합 검토 박스 | 박스 목록. 자동 저장 없음 |
| `pore_overlap.exclusive_existing(masks, annotations=None)` | 중복 픽셀을 하나의 pore에만 배정 | 정리된 마스크와 변경 정보. 자동 저장 없음 |
| `analyze_candidates.measure_masks(masks, shape, um_per_pixel)` | 면적·등가직경·형상·면적 비율 계산 | 표, 요약, 합집합 마스크, 공간별 개수 등. 자동 저장 없음 |
| `analyze_candidates.export_folder(folder)` | 저장된 마스크에서 측정 파일 생성 | `<folder>/measurements/`. 보고서용 overlay는 별도 준비되어 있어야 함 |
| `Editor.save(...)` | 마스크와 편집 이력을 새 revision으로 저장 | `manual_edits/<dataset>/revision_NNNN/`, `latest.json` |
| `Editor.generate_report(state, payload)` | 최신 마스크의 보고서를 필요할 때 생성 | 해당 분석/revision의 `images/`, `measurements/`; 지정 시 외부 보고서 폴더 |
| `report_bundle.export_images(editor, state, payload)` | 원본 비교·단색·개별 색상 이미지 내보내기 | 사용자가 지정한 폴더 아래 새 하위 폴더 |
| `report_bundle.export_bundle(editor, state, directory)` | 이미 생성된 보고서를 외부 폴더로 내보내기 | 새 하위 폴더에 `report.pdf`, `report.html`만 저장 |
| `pore_details_export.export_plot(state, payload)` | 선택 pore의 histogram/scatter 그림 생성 | PNG data URL 반환. 실제 파일 저장은 호출자/UI에서 수행 |

`gray`는 보통 2차원 `uint8` 배열, `mask`는 같은 높이·너비의 `bool` 배열이다. `state['masks']`는 `{pore_id: mask}` 형태다. 수동 수정 및 전체 Automate는 UI에서 실행하면 revision, Stop, Undo 처리를 함께 이용할 수 있다.

## 4. 전처리 함수만 실행하기

아래 블록을 PowerShell에 붙여 넣으면 SAM 실행 없이 전처리 결과를 저장한다. 예시는 8비트 GF 이미지와 분석 높이 768을 사용한다. 다른 이미지에서는 입력 경로·영역을 바꾼다. 16비트 원본 대신 앱에서 만든 `input/normalized.png`를 사용하면 입력 형식 변환도 맞출 수 있다.

```powershell
@'
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "scripts")
from pore_preprocessing import prepare_image

source = Path("data/GF_1kx_1_BSE.tif")
gray = np.asarray(Image.open(source).convert("L"))[:768]
config = dict(mode="adjustable", normalize_enabled=True,
              background_strength=3, blur_method="gaussian", blur_strength=2)
processed, metadata = prepare_image(gray, config)
out = Path("outputs/python_examples/preprocess")
out.mkdir(parents=True, exist_ok=True)
Image.fromarray(processed).save(out / "processed.png")
(out / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(out.resolve())
'@ | python -
```

출력: `outputs/python_examples/preprocess/processed.png`, `settings.json`.

## 5. 저장된 마스크에서 측정값만 CSV로 얻기

2번의 CLI 분석을 만든 뒤 사용할 수 있다. `folder`를 원하는 분석 또는 revision 폴더로 바꾼다. `read_artifact()`는 새 `masks/` 구조와 수정본의 평면 구조를 모두 지원한다.

```powershell
@'
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
from result_paths import read_artifact
from analyze_candidates import measure_masks

folder = Path("outputs/cli/GF_BSE_first_pass")
report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
with np.load(read_artifact(folder, "entrance_candidates.npz"), allow_pickle=False) as data:
    masks = sorted((int(k.rsplit("_", 1)[1]), data[k].copy()) for k in data.files)
if not masks:
    raise SystemExit("No pore masks in this result.")
table, stats, union, counts, complete_counts = measure_masks(
    masks, masks[0][1].shape, report["scale"]["um_per_pixel"])
table["image_area_percent"] = table["area_pixels"] / union.size * 100
out = Path("outputs/python_examples/measurements")
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "pores.csv", index=False, encoding="utf-8-sig")
(out / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(out.resolve())
'@ | python -
```

출력: `outputs/python_examples/measurements/pores.csv`, `summary.json`.

면적 비율은 분석 영역의 2D 마스크 면적 비율이다. `measure_masks()`의 요약 중 `complete_*` 통계와 기존 보고서의 크기 분포는 이미지 경계에 닿는 pore를 제외하며, 전체 pore 수와 전체 면적 비율에는 포함한다.

## 6. Automate 박스만 확인하기

SAM을 실행하지 않고 **어디에 박스를 제안하는지** 확인하는 예시다. 먼저 2번의 CLI 분석 결과가 있어야 한다. 원래 마스크는 수정하지 않는다.

```powershell
@'
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "scripts")
from result_paths import read_artifact
from automate_pores import propose_boxes

folder = Path("outputs/cli/GF_BSE_first_pass")
report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
gray = np.asarray(Image.open(report["image"]).convert("L"))[:report["analysis_bottom_exclusive"]]
sam_gray = np.asarray(Image.open(report["sam_input_image"]).convert("L")) if report.get("sam_input_image") else gray
with np.load(read_artifact(folder, "entrance_candidates.npz"), allow_pickle=False) as data:
    masks = {int(k.rsplit("_", 1)[1]): data[k].copy() for k in data.files}
state = dict(gray=gray, sam_gray=sam_gray, masks=masks, report=report)
result = propose_boxes(state)
out = Path("outputs/python_examples/automate_boxes")
out.mkdir(parents=True, exist_ok=True)
canvas = Image.fromarray(gray).convert("RGB")
draw = ImageDraw.Draw(canvas)
for number, item in enumerate(result["boxes"], 1):
    x0, y0, x1, y1 = item["box"]
    draw.rectangle((x0, y0, x1, y1), outline="cyan", width=2)
    draw.text((x0, y0), str(number), fill="yellow")
canvas.save(out / "boxes.png")
(out / "boxes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(len(result["boxes"]), out.resolve())
'@ | python -
```

출력: `outputs/python_examples/automate_boxes/boxes.png`, `boxes.json`.

현재 누락 탐색 상한은 **32개**, 후속 접촉 그룹 병합 검토 상한은 **16개**다. 새 추가·병합의 SAM 점수 기준은 **0.7**이고, 첫 `Run Analysis`의 품질 기준 **0.8 초과**와는 별개다.

## 7. 앱 결과가 저장되는 구조

아래의 `<데이터 루트>`는 실행 방법에 따라 다르다.

| 실행 방법 | 데이터 루트 |
|---|---|
| 설치 EXE / 기본 `pore_app.py` | 위 `default_data_dir()` 명령으로 확인한 폴더 |
| `pore_app.py --data-dir .\outputs` | 프로젝트의 `outputs` (첫 창) |
| `pore_editor.py` | 프로젝트의 `outputs` (환경변수 `PORESAM_DATA_DIR` 미설정 시) |
| `segment_first_pass.py --output ...` | 직접 지정한 결과 폴더 |

```text
<데이터 루트>/
  projects/<image_id>/
    project.json                         이미지 이름·설정·실행 이력
    input/
      original.tif                       원본; 확장자는 업로드 형식에 따라 다름
      normalized.png                     분석용 8비트 흑백 이미지, 하단 포함
    runs/run_NNNN/
      report.json
      images/analysis_input.png          선택한 전처리 적용 후 실제 SAM 입력
      images/raw_sam_overlay.png
      sam_raw/raw_masks.npz
      sam_raw/raw_metadata.json
      masks/entrance_candidates.npz
      images/...                         보고서 요청 시 추가 생성되는 이미지
      measurements/...                   보고서 요청 시 생성
  manual_edits/<image_id>__run_NNNN/
    latest.json                          현재 선택된 revision 번호
    revision_NNNN/
      entrance_candidates.npz            이 수정본의 실제 마스크
      report.json                        스케일·설정·개수
      edit.json                          편집 내역·ID·출처·Undo 이력
      images/...                         보고서 요청 시 생성
      measurements/...                   보고서 요청 시 생성
```

`project.json`만으로 마스크를 복구할 수는 없다. `Open Analysis`는 해당 run과 `latest.json`이 가리키는 수정본을 읽으며, 이미지를 자동 재분석하지 않는다. 수정본이 없으면 최초 run 마스크를 읽는다.

### 사용자가 선택한 외부 폴더

- **Save Images:** `<선택 폴더>/<이미지명>_images_revision_NNNN_<시각>_<식별자>/`에 `comparison.png`, `segmentation.png`, `pores_colored.png`. 기존 하단 정보 영역을 포함하며, 비교 이미지는 원본 표시와 segmentation을 나란히 붙인다. 원본 파일 자체는 프로젝트의 `input/original.*`에 있다.
- **Generate Report:** `<선택 폴더>/<이미지명>_report_revision_NNNN_<시각>_<식별자>/`에 `report.pdf`, `report.html`만 내보낸다. 내부 `measurements`의 CSV 등은 데이터 루트에 남는다.
- **Pore Details의 CSV·그림 저장:** Windows 저장 대화상자에서 선택한 파일 경로.
- 데스크톱 로그: `<데이터 루트의 상위 폴더>/logs/desktop.log`, `console.log`.

## 8. 핵심 검증

`scripts/`에는 앱 실행·분석·환경 확인에 필요한 파일만 있다. 핵심 회귀 검증은 `tests/`에 있으며, 일반 분석을 위해 실행할 필요는 없다. 사용하지 않는 실험 및 진단 코드는 제거했다.

Automate 회귀 검증 예시(Playwright가 설치된 개발 환경 필요):

```powershell
python -m pip install playwright==1.62.0
python tests/check_automate_merge.py
```

출력: `outputs/ui_checks/automate_merge_<시각>/verification.json`. 누락 탐색 32개 상한, 점수 기준, 병합·trim·수동 Cut 보호·Undo·Stop 동작을 검사한다.

데스크톱 검증은 `outputs/desktop_checks/`, UI 회귀 검증은 `outputs/ui_checks/`에 저장된다. [tests/README.md](tests/README.md)에 실행 명령을 정리했다.

## 9. 별도 파인튜닝

학습 코드는 앱 실행 코드와 분리해 `fine_tuning/`에서 관리한다. GPU용 `pore` 환경에서 프로젝트 루트를 기준으로 실행한다.

```powershell
conda activate pore
python fine_tuning/cross_validate.py
```

`fine_tuning/datasets/260917_v1/`의 검토된 마스크를 사용해 5-fold 교차검증을 실행하고, 마지막에 전체 데이터 재학습본을 만든다. 모델은 `fine_tuning/models/<실험명>/`, 로그·지표·비교 그림·HTML은 `fine_tuning/runs/<실험명>/`에 저장한다. 기존 앱의 체크포인트는 교체하지 않는다.

데이터 준비, 검증 기준과 저장 파일은 [fine_tuning/README.md](fine_tuning/README.md)를 참고한다.

16장을 한 장씩 제외하는 **15장 학습 / 1장 평가**와 면적 중심 비교는 다음으로 실행한다.

```powershell
python fine_tuning/leave_one_out.py
```

출력: `fine_tuning/runs/loo16_area_<시각>/index.html`과 `area_metrics.csv`. 각 `fold_N/area_evaluation/`의 `comparison.png`는 전체 비교, `area_errors_comparison.png`는 두 색의 오차 비교다. **하늘색은 놓친 reference 면적, 분홍색은 reference 밖으로 잘못 채운 면적**이다. 모델은 `fine_tuning/models/loo16_area_<시각>/fold_N.pt`에 각각 저장된다.

학습·평가 모두 밝기 정규화를 적용한다. 주 지표는 전체 pore 영역 IoU이며, 개수 기준 F1이 아닌 픽셀 기준 면적 지표를 사용한다. 서로 비슷한 촬영 이미지가 학습·평가에 나뉠 수 있으므로 새 시편에 대한 독립 검증으로 해석하지 않는다.
