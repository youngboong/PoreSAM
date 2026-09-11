# PoreSAM

SAM 2.1 기반 SEM 이미지 pore 분할·편집·측정 도구입니다. 현재 UI는 한국어이며, Windows에서 로컬 서버를 실행하고 브라우저로 사용합니다.

한국어판은 `korean` 브랜치에서 관리합니다. 영어판은 이 버전을 바탕으로 `english` 브랜치에서 별도로 개발할 예정입니다.

## 주요 기능

- 이미지 불러오기와 기존 분석·수정본 이어서 열기
- 스케일바 보정과 분석 영역 설정
- 밝기 정규화, background 제거, Gaussian·평균·Median·Bilateral·Kuwahara 필터 및 실시간 미리보기
- 자동 분할과 박스·타원형·포함/제외 점을 이용한 pore 보완
- 다각형으로 pore 추가, 드래그 절단, 대체·삭제·실행 취소
- 원본과 결과 비교, 색상·투명도 조절
- Pore details 창에서 개별 측정값, 요약 통계, Histogram, Scatter plot 확인
- 수정본별 이미지·측정 결과 저장 및 CSV·PNG·PDF 내보내기

## 실행

Python 3.11 `pore` 환경의 설치 방법은 [SETUP.md](SETUP.md)를 참고하세요. [공식 SAM 2 저장소](https://github.com/facebookresearch/sam2)의 SAM 2.1 Small 가중치를 별도로 받아 `checkpoints/sam2.1_hiera_small.pt`에 준비합니다. 모델 가중치는 이 저장소에 포함하지 않습니다.

```powershell
conda activate pore
python scripts/pore_editor.py
```

브라우저에서 http://127.0.0.1:8765 를 엽니다. 자세한 사용법은 [EDITOR.md](EDITOR.md)에 있습니다.

GPU가 없으면 CPU를 자동으로 사용합니다. CPU 실행을 강제하려면 `python scripts/pore_editor.py --device cpu`로 실행하세요. CPU 전용 PyTorch 설치 방법은 [SETUP.md](SETUP.md)에 있습니다.

## 저장소 구성

- `ui/`: 사용자 화면
- `scripts/`: 분할·전처리·편집 서버·측정·검증 코드
- `requirements*.txt`: 실행 및 개발 의존성
- `data/`: 로컬 원본 이미지 — Git 제외
- `checkpoints/`: 모델 가중치 — Git 제외
- `outputs/`: 분석 결과·수정본·실행 로그 — Git 제외

자동 분할 결과는 사용자가 검토·보완할 수 있습니다. 측정값은 2D 이미지의 pore 마스크를 기준으로 하며, Roundness는 3D 구형도를 뜻하지 않습니다.
