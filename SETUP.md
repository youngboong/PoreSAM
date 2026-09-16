# pore 환경

이 문서는 `korean` 브랜치의 한국어 브라우저판 환경입니다. 최신 영어 Windows 앱은 기본 [app 브랜치의 DESKTOP.md](https://github.com/youngboong/PoreSAM/blob/app/DESKTOP.md)에 있는 `pore-app-cpu` 환경과 `scripts/pore_app.py`를 사용합니다.

Windows PowerShell에서 실행합니다. GPU가 없어도 CPU로 실행할 수 있습니다.

## CPU 환경

```powershell
conda create -n pore python=3.11 -y
conda activate pore
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python scripts/check_environment.py --device cpu
python scripts/pore_editor.py --device cpu
```

SAM 2.1 Small 가중치는 별도로 `checkpoints/sam2.1_hiera_small.pt`에 준비합니다. CPU 환경에서는 CUDA 패키지 버전이 기록된 `requirements-lock.txt` 대신 `requirements.txt`를 사용하세요. GPU용 PyTorch가 이미 설치된 환경에서도 `--device cpu`로 실행할 수 있습니다.

## NVIDIA GPU 환경

```powershell
conda create -n pore python=3.11 -y
conda activate pore
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python scripts/check_environment.py
```

실제 설치된 전체 패키지 버전은 `requirements-lock.txt`에 기록했습니다.
동일한 환경을 재현하려면 위의 PyTorch 설치와 `SAM2_BUILD_CUDA` 설정 후
`python -m pip install --no-build-isolation -r requirements-lock.txt`를 사용합니다.

SAM 2.1을 지원하는 공식 SAM 2 소스의 버전을 고정했습니다.
Windows에서 별도 CUDA 컴파일러 없이 설치하도록 선택적 SAM CUDA 확장을 생략합니다.
GPU 추론은 PyTorch의 CUDA를 사용합니다. SAM 확장에 의존하는 작은 구멍/점 제거 후처리는 사용할 수 없습니다.

VS Code에서 `Python: Select Interpreter`로 `pore`를 선택합니다.
노트북에서도 커널로 같은 환경을 선택하면 됩니다.

`scripts/check_environment.py`는 패키지 로딩, 선택된 장치의 연산, TorchVision NMS,
SAM 2.1 설정 로딩 및 TIFF 읽기를 확인합니다. 학습된 가중치 다운로드나 분할 정확도 검증은 별도입니다.

사용자가 박스·포함/제외 점으로 누락을 추가하는 화면은 `python scripts/pore_editor.py`로 실행합니다.
브라우저에서 http://127.0.0.1:8765 를 열며, 자세한 사용법은 [EDITOR.md](EDITOR.md)에 있습니다.
