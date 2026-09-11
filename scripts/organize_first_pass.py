"""Move existing first-pass artifacts without changing masks or measurements."""
import hashlib
import json
from pathlib import Path
import re
import time

from result_paths import ARTIFACT_PATHS

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (ROOT / 'outputs').resolve()
LAYOUT_NOTE = '''## 결과 파일 위치

- [분석 보고서](measurements/index.html): 면적·개수·직경·공간 분포
- [원본과 분할 결과 비교](images/comparison.png)
- [후보 번호와 윤곽](images/entrance_candidates_overlay.png)
- `images/`: 비교·윤곽·분석 영역과 스케일 확인 이미지
- `masks/`: 최종 입구 후보 마스크 (`entrance_candidates.npz`)
- `sam_raw/`: SAM 원시 마스크와 메타데이터
- `measurements/`: CSV, 요약 JSON, 그래프, HTML/PDF 보고서
- `report.json`: 입력 이미지, 스케일, 분할 설정과 후보 정보

'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folders = sorted(OUTPUTS.glob('*_first_pass'))
    moves = []
    protected = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        if not folder.resolve().is_relative_to(OUTPUTS):
            raise ValueError(f'Folder outside outputs: {folder}')
        for name, relative in ARTIFACT_PATHS.items():
            source, target = folder / name, folder / relative
            if not source.exists():
                continue
            # Validate resolved file paths before any move; never overwrite a target.
            if not source.resolve().is_relative_to(folder.resolve()) or not target.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f'Path outside intended folder: {source}, {target}')
            if target.exists():
                raise FileExistsError(target)
            moves.append(dict(source=str(source), target=str(target), sha256=digest(source)))
        for path in (folder / 'measurements').glob('*'):
            if path.is_file() and path.name != 'index.html':
                protected[str(path)] = digest(path)
    log_folder = OUTPUTS / 'layout_migrations'
    log_folder.mkdir(exist_ok=True)
    journal = log_folder / f'first_pass_{time.time_ns()}.json'
    texts = {}
    for folder in folders:
        for relative in ['README.md', 'report.json', 'measurements/index.html']:
            path = folder / relative
            texts[str(path)] = path.read_text(encoding='utf-8') if path.exists() else None
    record = dict(status='planned', moves=moves, original_texts=texts, protected_sha256=protected)
    journal.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    for item in moves:
        source, target = Path(item['source']), Path(item['target'])
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        assert digest(target) == item['sha256'], target
    for folder in folders:
        report_path = folder / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['overlay_relative_path'] = 'images/entrance_candidates_overlay.png'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        page = folder / 'measurements/index.html'
        if page.exists():
            text = page.read_text(encoding='utf-8').replace('../entrance_candidates_overlay.png', '../images/entrance_candidates_overlay.png')
            page.write_text(text, encoding='utf-8')
        readme = folder / 'README.md'
        text = readme.read_text(encoding='utf-8') if readme.exists() else f'# {folder.name}\n\n'
        if '## 결과 파일 위치' not in text:
            for name, relative in ARTIFACT_PATHS.items():
                text = re.sub(r'(?<![/\w])' + re.escape(name), relative, text)
            title, separator, body = text.partition('\n')
            text = title + '\n\n' + LAYOUT_NOTE + body.lstrip('\n')
            if (folder / 'quality_review.json').exists():
                quality = json.loads((folder / 'quality_review.json').read_text(encoding='utf-8'))
                text += '\n## 검토 메모\n\n' + quality['summary_ko'] + '\n'
            readme.write_text(text, encoding='utf-8')
    for path, expected in protected.items():
        assert digest(Path(path)) == expected, path
    record['status'] = 'verified'
    journal.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(status='verified', folders=len(folders), moved_files=len(moves),
                          unchanged_measurement_files=len(protected), journal=str(journal))))


if __name__ == '__main__':
    main()
