"""Recover exact instance labels from app revisions, never from colored PNGs."""
import argparse
import colorsys
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(original, masks, alpha, labels, colored):
    result = original.copy()
    for candidate_id, mask in sorted(masks.items()):
        tint = np.array(colorsys.hsv_to_rgb((candidate_id*.61803398875)%1,.8,1))*255 if colored else np.array([255,215,0])
        roi = result[:mask.shape[0]]
        roi[mask] = np.rint(roi[mask]*(1-alpha)+tint*alpha).astype(np.uint8)
        if labels:
            y,x = np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3).argmax(),mask.shape)
            cv2.putText(roi,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3)
            cv2.putText(roi,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'fine_tuning/data/260917')
    parser.add_argument('--out', type=Path, default=ROOT/'fine_tuning/datasets/260917_v1')
    parser.add_argument('--only', help='Exact export folder name to prepare separately.')
    parser.add_argument('--split', choices=['train','validation','test'], help='Override the pilot split for a new evaluation dataset.')
    parser.add_argument('--label-review', default='User confirmed manually reviewed final revisions.')
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output already exists; use a new dataset version.')
    sources = sorted(args.source.glob('*/comparison.png'))
    if args.only:
        sources=[p for p in sources if p.parent.name==args.only]
    assert sources, 'No image export folders found'
    projects = []
    app_root = Path(os.environ['LOCALAPPDATA'])/'PoreSAM/English'
    roots = [app_root/'outputs/projects', ROOT/'outputs/projects']
    roots += list((app_root/'outputs_workspaces').glob('*/outputs/projects'))
    for root in roots:
        for path in root.glob('*/project.json'):
            obj = json.loads(path.read_text(encoding='utf-8'))
            projects.append((path, obj))
    records = []
    for comparison in sources:
        name, rev = re.match(r'(.+)_images_revision_(\d+)_',comparison.parent.name).groups()
        compare = np.array(Image.open(comparison).convert('RGB'))
        assert compare.shape[1]%2 == 0
        original = compare[:,:compare.shape[1]//2]
        exported = np.array(Image.open(comparison.parent/'pores_colored.png').convert('RGB'))
        assert exported.shape == original.shape
        assert np.array_equal(compare[:,compare.shape[1]//2:],np.array(Image.open(comparison.parent/'segmentation.png').convert('RGB')))
        matches = []
        for project_path, project in projects:
            stem = re.sub(r'[^\w.-]+','_',Path(project['name']).stem).strip('._')[:60]
            if stem != name:
                continue
            normalized = project_path.parent/'input/normalized.png'
            if not normalized.exists() or not np.array_equal(np.array(Image.open(normalized).convert('RGB')),original):
                continue
            for run in project['runs']:
                revision = project_path.parent.parent.parent/'manual_edits'/run['dataset']/f'revision_{int(rev):04d}'
                mask_path = revision/'entrance_candidates.npz'
                if not mask_path.exists():
                    continue
                with np.load(mask_path,allow_pickle=False) as data:
                    masks = {int(k.rsplit('_',1)[1]):data[k].astype(bool) for k in data.files}
                # The colored export is regenerated pixel-for-pixel to identify the exact run.
                # Search alpha only when the normal UI default does not match.
                found = None
                alpha_estimates=[]
                for candidate_id,mask in sorted(masks.items()):
                    tint=np.array(colorsys.hsv_to_rgb((candidate_id*.61803398875)%1,.8,1))*255
                    a=original[:mask.shape[0]][mask].astype(float)[::20]
                    b=exported[:mask.shape[0]][mask].astype(float)[::20]
                    delta=tint-a
                    valid=abs(delta)>40
                    alpha_estimates.extend(np.rint(100*(b-a)[valid]/delta[valid]).astype(int).tolist())
                plausible=[i for i in alpha_estimates if 1<=i<=100]
                estimated=int(np.bincount(plausible,minlength=101).argmax()) if plausible else 40
                for alpha in sorted({.4,estimated/100}):
                    for labels in [True,False]:
                        if np.array_equal(render(original,masks,alpha,labels,True),exported):
                            found = (alpha,labels)
                            break
                    if found:
                        break
                if found:
                    matches.append((project_path,project,run,mask_path,masks,found))
        assert matches, f'Cannot locate exact saved masks for {name}; no pseudo-label fallback.'
        # Duplicated workspaces may contain identical saved revisions.
        hashes = {digest(m[3]) for m in matches}
        if len(hashes)>1:
            first = matches[0][4]
            assert all(set(first)==set(m[4]) and all(np.array_equal(first[k],m[4][k]) for k in first) for m in matches)
        project_path,project,run,mask_path,masks,rendering = matches[0]
        shapes = {m.shape for m in masks.values()}
        assert len(shapes)==1 and masks
        height,width = next(iter(shapes))
        assert width == original.shape[1] and height <= original.shape[0]
        instance = np.zeros((height,width),np.uint16)
        ids = {}
        for label,(candidate_id,mask) in enumerate(sorted(masks.items()),1):
            assert mask.any() and not np.any(instance[mask]), f'Empty or overlapping pore in {name}'
            instance[mask]=label
            ids[str(label)]=candidate_id
        folder=args.out/name
        folder.mkdir(parents=True)
        Image.fromarray(original[:height]).save(folder/'image.png')
        Image.fromarray(instance).save(folder/'instances.tif')
        shutil.copy2(mask_path,folder/'source_masks.npz')
        # Group neighboring captures of the same named sample/magnification together.
        group = re.sub(r'-\d+.*$','',name)
        split = args.split or ('test' if group.startswith('PI300_10kx') else ('validation' if group=='PI100_5kx' else 'train'))
        record = dict(name=name,group=group,split=split,image=f'{name}/image.png',labels=f'{name}/instances.tif',
                      instances=len(masks),shape=[height,width],label_to_original_id=ids,
                      exported_folder=str(comparison.parent.resolve()),project=str(project_path),dataset=run['dataset'],revision=int(rev),
                      source_masks=str(mask_path),source_masks_sha256=digest(mask_path),original_file_sha256=digest(project_path.parent/project['original_file']),
                      image_sha256=digest(folder/'image.png'),labels_sha256=digest(folder/'instances.tif'),
                      source_preprocessing=run['config'],export_render=dict(opacity=rendering[0],labels=rendering[1],exact_match=True))
        records.append(record)
        print(f"{name}: {len(masks)} pores, {split}, exact export match",flush=True)
    assert len({r['image_sha256'] for r in records}) == len(records), 'Duplicate input images'
    for field in ['group','original_file_sha256']:
        for key in {r[field] for r in records}:
            assert len({r['split'] for r in records if r[field]==key}) == 1, 'Data leakage'
    manifest=dict(version=1,source=str(args.source.resolve()),label_review=args.label_review,
                  input_processing='App imported grayscale/RGB image cropped to analysis ROI; no background removal or extra blur.',
                  split_notes='Grouped by filename sample/magnification, not guaranteed independent physical specimens. Test held out from optimization and checkpoint selection.',records=records)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({s:dict(images=sum(r['split']==s for r in records),pores=sum(r['instances'] for r in records if r['split']==s)) for s in ['train','validation','test']},indent=2))


if __name__=='__main__':
    main()
