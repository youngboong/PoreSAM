"""Persistent uploads and immutable analysis runs for the local Pore Editor."""
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from app_paths import checkpoint_path
import secrets
import threading
import traceback

import cv2
import numpy as np
from PIL import Image
import tifffile
import torch

from analyze_candidates import export_folder
from result_paths import read_artifact, write_artifact
from pore_preprocessing import preprocessing_config, prepare_image, adjustable_config
from segment_first_pass import detect_footer, detect_scale_bar, overlay
from sam2.build_sam import build_sam2
from sam_runtime import inference_context, release_device_cache
from operation_cancel import OperationCancelled
from pore_selection import TraceGenerator, select
from analysis_timing import AnalysisTiming


class ProjectWorkflow:
    def __init__(self, editor, root, image_exporter):
        self.editor, self.root, self.image_exporter = editor, Path(root), image_exporter
        self.lock = threading.RLock()
        self.projects, self.jobs = {}, {}
        self.job_timings = {}
        self.running = False
        self.digest_cache = {}
        for path in self.root.glob('*/project.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            # Deleted analyses must not reappear as broken library entries.
            data['runs'] = [r for r in data.get('runs', []) if (path.parent/'runs'/r['id']/'report.json').is_file()]
            self.projects[data['id']] = data

    def persist(self, project):
        folder = self.root / project['id']
        pending = folder / 'project.pending.json'
        pending.write_text(json.dumps(project, indent=2, ensure_ascii=False), encoding='utf-8')
        pending.replace(folder / 'project.json')

    def datasets(self):
        with self.lock:
            return {run['dataset']: (self.root/p['id']/'runs'/run['id'], f"{p['name']} · Analysis {run['number']}")
                    for p in self.projects.values() for run in p.get('runs', [])}

    def list_projects(self):
        with self.lock:
            return [dict(id=p['id'], name=p['name'], run_count=len(p['runs'])) for p in self.projects.values()]

    def file_digest(self, path):
        path=Path(path).resolve()
        stat=path.stat()
        key=(str(path),stat.st_mtime_ns,stat.st_size)
        if key not in self.digest_cache:
            self.digest_cache[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        return self.digest_cache[key]

    def upload(self, name, content, seed=None):
        if not content or len(content) > 32*1024*1024:
            raise ValueError('Choose an image up to 32 MB.')
        suffix = Path(name).suffix.lower()
        if suffix not in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            raise ValueError('Choose a TIF, PNG or JPG image.')
        try:
            if suffix in ['.tif', '.tiff']:
                with tifffile.TiffFile(io.BytesIO(content)) as tif:
                    if len(tif.pages) != 1:
                        raise ValueError('Choose a single-page TIF image.')
                    page = tif.pages[0]
                    if math.prod(page.shape) > 32_000_000:
                        raise ValueError('Image is too large. Crop it and try again.')
                    pixels = page.asarray()
            else:
                with Image.open(io.BytesIO(content)) as image:
                    if image.width*image.height > 16_000_000:
                        raise ValueError('Image is too large. Crop it and try again.')
                    pixels = np.asarray(image.convert('RGB'))
        except (OSError, tifffile.TiffFileError) as exc:
            raise ValueError('Cannot read image.') from exc
        if pixels.ndim not in [2, 3] or (pixels.ndim == 3 and pixels.shape[2] not in [3, 4]):
            raise ValueError('Only single grayscale or RGB images are supported.')
        if min(pixels.shape[:2]) < 32 or np.prod(pixels.shape[:2]) > 16_000_000:
            raise ValueError('Image dimensions must be at least 32 px, with at most 16 million pixels.')
        if pixels.dtype.kind not in 'uif' or not np.isfinite(pixels).all():
            raise ValueError('Unsupported pixel format.')
        original_dtype = str(pixels.dtype)
        normalization = None
        if pixels.dtype != np.uint8:
            low, high = float(pixels.min()), float(pixels.max())
            if high <= low: raise ValueError('Image has no intensity variation.')
            pixels = np.clip((pixels.astype(np.float64)-low)*255/(high-low), 0, 255).astype(np.uint8)
            normalization = dict(method='linear min-max to uint8', low=low, high=high)
        gray = cv2.cvtColor(pixels[:,:,:3], cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
        h, w = gray.shape
        try: bottom = detect_footer(gray)
        except ValueError: bottom = h
        try: scale = detect_scale_bar(gray, bottom, 1) if bottom < h else None
        except ValueError: scale = None
        project_id = 'image_' + secrets.token_hex(8)
        folder = self.root / project_id
        (folder/'input').mkdir(parents=True)
        (folder/'input'/('original'+suffix)).write_bytes(content)
        Image.fromarray(gray).save(folder/'input/normalized.png')
        config = dict(analysis_bottom=bottom, scale_um=None,
                      scale_pixels=scale['length_pixels'] if scale else None,
                      min_contrast=8, min_area_pixels=100, points_per_side=48,
                      preprocessing_mode='adjustable', normalize_enabled=True,
                      background_strength=0, blur_method='none', blur_strength=2)
        if seed:
            report = json.loads((seed/'report.json').read_text(encoding='utf-8'))
            config.update(analysis_bottom=report['analysis_bottom_exclusive'],
                          scale_um=report['scale']['label_um'], scale_pixels=report['scale']['length_pixels'],
                          **report.get('selection_settings', {}))
            config['points_per_side'] = report['settings']['points_per_side']
            previous=report.get('preprocessing_config',dict(mode='none'))
            config.update(preprocessing_mode=previous['mode'],coarse_strength=previous.get('strength','medium'))
            if previous['mode']=='adjustable': config.update({k:v for k,v in previous.items() if k!='mode'})
        project = dict(id=project_id, name=Path(name.replace('\\','/')).name, width=w, height=h,
                       source_sha256=hashlib.sha256(content).hexdigest(),
                       original_dtype=original_dtype, normalization=normalization,
                       original_file='input/original'+suffix, config=config, scale_detection=scale,
                       seed=str(seed.resolve()) if seed else None, runs=[])
        with self.lock:
            self.projects[project_id] = project
            self.persist(project)
        return self.inspect(project_id)

    def inspect(self, project_id):
        with self.lock:
            if project_id not in self.projects: raise ValueError('Select the image again.')
            p = copy.deepcopy(self.projects[project_id])
        p['preview_url'] = f"/project-files/{project_id}/input/normalized.png"
        return p

    def validate_config(self, project, payload):
        values = {}
        for name in ['analysis_bottom','scale_um','scale_pixels','min_contrast','min_area_pixels','points_per_side']:
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                raise ValueError('Enter numeric region, scale, contrast and area values.')
            values[name] = value
        if int(values['analysis_bottom']) != values['analysis_bottom'] or not 32 <= values['analysis_bottom'] <= project['height']:
            raise ValueError('Set the bottom boundary within the image.')
        if not 0 < values['scale_um'] <= 1e6 or not 1 <= values['scale_pixels'] <= math.hypot(project['width'],project['height']):
            raise ValueError('Check the scale lengths in micrometers and pixels.')
        if not 0 <= values['min_contrast'] <= 255:
            raise ValueError('Contrast must be from 0 to 255.')
        if int(values['min_area_pixels']) != values['min_area_pixels'] or not 1 <= values['min_area_pixels'] <= project['width']*values['analysis_bottom']:
            raise ValueError('Minimum area must be a positive integer within the analysis area.')
        if values['points_per_side'] not in [16,32,48,64]:
            raise ValueError('Select a valid sampling density.')
        for key in ['analysis_bottom','min_area_pixels','points_per_side']: values[key] = int(values[key])
        if payload.get('scale_confirmed') is not True:
            raise ValueError('Confirm the scale length.')
        pre=self.preprocessing(project,payload)
        values.update(preprocessing_mode=pre['mode'],coarse_strength=pre.get('strength','medium'))
        if pre['mode']=='adjustable': values.update({k:v for k,v in pre.items() if k!='mode'})
        return values

    def preprocessing(self,project,payload):
        values={**project['config'], **payload}
        if values.get('preprocessing_mode')=='adjustable': return adjustable_config(values)
        return preprocessing_config(payload.get('preprocessing_mode',project['config'].get('preprocessing_mode','none')),
                                    payload.get('coarse_strength',project['config'].get('coarse_strength','medium')))

    def preview_preprocessing(self,project_id,payload):
        project=self.inspect(project_id)
        bottom=payload.get('analysis_bottom')
        if type(bottom) is not int or not 32<=bottom<=project['height']:
            raise ValueError('Check the bottom boundary.')
        gray=np.asarray(Image.open(self.root/project_id/'input/normalized.png'))[:bottom]
        pixels,meta=prepare_image(gray,self.preprocessing(project,payload))
        buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
        return dict(image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'),metadata=meta)

    def start(self, project_id, payload):
        with self.lock:
            project = self.projects.get(project_id)
            if project is None: raise ValueError('Choose an image first.')
            config = self.validate_config(project, payload)
            if self.running: raise ValueError('Wait for the current analysis to finish.')
            folder = self.root/project_id/'runs'
            number = max([int(p.name.split('_')[-1]) for p in folder.glob('run_*') if p.is_dir()], default=0)+1
            run_id = f'run_{number:04d}'
            (folder/run_id).mkdir(parents=True)
            job_id = secrets.token_hex(12)
            self.running = True
            self.job_timings[job_id] = AnalysisTiming()
            self.jobs[job_id] = dict(id=job_id, project_id=project_id, status='running', progress=1,
                                     message='Preparing analysis…')
            threading.Thread(target=self.run, args=(job_id, copy.deepcopy(project), config, run_id, number), daemon=True).start()
            return dict(job_id=job_id)

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs: raise ValueError('Analysis job not found. Check saved analyses.')
            result=copy.deepcopy(self.jobs[job_id])
            timing=self.job_timings.get(job_id)
            if timing:result.update(timing.snapshot(result['status'],10<=result['progress']<82))
            return result

    def batch_progress(self,job_id,completed,total):
        with self.lock:
            self.job_timings[job_id].batch(completed,total)
            self.progress(job_id,min(80,10+int(70*completed/total)),'Detecting pores…')

    def progress(self, job_id, percent, message):
        with self.lock:
            self.check_cancel(job_id)
            self.jobs[job_id].update(progress=percent,message=message)

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs: raise ValueError('Analysis job not found.')
            if self.jobs[job_id]['status']=='running':
                self.jobs[job_id].update(status='stopping',message='Stopping…')
            return copy.deepcopy(self.jobs[job_id])

    def check_cancel(self, job_id):
        with self.lock:
            if self.jobs[job_id]['status']=='stopping':
                raise OperationCancelled('Analysis stopped.')

    def run(self, job_id, project, config, run_id, number):
        folder = self.root/project['id']/'runs'/run_id
        try:
            # Serialize model operations and report rendering with manual edits.
            with self.editor.lock:
                self.check_cancel(job_id)
                self.editor.predictor = None
                self.editor.encoded_dataset = None
                device = self.editor.device
                model_checkpoint = checkpoint_path().resolve()
                model_digest = self.file_digest(model_checkpoint)
                release_device_cache(device)
                source = (self.root/project['id']/'input/normalized.png').resolve()
                gray = np.asarray(Image.open(source))[:config['analysis_bottom']]
                preprocessing=self.preprocessing(project,config)
                sam_gray,preprocessing_metadata=prepare_image(gray,preprocessing)
                sam_input=folder/'images/analysis_input.png'
                sam_input.parent.mkdir(exist_ok=True)
                Image.fromarray(sam_gray).save(sam_input)
                settings = dict(points_per_side=config['points_per_side'],points_per_batch=8,pred_iou_thresh=.7,
                                stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle')
                possible = [self.root/project['id']/'runs'/r['id'] for r in reversed(project['runs'])]
                if project.get('seed'): possible.append(Path(project['seed']))
                cached = None
                for candidate in possible:
                    if not (candidate/'report.json').is_file(): continue
                    report = json.loads((candidate/'report.json').read_text(encoding='utf-8'))
                    if (report['analysis_bottom_exclusive']==config['analysis_bottom'] and report['settings']==settings
                            and report.get('preprocessing_config',dict(mode='none'))==preprocessing
                            and report.get('checkpoint_sha256')==model_digest
                            and report.get('inference_device')==device
                            and (candidate/'sam_raw/pool.json').is_file()):
                        cached = candidate
                        break
                if cached:
                    self.progress(job_id,15,'Filtering cached masks…')
                    metadata = json.loads(read_artifact(cached,'raw_metadata.json').read_text())
                    with np.load(read_artifact(cached,'raw_masks.npz'),allow_pickle=False) as data:
                        raw = [dict(item,segmentation=data[f'mask_{i}'].copy()) for i,item in enumerate(metadata)]
                    pool = json.loads((cached/'sam_raw/pool.json').read_text(encoding='utf-8'))
                else:
                    self.progress(job_id,5,f'Loading SAM ({device.upper()})')
                    model = generator = None
                    try:
                        with inference_context(device):
                            model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(model_checkpoint),device=device,apply_postprocessing=False)
                            self.check_cancel(job_id)
                            generator = TraceGenerator(model,**settings)
                            generator.cancel_callback = lambda:self.check_cancel(job_id)
                            generator.batch_started_callback = self.job_timings[job_id].start_detection
                            generator.progress_callback = lambda n:self.batch_progress(job_id,n,math.ceil(settings['points_per_side']**2/8))
                            generator.generate(cv2.cvtColor(sam_gray,cv2.COLOR_GRAY2RGB))
                            pool = generator.pool
                    finally:
                        del generator,model
                        release_device_cache(device)
                    from pore_selection import decode_candidates
                    raw = decode_candidates(pool,.7,.85,device=device,score_dtype=torch.bfloat16 if device=='cuda' else torch.float32)
                    metadata = [{k:v for k,v in item.items() if k!='segmentation'} for item in raw]
                self.progress(job_id,82,'Filtering pores…')
                selected = select(pool,sam_gray,config['min_area_pixels'],config['min_contrast'],device,'two_stage_nested')
                self.check_cancel(job_id)
                write_artifact(folder,'raw_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                (folder/'sam_raw/pool.json').write_text(json.dumps(pool),encoding='utf-8')
                np.savez_compressed(write_artifact(folder,'raw_masks.npz'),**{f'mask_{i}':item['segmentation'] for i,item in enumerate(raw)})
                masks = {i+1:mask for i,mask in enumerate(selected['masks'])}
                np.savez_compressed(write_artifact(folder,'entrance_candidates.npz'),**{f'candidate_{i}':m for i,m in masks.items()})
                selection = dict(method='two_stage_nested',min_contrast=config['min_contrast'],min_area_pixels=config['min_area_pixels'])
                report = dict(image=str(source),source_name=project['name'],analysis_bottom_exclusive=config['analysis_bottom'],
                              scale=dict(label_um=config['scale_um'],length_pixels=config['scale_pixels'],um_per_pixel=config['scale_um']/config['scale_pixels'],label_source='user confirmed in Pore Editor'),
                              model='SAM 2.1 Small',settings=settings,selection_settings=selection,report_deferred=True,
                              checkpoint=str(model_checkpoint),checkpoint_sha256=model_digest,
                              inference_device=report.get('inference_device','unknown') if cached else device,
                              raw_mask_count=len(raw),entrance_candidate_count=len(masks),
                              raw_masks_reused_from=str(cached) if cached else None,
                              preprocessing_config=preprocessing,preprocessing_metadata=preprocessing_metadata,
                              sam_input_image=str(sam_input.resolve()),
                              overlay_relative_path='images/entrance_candidates_overlay.png',
                              status='unreviewed automatic candidates',normalization=project['normalization'],
                              selection='two_stage_nested',two_stage_settings=selected['settings'],
                              nested_settings=selected['nested_settings'],nested_decisions=selected['nested_decisions'],
                              candidates=[dict(candidate_id=i,area=int(mask.sum())) for i,mask in masks.items()])
                (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                self.progress(job_id,90,'Saving masks…')
                overlay(gray,[a['segmentation'] for a in raw]).save(write_artifact(folder,'raw_sam_overlay.png'))
            dataset = project['id']+'__'+run_id
            with self.lock:
                self.check_cancel(job_id)
                current = self.projects[project['id']]
                current['config'] = config
                current['runs'].append(dict(id=run_id,number=number,dataset=dataset,config=config))
                self.persist(current)
                self.jobs[job_id].update(status='complete',progress=100,message='Analysis complete.',dataset=dataset,candidate_count=len(masks))
                self.running=False
        except OperationCancelled:
            (folder/'report.json').unlink(missing_ok=True)
            with self.lock:
                self.jobs[job_id].update(status='cancelled',message='Analysis stopped.')
                self.running=False
        except Exception as exc:
            (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            with self.lock:
                self.jobs[job_id].update(status='failed',message=f'Analysis failed: {exc}')
                self.running=False

        finally:
            with self.lock:
                if job_id in self.job_timings:self.job_timings[job_id].snapshot(self.jobs[job_id]['status'],False)
