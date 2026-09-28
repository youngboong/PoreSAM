"""Experimental parent/child pore selection using image evidence only.

Accepts saved SAM candidates or live decoded masks. No reference labels are used.
Resolution is a single pass over the original parents, so accepted children are
not recursively split. Defaults are development heuristics, not calibrated scores.
"""
import hashlib

import cv2
import numpy as np
from scipy import ndimage as ndi

from pore_conservative import rim_evidence


DEFAULTS = dict(
    min_score=.70, min_stability=.85, min_containment=.97,
    max_child_fraction=.85, min_child_fraction=.003,
    min_rim_support=.85, rim_contrast=5., rim_distance=3.,
    max_sibling_overlap=.10, min_children=2,
    min_coverage=.25, max_coverage=.85,
    min_core_fraction=.30, min_bright_fraction=.65,
    min_surface_ratio=.55, min_gap_contrast=20.,
    min_core_radius=3., core_radius_fraction=.02,
)


def _decode(segmentation, shape):
    if isinstance(segmentation, dict):
        if tuple(segmentation['size']) != tuple(shape):
            raise ValueError('Candidate RLE shape differs from image.')
        counts = np.asarray(segmentation['counts'], dtype=np.int64)
        if counts.ndim != 1 or np.any(counts < 0) or counts.sum() != np.prod(shape):
            raise ValueError('Invalid uncompressed RLE.')
        return np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts).reshape(shape, order='F').astype(bool)
    mask = np.asarray(segmentation, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError('Candidate mask shape differs from image.')
    return mask


def prepare_candidates(raw, gray, min_area=100, settings=None):
    """Keep nested hypotheses before global box NMS; store small local masks.

    Saved TraceGenerator pools already contain generator quality filtering and
    within-batch NMS. This cannot recover hypotheses absent from that pool.
    Exact mask duplicates are collapsed; overlapping scales are retained.
    """
    config = {**DEFAULTS, **(settings or {})}
    if np.asarray(gray).ndim != 2 or min_area < 1:
        raise ValueError('Expected a grayscale image and positive minimum area.')
    candidates = {}
    for index, item in enumerate(raw):
        score, stability = float(item['predicted_iou']), float(item['stability_score'])
        if not np.isfinite(score) or not np.isfinite(stability):
            continue
        if score <= config['min_score'] or stability < config['min_stability']:
            continue
        mask = _decode(item['segmentation'], gray.shape)
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        if not w or not h:
            continue
        local = mask[y:y+h, x:x+w]
        components, n = ndi.label(local)
        sizes = np.bincount(components.ravel()); sizes[0] = 0
        largest = components == sizes.argmax()
        if largest.sum() < .9 * local.sum():
            continue
        local = ndi.binary_fill_holes(largest)
        area = int(local.sum())
        if area < min_area or area > .2 * gray.size:
            continue
        key = (x, y, w, h, hashlib.sha256(np.packbits(local).tobytes()).digest())
        candidate = dict(pool_id=int(item.get('pool_id', index)), x=x, y=y, w=w, h=h,
                         mask=local, area=area, score=score, stability=stability)
        if key not in candidates or score > candidates[key]['score']:
            candidates[key] = candidate
    return sorted(candidates.values(), key=lambda c: (-c['area'], -c['score'], c['pool_id']))


def _rim(candidate, gray, smooth, config):
    # Pad the local ROI so the contour normals sample real surrounding pixels.
    pad = int(np.ceil(config['rim_distance'])) + 5
    x, y, w, h = (candidate[k] for k in ('x', 'y', 'w', 'h'))
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
    mask = np.zeros((y1-y0, x1-x0), bool)
    mask[y-y0:y-y0+h, x-x0:x-x0+w] = candidate['mask']
    roi = gray[y0:y1, x0:x1]
    ring = ndi.binary_dilation(mask, iterations=4) & ~mask
    contrast = float(roi[ring].mean() - roi[mask].mean()) if ring.any() else 0.
    evidence = rim_evidence(mask, smooth[y0:y1, x0:x1], config['rim_distance'])
    support = float(np.mean(evidence >= config['rim_contrast'])) if evidence.size else 0.
    return contrast, support


def resolve_nested(labels, candidates, gray, min_area=100, min_contrast=8, settings=None):
    """Replace a parent only when separate rims and a broad bright gap agree.

    Output retains unchanged instance IDs. Accepted children get new IDs and are
    clipped to their own parent. Thus this pass cannot create overlaps or add
    pixels outside the original segmentation. Logs include deferred decisions.
    """
    config = {**DEFAULTS, **(settings or {})}
    labels, gray = np.asarray(labels), np.asarray(gray, dtype=np.float32)
    if labels.shape != gray.shape or labels.ndim != 2:
        raise ValueError('Labels and grayscale image must have matching 2D shapes.')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Labels must be nonnegative integers.')
    if not np.isfinite(gray).all() or min_area < 1 or not np.isfinite(min_contrast):
        raise ValueError('Invalid image or minimum area/contrast.')
    out = labels.astype(np.uint32, copy=True)
    next_id = int(labels.max()) + 1
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.)
    evidence_cache = {}
    log = []
    for pid in np.unique(labels):
        if not pid:
            continue
        parent = labels == pid
        area = int(parent.sum())
        entry = dict(parent_id=int(pid), parent_area=area, replaced=False,
                     reason='insufficient_children', children=[], eligible_children=0)
        log.append(entry)
        if area < config['min_children'] * min_area:
            continue
        union = np.zeros(gray.shape, bool)
        children = []
        for c in candidates:
            fraction = c['area'] / area
            if not config['min_child_fraction'] <= fraction <= config['max_child_fraction']:
                continue
            x, y, w, h = (c[k] for k in ('x', 'y', 'w', 'h'))
            sl = np.s_[y:y+h, x:x+w]
            inside = c['mask'] & parent[sl]
            if inside.sum() < config['min_containment'] * c['area']:
                continue
            if c['pool_id'] not in evidence_cache:
                evidence_cache[c['pool_id']] = _rim(c, gray, smooth, config)
            contrast, support = evidence_cache[c['pool_id']]
            if contrast < max(min_contrast, config['rim_contrast']) or support < config['min_rim_support']:
                continue
            entry['eligible_children'] += 1
            if np.count_nonzero(inside & union[sl]) > config['max_sibling_overlap'] * inside.sum():
                continue
            # Keep the largest connected remainder after tiny sibling overlaps.
            parts, n = ndi.label(inside & ~union[sl])
            sizes = np.bincount(parts.ravel()); sizes[0] = 0
            if not n or sizes.max() < min_area or sizes.max() < .9 * inside.sum():
                continue
            novel = parts == sizes.argmax()
            union[sl] |= novel
            children.append((c, novel))
            entry['children'].append(dict(pool_id=c['pool_id'], area=int(novel.sum()),
                                          score=c['score'], stability=c['stability'],
                                          contrast=contrast, rim_support=support))
        if len(children) < config['min_children']:
            continue
        coverage = float(union.sum() / area)
        entry['coverage'] = coverage
        if not config['min_coverage'] <= coverage <= config['max_coverage']:
            entry['reason'] = 'coverage'; continue
        gap = parent & ~union
        radius = max(config['min_core_radius'], config['core_radius_fraction'] * np.sqrt(area))
        distance = cv2.distanceTransform(gap.astype(np.uint8), cv2.DIST_L2, 5)
        core = gap & (distance > radius)
        core_fraction = float(core.sum() / gap.sum())
        entry.update(core_radius=float(radius), gap_pixels=int(gap.sum()),
                     core_pixels=int(core.sum()), core_fraction=core_fraction)
        if not core.any() or core_fraction < config['min_core_fraction']:
            entry['reason'] = 'thin_gap'; continue
        pore_core = ndi.binary_erosion(union, iterations=2)
        if not pore_core.any():
            entry['reason'] = 'no_pore_core'; continue
        # Parent exterior is the local brightness reference; exclude other pores.
        ring = ndi.binary_dilation(parent, iterations=6) & ~parent & (labels == 0)
        if ring.sum() < 20:
            entry['reason'] = 'no_surface_reference'; continue
        dark = float(np.median(gray[pore_core]))
        surface = float(np.median(gray[ring]))
        gap_level = float(np.median(gray[core]))
        span = surface - dark
        bright_threshold = dark + config['min_surface_ratio'] * max(span, config['min_gap_contrast'])
        bright_fraction = float(np.mean(gray[core] >= bright_threshold))
        entry.update(pore_gray=dark, exterior_gray=surface, gap_gray=gap_level,
                     gap_contrast=gap_level-dark, surface_ratio=(gap_level-dark)/span if span>0 else None,
                     bright_threshold=bright_threshold, bright_fraction=bright_fraction)
        if span < config['min_gap_contrast'] or gap_level-dark < config['min_gap_contrast']:
            entry['reason'] = 'weak_gap_contrast'; continue
        if bright_fraction < config['min_bright_fraction']:
            entry['reason'] = 'dark_or_textured_gap'; continue
        out[parent] = 0
        for c, novel in children:
            sl = np.s_[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']]
            out[sl][novel] = next_id
            entry['children'][len(entry.get('output_ids', []))]['output_id'] = next_id
            entry.setdefault('output_ids', []).append(next_id)
            next_id += 1
        entry.update(replaced=True, reason='bright_surface_between_supported_rims', removed_pixels=int(gap.sum()))
    assert not np.any((out > 0) & (labels == 0))
    return out, log
