"""Reproducible grayscale inputs; these transforms do not infer physical depth."""
import cv2
import numpy as np


def adjustable_config(values):
    normalize = values.get('normalize_enabled', True)
    if type(normalize) is not bool:
        raise ValueError('Check normalization settings.')
    method = values.get('blur_method', 'none')
    if method not in ['none', 'gaussian', 'box', 'median', 'bilateral', 'kuwahara']:
        raise ValueError('Select a valid filter.')
    result = dict(mode='adjustable', normalize_enabled=normalize, blur_method=method)
    for name, default, maximum in [('background_strength', 0, 30), ('blur_strength', 2, 10)]:
        value = values.get(name, default)
        if type(value) not in (int, float) or not np.isfinite(value) or int(value) != value or not 0 <= value <= maximum:
            raise ValueError('Processing strength is out of range.')
        result[name] = int(value)
    return result


def kuwahara(gray, radius):
    """Classic four-quadrant minimum-variance Kuwahara, reflected boundaries.

    Reference: Kyprianidis et al. 2009, section on the original Kuwahara filter.
    Keep only the current best quadrant to bound memory for large inputs.
    """
    source = gray.astype(np.float32)
    best_variance = np.full(gray.shape, np.inf, np.float32)
    result = source.copy()
    size = radius + 1
    for anchor in [(radius, radius), (0, radius), (radius, 0), (0, 0)]:
        mean = cv2.boxFilter(source, -1, (size, size), anchor=anchor, borderType=cv2.BORDER_REFLECT)
        square = cv2.sqrBoxFilter(source, cv2.CV_32F, (size, size), anchor=anchor, borderType=cv2.BORDER_REFLECT)
        variance = np.maximum(square - mean * mean, 0)
        use = variance < best_variance
        result[use] = mean[use]
        best_variance[use] = variance[use]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def prepare_adjustable(gray, config):
    config = adjustable_config(config)
    pixels, metadata = gray.copy(), dict(config)
    if config['normalize_enabled']:
        lo, hi = np.percentile(gray, [2, 98])
        if hi > lo:
            pixels = np.clip((gray.astype(np.float32)-lo)*255/(hi-lo), 0, 255).astype(np.uint8)
        metadata.update(percentiles=[2, 98], low=float(lo), high=float(hi))
    radius = config['background_strength']
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        pixels = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
    metadata['background_radius_pixels'] = radius
    strength, method = config['blur_strength'], config['blur_method']
    if strength and method != 'none':
        size = 2*strength+1
        if method == 'gaussian': pixels = cv2.GaussianBlur(pixels, (0, 0), strength/2)
        elif method == 'box': pixels = cv2.blur(pixels, (size, size))
        elif method == 'median': pixels = cv2.medianBlur(pixels, size)
        elif method == 'bilateral': pixels = cv2.bilateralFilter(pixels, size, strength*10, strength*2)
        elif method == 'kuwahara': pixels = kuwahara(pixels, strength)
    return pixels, metadata


def preprocessing_config(mode='normalize', strength='medium'):
    if mode not in ['none','normalize','coarse','structure']:
        raise ValueError('Select a valid preprocessing method.')
    if strength not in ['weak','medium','strong','detail']:
        raise ValueError('Select a valid background removal strength.')
    config=dict(mode=mode)
    if mode in ['normalize','coarse']: config['percentiles']=[2,98]
    if mode in ['coarse','structure']:
        config.update(strength=strength,radius_fraction={'weak':.005,'medium':.009,'strong':.013,'detail':.009}[strength],
                      sigma=1.5,original_blend=.2 if strength=='detail' else 0.)
    return config


def prepare_image(gray, config):
    if config['mode']=='adjustable': return prepare_adjustable(gray, config)
    if config['mode']=='none':return gray.copy(),dict(mode='none')
    metadata=dict(config)
    normalized=gray.copy()
    if 'percentiles' in config:
        lo,hi=np.percentile(gray,config['percentiles'])
        normalized=np.clip((gray.astype(float)-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)
        metadata.update(low=float(lo),high=float(hi))
    if config['mode']=='normalize':return normalized,metadata
    radius=max(2,round(min(gray.shape)*config['radius_fraction']))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    coarse=cv2.GaussianBlur(cv2.morphologyEx(normalized,cv2.MORPH_OPEN,kernel),(0,0),config['sigma'])
    blend=config['original_blend']
    if blend:
        # Restore some fine intensity transitions instead of fully erasing thin rims.
        coarse=cv2.addWeighted(coarse,1-blend,normalized,blend,0)
    metadata['radius_pixels']=radius
    return coarse,metadata
