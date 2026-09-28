"""Additional 2D geometry and grayscale measurements for pore details."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter_crofton
from skimage.morphology import convex_hull_image

LABELS = {
    'angle_deg': 'Angle (°)', 'area_box_ratio': 'Area/Box',
    'brightness_max': 'Brightness max', 'brightness_mean': 'Brightness mean',
    'brightness_min': 'Brightness min', 'brightness_std': 'Brightness std dev',
    'centroid_x_um': 'Center X (µm)', 'centroid_y_um': 'Center Y (µm)',
    'convexity': 'Convexity', 'integral_density': 'Integral density',
    'mass_center_x_um': 'Mass center X (µm)', 'mass_center_y_um': 'Mass center Y (µm)',
    'perimeter_um': 'Perimeter (µm)', 'rectangle_bottom_um': 'Rectangle bottom (µm)',
    'rectangle_left_um': 'Rectangle left (µm)', 'rectangle_right_um': 'Rectangle right (µm)',
    'rectangle_top_um': 'Rectangle top (µm)', 'solidity': 'Solidity',
}
EXTRA_COLUMNS = [key for key in LABELS if key not in ('centroid_x_um', 'centroid_y_um', 'perimeter_um')] + ['complete']


def extra_metrics(cropped, yy, xx, covariance, scale, gray=None):
    """Coordinates use image x right/y down; box right/bottom are exclusive.

    Convexity compares Crofton perimeters of the convex hull and the filled
    exterior. Solidity is mask area / raster convex hull area. Brightness is
    from the calibrated source grayscale image, before optional preprocessing.
    Intensity-weighted centers describe brightness, not physical mass.
    """
    area = len(xx)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180) if not np.isclose(eigenvalues[0], eigenvalues[1]) else None
    hull = convex_hull_image(cropped)
    outer_perimeter = float(perimeter_crofton(ndi.binary_fill_holes(cropped), directions=4))
    hull_perimeter = float(perimeter_crofton(hull, directions=4))
    result = dict(angle_deg=angle, area_box_ratio=area/cropped.size,
                  convexity=float(np.clip(hull_perimeter/outer_perimeter, 0, 1)) if outer_perimeter else None,
                  solidity=area/int(hull.sum()),
                  rectangle_left_um=float(xx.min()*scale), rectangle_right_um=float((xx.max()+1)*scale),
                  rectangle_top_um=float(yy.min()*scale), rectangle_bottom_um=float((yy.max()+1)*scale),
                  brightness_max=None, brightness_mean=None, brightness_min=None, brightness_std=None,
                  integral_density=None, mass_center_x_um=None, mass_center_y_um=None)
    if gray is not None:
        values = gray[yy, xx].astype(np.float64)
        total = float(values.sum())
        result.update(brightness_max=float(values.max()), brightness_mean=float(values.mean()),
                      brightness_min=float(values.min()), brightness_std=float(values.std()), integral_density=total,
                      mass_center_x_um=float(np.dot(xx, values)/total*scale) if total else None,
                      mass_center_y_um=float(np.dot(yy, values)/total*scale) if total else None)
    return result
