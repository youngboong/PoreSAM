"""Check the installed environment without downloading model weights."""

import importlib.metadata
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
import pandas
import scipy
import skimage
import tifffile
import torch
import torchvision
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    assert torch.cuda.is_available(), "CUDA GPU is unavailable"
    with torch.inference_mode():
        matrix = torch.eye(4, device="cuda")
        assert torch.allclose(matrix @ matrix, matrix)
        boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], device="cuda", dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], device="cuda")
        assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
        model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", device="cuda", apply_postprocessing=False)
        SAM2AutomaticMaskGenerator(model, points_per_side=8, points_per_batch=8, min_mask_region_area=0)
        del model
    image_path = Path(__file__).resolve().parents[1] / "data" / "PI35_5kx-4_bse.tif"
    image = tifffile.imread(image_path)
    assert image.ndim == 2 and image.size > 0
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "sam2": importlib.metadata.version("SAM-2"),
        "tiff_shape": list(image.shape),
        "tiff_dtype": str(image.dtype),
        "checks": "imports, CUDA matmul, CUDA NMS, SAM 2.1 construction, TIFF read passed",
        "trained_weights_loaded": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
