"""Check the installed environment without downloading model weights."""

import importlib.metadata
import argparse
import io
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
from sam_runtime import configure_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    device = configure_device(parser.parse_args().device)
    with torch.inference_mode():
        matrix = torch.eye(4, device=device)
        assert torch.allclose(matrix @ matrix, matrix)
        boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], device=device, dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], device=device)
        assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
        model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", device=device, apply_postprocessing=False)
        SAM2AutomaticMaskGenerator(model, points_per_side=8, points_per_batch=8, min_mask_region_area=0)
        del model
    image_path = Path(__file__).resolve().parents[1] / "data" / "PI35_5kx-4_bse.tif"
    if image_path.exists():
        image = tifffile.imread(image_path)
    else:
        stream = io.BytesIO()
        tifffile.imwrite(stream, np.arange(256, dtype=np.uint8).reshape(16,16))
        stream.seek(0)
        image = tifffile.imread(stream)
    assert image.ndim == 2 and image.size > 0
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else None,
        "sam2": importlib.metadata.version("SAM-2"),
        "tiff_shape": list(image.shape),
        "tiff_dtype": str(image.dtype),
        "checks": f"imports, {device} matmul, {device} NMS, SAM 2.1 construction, TIFF read passed",
        "trained_weights_loaded": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
