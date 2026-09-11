"""Shared inference device selection for the editor and command-line analysis."""
from contextlib import contextmanager, nullcontext
import os

import torch


def select_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu, or cuda.')
    if requested == 'cpu':
        return 'cpu'
    available = torch.cuda.is_available()
    if requested == 'cuda' and not available:
        raise ValueError('CUDA GPU를 사용할 수 없습니다. --device cpu 또는 auto로 실행해주세요.')
    return 'cuda' if available else 'cpu'


def configure_device(requested='auto'):
    device = select_device(requested)
    if device == 'cpu':
        # Avoid excessive thread contention in interactive desktop use.
        torch.set_num_threads(min(torch.get_num_threads(), os.cpu_count() or 1, 8))
    return device


@contextmanager
def inference_context(device):
    # CPU uses float32; CUDA autocast must never be entered on a CPU-only build.
    precision = nullcontext()
    if device == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        precision = torch.autocast('cuda', dtype=dtype)
    with torch.inference_mode(), precision:
        yield


def release_device_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
