"""Device selection must not initialize CUDA when CPU is requested."""
import unittest
from unittest.mock import patch
import torch
from sam_runtime import select_device, inference_context, release_device_cache


class RuntimeTests(unittest.TestCase):
    def test_auto_without_gpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(select_device(), 'cpu')
            with self.assertRaises(ValueError):
                select_device('cuda')

    def test_auto_with_gpu(self):
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(select_device(), 'cuda')

    def test_explicit_cpu_never_uses_cuda(self):
        with patch('torch.cuda.is_available', side_effect=AssertionError('CUDA queried')), \
             patch('torch.autocast', side_effect=AssertionError('autocast entered')), \
             patch('torch.cuda.empty_cache', side_effect=AssertionError('CUDA cache used')):
            device = select_device('cpu')
            with inference_context(device):
                result = torch.eye(3) @ torch.ones(3)
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.is_inference_mode_enabled())
            release_device_cache(device)


if __name__ == '__main__':
    unittest.main()
