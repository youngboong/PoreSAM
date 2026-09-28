"""Resolve the verified deployment model without overriding an explicit path."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import app_paths


class AppCheckpointTests(unittest.TestCase):
    def test_default_manifest_override_and_invalid_name(self):
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{},clear=False):
            os.environ.pop('PORESAM_CHECKPOINT',None)
            root=Path(temporary);folder=root/'checkpoints';folder.mkdir()
            with patch.object(app_paths,'ASSET_ROOT',root):
                self.assertEqual(app_paths.checkpoint_path(),folder/'sam2.1_hiera_small.pt')
                manifest=folder/'default_model.json'
                manifest.write_text(json.dumps(dict(checkpoint='clean16.pt')))
                self.assertEqual(app_paths.checkpoint_path(),folder/'clean16.pt')
                os.environ['PORESAM_CHECKPOINT']=str(root/'custom.pt')
                self.assertEqual(app_paths.checkpoint_path(),(root/'custom.pt').resolve())
                os.environ.pop('PORESAM_CHECKPOINT')
                manifest.write_text(json.dumps(dict(checkpoint='../other.pt')))
                with self.assertRaises(ValueError):app_paths.checkpoint_path()


if __name__=='__main__':unittest.main()
