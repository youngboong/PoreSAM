# Build on Windows: python -m PyInstaller packaging/PoreSAM.spec --noconfirm
from pathlib import Path
import os
import torch
from PyInstaller.utils.hooks import collect_data_files,collect_submodules,copy_metadata
root=Path(SPECPATH).parent
if torch.version.cuda is not None:raise SystemExit('Build in the CPU-only pore-app-cpu environment; see DESKTOP.md.')
os.environ['MPLCONFIGDIR']=str(root/'build/matplotlib-cache')
weights=root/'checkpoints/sam2.1_hiera_small.pt'
if not weights.is_file():raise SystemExit('Place SAM 2.1 Small at '+str(weights))
datas=[(str(root/'ui'),'ui'),(str(weights),'checkpoints')]
datas+=collect_data_files('sam2')+copy_metadata('SAM-2')
hidden=collect_submodules('sam2')+['automate_pores','report_bundle','pore_details_export','app_paths','matplotlib.backends.backend_pdf','matplotlib.backends.backend_svg','matplotlib.backends.backend_agg']
a=Analysis([str(root/'scripts/pore_app.py')],pathex=[str(root/'scripts')],binaries=[],datas=datas,hiddenimports=hidden,
           excludes=['IPython','ipykernel','jupyter','notebook','PyQt5','PyQt6','PySide2','PySide6','playwright','pytest','tensorboard'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='PoreSAM',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='PoreSAM')
