"""Show identical crops before/after segmentation for a first-pass review."""
from pathlib import Path

from PIL import Image, ImageDraw
from result_paths import read_artifact, write_artifact


root = Path(__file__).resolve().parents[1]
folder = root / "outputs/PI35_5kx-4_bse_first_pass"
original = Image.open(read_artifact(folder, "analysis_region_and_scale.png")).convert("RGB")
overlay = Image.open(read_artifact(folder, "entrance_candidates_overlay.png")).convert("RGB")
regions = [
    ("#16: large entrance", (395, 85, 645, 335)),
    ("#51: large entrance", (675, 440, 925, 690)),
    ("#70: partial / missed entrance", (55, 600, 305, 768)),
    ("#57: confirm target definition", (15, 500, 265, 750)),
]
canvas = Image.new("RGB", (1020, 580), "#222222")
draw = ImageDraw.Draw(canvas)
for index, (title, box) in enumerate(regions):
    x, y = (index % 2) * 510, (index // 2) * 290
    draw.text((x + 5, y + 5), title, fill="white")
    draw.text((x + 5, y + 23), "Original", fill="white")
    draw.text((x + 260, y + 23), "Automatic candidate", fill="white")
    canvas.paste(original.crop(box), (x + 5, y + 40))
    canvas.paste(overlay.crop(box), (x + 260, y + 40))
canvas.save(write_artifact(folder, "review_panels.png"))
