"""Rebuild PDF graphics through SVG, retaining vector outlines and photo pixels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz
from PIL import Image, ImageChops, ImageStat

source = Path('paper_0912/joint_qualitative.pdf')
output = Path('.codex_tmp/joint_qualitative_compatible_20261002.pdf')
assert not output.exists()
doc = fitz.open(source)
svg = doc[0].get_svg_image(text_as_path=True)
# Outlined glyphs retain their exact shapes and do not depend on font mapping.
svgdoc = fitz.open('svg', svg.encode('utf-8'))
rebuilt = fitz.open('pdf', svgdoc.convert_to_pdf())
rebuilt.set_metadata({'title': 'Qualitative detection comparison', 'author': ''})
rebuilt.save(output, garbage=4, deflate=True)
check = fitz.open(output)
assert abs(check[0].rect.width-doc[0].rect.width) < .01
assert abs(check[0].rect.height-doc[0].rect.height) < .01
assert len(check[0].get_drawings()) > 82
for pdf, name in [(doc, 'before'), (check, 'after')]:
    pix = pdf[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(str(output.with_suffix(f'.{name}.png')))
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    if name == 'before':
        before = im
    else:
        after = im
diff = ImageChops.difference(before, after)
print('SOURCE_PATHS', len(doc[0].get_drawings()))
print('OUTPUT_PATHS', len(check[0].get_drawings()))
print('IMAGE_PLACEMENTS', len(check[0].get_image_info()))
print('MEAN_ABSOLUTE_PIXEL_DIFFERENCE', ImageStat.Stat(diff).mean)
print('OUTPUT', output)
