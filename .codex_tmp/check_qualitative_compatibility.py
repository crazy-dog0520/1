import sys
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz
import pypdfium2 as pdfium
from PIL import ImageStat, Image
import io

out = Path('.codex_tmp/qualitative_compatibility')
out.mkdir(exist_ok=True)
for path, name in [(Path('.codex_tmp/joint_qualitative_zoom_circular_checked_20261002.pdf'), 'verified_stage'),
                   (Path('paper_0912/joint_qualitative.pdf'), 'current'),
                   (Path('paper_0912/joint_qualitative.before_circular_blur_20261002.pdf'), 'original')]:
    pdf = pdfium.PdfDocument(str(path))
    image = pdf[0].render(scale=1.5).to_pil()
    image.save(out / f'{name}_pdfium.png')
    print(name, 'PDFIUM_PIXEL_STDDEV', ImageStat.Stat(image).stddev)

svg = Path('paper_0912/joint_qualitative.svg').read_bytes()
root = ET.fromstring(svg)
for i, element in enumerate(root.findall('.//{http://www.w3.org/2000/svg}image')):
    print('SVG_IMAGE', i, {k:v for k,v in element.attrib.items() if 'href' not in k})
    data = base64.b64decode(element.attrib['{http://www.w3.org/1999/xlink}href'].split(',',1)[1])
    Image.open(io.BytesIO(data)).save(out / f'svg_image_{i}.png')
svgdoc = fitz.open('svg', svg)
result = fitz.open('pdf', svgdoc.convert_to_pdf())
testpdf = out / 'original_svg_converted.pdf'
result.save(testpdf)
print('SVG_CONVERSION', len(result[0].get_drawings()), len(result[0].get_text()))
pdf = pdfium.PdfDocument(str(testpdf))
image = pdf[0].render(scale=1.5).to_pil()
image.save(out / 'original_svg_pdfium.png')
print('SVG_PDFIUM_PIXEL_STDDEV', ImageStat.Stat(image).stddev)
