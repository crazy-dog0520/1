import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz

source = Path(sys.argv[1])
doc = fitz.open(source)
out = Path(tempfile.mkdtemp(prefix='ral_layout_oct02_'))
print('RENDER_DIR', out)
print('PAGES', len(doc), 'METADATA', doc.metadata)
for i, page in enumerate(doc, 1):
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(str(out / f'page_{i}.png'))
    print(f'\n--- PAGE {i} SIZE {page.rect} ---\n{page.get_text()}')
    fonts = [(f[3], bool(doc.extract_font(f[0])[3])) for f in page.get_fonts()]
    print('FONT_EMBEDDING', fonts)
    print('VECTOR_PATHS', len(page.get_drawings()), 'IMAGE_PLACEMENTS', len(page.get_image_info()))
