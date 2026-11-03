import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz
import pypdfium2 as pdfium
from PIL import ImageStat

source = Path(sys.argv[1])
doc = fitz.open(source)
original = fitz.open('paper_0912/joint_qualitative.before_circular_blur_20261002.pdf')
assert doc[0].read_contents() == original[0].read_contents()
assert doc[0].get_text() == original[0].get_text()
assert doc[0].get_drawings() == original[0].get_drawings()
assert len(doc[0].get_drawings()) == 82
for xref in range(1, doc.xref_length()):
    if doc.xref_is_stream(xref):
        assert doc.xref_stream(xref) is not None
out = Path('.codex_tmp/qualitative_compatibility')
composite = fitz.open()
page = composite.new_page(width=612, height=792)
page.show_pdf_page(fitz.Rect(54, 54, 558, 640), doc, 0)
composite.save(out / 'embedded_check.pdf')
for path, name in [(source, 'final'), (out / 'embedded_check.pdf', 'embedded')]:
    p = pdfium.PdfDocument(str(path))
    image = p[0].render(scale=1.5).to_pil()
    assert min(ImageStat.Stat(image).stddev) > 20
    image.save(out / f'{name}_pdfium.png')
    print(name, 'PDFium: nonblank, rendered OK')
    m = fitz.open(path)
    pix = m[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    print(name, 'MuPDF: rendered OK', pix.width, pix.height)
print('All 82 vector paths, labels, scores, and page content unchanged.')
