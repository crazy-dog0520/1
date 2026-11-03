import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz
doc = fitz.open('paper_0912/joint_qualitative.pdf')
out = Path('.codex_tmp/zoom_face_inspect')
out.mkdir(exist_ok=True)
for item in doc[0].get_image_info(xrefs=True):
    print({k: v for k, v in item.items() if k != 'digest'})
for img in doc[0].get_images(full=True):
    xref = img[0]
    print('IMAGE', img, 'STREAM', len(doc.xref_stream(xref)))
    if xref == 21:
        print(doc.xref_object(xref))
        print('SAMPLES_MATCH_STREAM', fitz.Pixmap(doc, xref).samples == doc.xref_stream(xref))
    fitz.Pixmap(doc, xref).save(str(out / f'image_{xref}.png'))
