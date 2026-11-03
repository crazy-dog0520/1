import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz
import pypdfium2 as pdfium
from PIL import ImageStat

source = Path('paper_0912/joint_qualitative.pdf')
out = Path('.codex_tmp/qualitative_compatibility')
doc = fitz.open(source)
print('CONTENTS', doc[0].get_contents())
for xref in doc[0].get_contents():
    print('CONTENT_OBJECT', xref, doc.xref_object(xref))
    print('RAW_HEAD', doc.xref_stream_raw(xref)[:40], 'DECODED_HEAD', doc.xref_stream(xref)[:60])
for name, options in [('clean', dict(clean=True, garbage=3, deflate=True)),
                      ('uncompressed', dict(garbage=0, expand=255, deflate=False))]:
    output = out / f'repaired_{name}.pdf'
    doc.save(output, **options)
    p = pdfium.PdfDocument(str(output))
    im = p[0].render(scale=1.5).to_pil()
    im.save(out / f'repaired_{name}_pdfium.png')
    print(name, ImageStat.Stat(im).stddev)

# Rewrite all decoded streams with explicit, fresh compression metadata.
normalized = fitz.open(source)
for xref in range(1, normalized.xref_length()):
    if not normalized.xref_is_stream(xref):
        continue
    raw = normalized.xref_stream(xref)
    normalized.xref_set_key(xref, 'Filter', 'null')
    normalized.xref_set_key(xref, 'DecodeParms', 'null')
    normalized.update_stream(xref, raw, compress=False)
output = out / 'repaired_normalized.pdf'
normalized.save(output, garbage=0, deflate=False)
p = pdfium.PdfDocument(str(output))
im = p[0].render(scale=1.5).to_pil()
im.save(out / 'repaired_normalized_pdfium.png')
print('normalized', ImageStat.Stat(im).stddev)
