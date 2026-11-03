"""Apply circular face blur to PDF photo streams without rasterizing overlays."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / '.codex_pdf_tools'))
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFilter, ImageChops


def blur_photo(raw, size, roi):
    original = Image.frombytes('RGB', size, raw)
    x0, y0, x1, y1 = roi
    sx, sy = size[0] / (x1-x0), size[1] / (y1-y0)
    assert abs(sx-sy) < 1e-9
    # Same face regions as the previously approved circular-blur version,
    # expressed in the upright 640 x 640 original-image coordinate system.
    circles = [(143*640/339, (338-154)*640/339, 90*640/339),
               (277*640/339, (338-60)*640/339, 22*640/339)]
    scale = 4
    mask = Image.new('L', (size[0]*scale, size[1]*scale), 0)
    draw = ImageDraw.Draw(mask)
    for cx, cy, radius in circles:
        bounds = ((cx-radius-x0)*sx*scale, (cy-radius-y0)*sy*scale,
                  (cx+radius-x0)*sx*scale, (cy+radius-y0)*sy*scale)
        draw.ellipse(bounds, fill=255)
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    blurred = original.filter(ImageFilter.GaussianBlur(18*640/339*sx))
    updated = Image.composite(blurred, original, mask)
    assert ImageChops.difference(original, updated).getbbox()
    assert all(a == b for a, b, m in zip(original.getdata(), updated.getdata(),
                                       mask.getdata()) if m == 0)
    return updated.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve()
    assert not args.output.exists()
    doc = fitz.open(args.source)
    assert len(doc) == 1
    page = doc[0]
    text, paths, contents = page.get_text(), page.get_drawings(), page.read_contents()
    rect = page.rect
    occurrences = page.get_image_info(xrefs=True)
    human_full = [i for i in occurrences if i['bbox'][1] > 420 and i['width'] == 312]
    human_zoom = [i for i in occurrences if i['bbox'][1] > 670 and i['width'] == 116]
    assert len(human_full) == 3 and len(human_zoom) == 2
    assert len({i['xref'] for i in human_full}) == 1
    assert len({i['xref'] for i in human_zoom}) == 1
    full_raw = doc.xref_stream(human_full[0]['xref'])
    zoom_raw = doc.xref_stream(human_zoom[0]['xref'])
    replacements = {
        full_raw: blur_photo(full_raw, (312, 312), (0, 0, 640, 640)),
        zoom_raw: blur_photo(zoom_raw, (116, 116), (460, 460, 640, 640)),
    }
    unchanged = {}
    changed = []
    # Also anonymize redundant copies left in the PDF resource dictionary.
    for item in page.get_images(full=True):
        xref = item[0]
        raw = doc.xref_stream(xref)
        if raw in replacements:
            # update_stream expects already decoded RGB bytes. Remove the old
            # PNG predictor so the PDF renderer does not decode them twice.
            doc.xref_set_key(xref, 'DecodeParms', 'null')
            doc.update_stream(xref, replacements[raw], compress=True)
            assert fitz.Pixmap(doc, xref).samples == replacements[raw]
            changed.append(xref)
        else:
            unchanged[xref] = raw
    assert len(changed) == 5
    assert page.read_contents() == contents
    assert all(doc.xref_stream(xref) == raw for xref, raw in unchanged.items())
    doc.save(args.output, garbage=4, deflate=True)
    doc.close()
    check = fitz.open(args.output)
    assert check[0].rect == rect
    assert check[0].get_text() == text
    assert check[0].get_drawings() == paths
    assert check[0].read_contents() == contents
    new_occurrences = check[0].get_image_info(xrefs=True)
    assert len(new_occurrences) == len(occurrences)
    for old, new in zip(occurrences, new_occurrences):
        assert old['bbox'] == new['bbox'] and old['transform'] == new['transform']
        if old['bbox'][1] < 420:
            assert old['digest'] == new['digest']
        else:
            assert old['digest'] != new['digest']
            assert fitz.Pixmap(check, new['xref']).samples in replacements.values()
    assert len({i['digest'] for i in new_occurrences if i['bbox'][1] > 420
                and i['width'] == 312}) == 1
    assert len({i['digest'] for i in new_occurrences if i['bbox'][1] > 670
                and i['width'] == 116}) == 1
    check[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(
        str(args.output.with_suffix('.preview.png')))
    print(json.dumps({'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      'output': str(args.output), 'changed_photo_resources': changed,
                      'vector_paths_preserved': len(paths),
                      'all_text_and_page_content_unchanged': True,
                      'fish_photos_unchanged': True,
                      'human_full_panels_identical': True,
                      'human_zoom_panels_identical': True}, indent=2))


if __name__ == '__main__':
    main()
