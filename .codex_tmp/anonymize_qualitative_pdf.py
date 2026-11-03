"""Blur face regions in a PDF's shared photograph; preserve vector overlays."""
from pathlib import Path
import argparse
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.codex_pdf_tools'))
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFilter, ImageChops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--circular', action='store_true',
                        help='Use regular circular masks rather than face-shaped masks.')
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve()
    assert not args.output.exists(), 'Do not overwrite an existing staged output.'
    doc = fitz.open(args.source)
    assert len(doc) == 1
    page = doc[0]
    occurrences = [item for item in page.get_image_info(xrefs=True)
                   if item['bbox'][1] > page.rect.height / 2]
    assert len(occurrences) == 3
    xrefs = {item['xref'] for item in occurrences}
    assert len(xrefs) == 1, 'All three panels must share the same original photo.'
    xref = xrefs.pop()
    assert all((item['width'], item['height']) == (339, 339)
               for item in occurrences)
    raw = doc.xref_stream(xref)
    assert len(raw) == 339 * 339 * 3
    original = Image.frombytes('RGB', (339, 339), raw)
    # Coordinates are in the embedded image, which the PDF displays vertically flipped.
    # Cover the foreground diver's face and the small background diver's head.
    if args.circular:
        # Equal horizontal and vertical radii; supersampling only smooths edges.
        scale = 4
        mask = Image.new('L', (339 * scale, 339 * scale), 0)
        drawing = ImageDraw.Draw(mask)
        for cx, cy, radius in [(143, 154, 90), (277, 60, 22)]:
            drawing.ellipse(tuple(value * scale for value in
                                  (cx-radius, cy-radius, cx+radius, cy+radius)),
                            fill=255)
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    else:
        mask = Image.new('L', original.size, 0)
        drawing = ImageDraw.Draw(mask)
        drawing.polygon([(75, 220), (133, 244), (180, 224), (209, 183),
                         (206, 130), (192, 80), (162, 61), (112, 92),
                         (73, 143)], fill=255)
        drawing.ellipse((260, 39, 295, 80), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(2))
    blurred = original.filter(ImageFilter.GaussianBlur(18))
    updated = Image.composite(blurred, original, mask)
    changes = ImageChops.difference(original, updated)
    assert changes.getbbox() is not None
    assert all(a == b for a, b, m in zip(original.getdata(), updated.getdata(),
                                       mask.getdata()) if m == 0)
    original_text = page.get_text()
    original_drawings = page.get_drawings()
    original_contents = page.read_contents()
    other_images = {item[0]: doc.xref_stream(item[0])
                    for item in page.get_images(full=True) if item[0] != xref}
    # Replace only decoded RGB samples, keeping dimensions, ICC profile and alpha mask.
    doc.update_stream(xref, updated.tobytes(), compress=True)
    assert page.read_contents() == original_contents
    assert all(doc.xref_stream(k) == v for k, v in other_images.items())
    doc.save(args.output, garbage=4, deflate=True)
    doc.close()
    check = fitz.open(args.output)
    assert check[0].get_text() == original_text
    assert check[0].get_drawings() == original_drawings
    assert check[0].read_contents() == original_contents
    new_occurrences = [item for item in check[0].get_image_info(xrefs=True)
                       if item['bbox'][1] > check[0].rect.height / 2]
    assert len({item['digest'] for item in new_occurrences}) == 1
    assert len(new_occurrences) == 3
    check[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(
        args.output.with_suffix('.preview.png'))
    print('Verified: identical face treatment in all three human panels;')
    print('all PDF text, vector drawings, page content and other photos unchanged.')
    print('Original SHA256:', hashlib.sha256(args.source.read_bytes()).hexdigest())
    print('Output:', args.output)
    print('Preview:', args.output.with_suffix('.preview.png'))


if __name__ == '__main__':
    main()
