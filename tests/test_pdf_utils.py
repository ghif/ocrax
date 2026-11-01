from pathlib import Path
from PIL import Image

from ocrax.application.pdf_utils import get_pdf_page_count, is_pdf, load_pdf_page

TEST_PDF = Path("docs/blog.pdf")


def test_is_pdf():
    assert is_pdf("document.pdf")
    assert is_pdf("document.PDF")
    assert is_pdf(b"%PDF-1.7 ...")
    assert not is_pdf("image.png")
    assert not is_pdf(b"\x89PNG...")


def test_load_pdf_page():
    if not TEST_PDF.is_file():
        return

    page_count = get_pdf_page_count(TEST_PDF)
    assert page_count > 0

    img, total = load_pdf_page(TEST_PDF, page_index=0, scale=1.0)
    assert isinstance(img, Image.Image)
    assert total == page_count
    assert img.size[0] > 0
    assert img.size[1] > 0

    # Test loading from bytes
    pdf_bytes = TEST_PDF.read_bytes()
    img_bytes, total_bytes = load_pdf_page(pdf_bytes, page_index=0, scale=1.0)
    assert img_bytes.size == img.size
    assert total_bytes == total
