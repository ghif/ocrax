"""PDF utility module for extracting and rendering PDF pages as high-resolution images."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pypdfium2 as pdfium


def is_pdf(source: str | bytes | Path) -> bool:
    """Check if the source is a PDF file or bytes starting with PDF magic header."""
    if isinstance(source, (str, Path)):
        return str(source).lower().endswith(".pdf")
    if isinstance(source, bytes):
        return source.startswith(b"%PDF")
    return False


def get_pdf_page_count(source: bytes | Path | str) -> int:
    """Return total number of pages in the PDF."""
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {source}")
        pdf = pdfium.PdfDocument(source_path)
    else:
        pdf = pdfium.PdfDocument(source)
    return len(pdf)


def load_pdf_page(
    source: bytes | Path | str, page_index: int = 0, scale: float = 2.0
) -> tuple[Image.Image, int]:
    """Render a single page from a PDF document to a PIL Image.
    
    Args:
        source: PDF file path or raw PDF bytes.
        page_index: 0-based page index to render.
        scale: DPI rendering scale factor (2.0 = 144 DPI, optimal for OCR).
        
    Returns:
        Tuple of (PIL.Image rendered in RGB, total page count).
    """
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {source}")
        pdf = pdfium.PdfDocument(source_path)
    else:
        pdf = pdfium.PdfDocument(source)

    total_pages = len(pdf)
    if total_pages == 0:
        raise ValueError("PDF document has 0 pages")

    if page_index < 0 or page_index >= total_pages:
        raise IndexError(
            f"Page index {page_index} out of range for PDF with {total_pages} page(s)"
        )

    page = pdf[page_index]
    # Render page to bitmap at requested resolution scale
    bitmap = page.render(scale=scale)
    pil_img = bitmap.to_pil().convert("RGB")
    return pil_img, total_pages
