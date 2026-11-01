import numpy as np
from PIL import Image

from ocrax.application.interactive_view import create_interactive_overlay_html
from ocrax.application.pipeline import OCRItem


def test_create_interactive_overlay_html():
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    box1 = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    crop1 = Image.new("RGB", (40, 20), color=(128, 128, 128))

    item1 = OCRItem(
        index=1,
        box=box1,
        det_score=0.88,
        text="Paracetamol",
        rec_score=0.96,
        crop=crop1,
    )

    html_code = create_interactive_overlay_html(img, [item1], alpha=0.3, line_width=2, show_badges=True)

    assert "data:image/jpeg;base64," in html_code
    assert 'class="ocr-poly"' in html_code
    assert 'data-text="Paracetamol"' in html_code
    assert 'data-rec="96.0%"' in html_code
    assert 'data-det="88.0%"' in html_code
    assert 'id="ocr-tooltip"' in html_code
    assert "addEventListener('mouseenter'" in html_code
    assert "viewBox=\"0 0 200 100\"" in html_code
