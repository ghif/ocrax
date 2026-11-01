"""Interactive HTML/SVG overlay generator with mouse-hover tooltips for document text segments."""

from __future__ import annotations

import base64
import html
import io
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from ocrax.application.pipeline import OCRItem


PALETTE = [
    ("#00e5ff", "0, 229, 255"),    # Cyan
    ("#10b981", "16, 185, 129"),   # Emerald
    ("#f59e0b", "245, 158, 11"),   # Amber
    ("#a855f7", "168, 85, 247"),   # Purple
    ("#ec4899", "236, 72, 153"),   # Pink
    ("#3b82f6", "59, 130, 246"),   # Blue
]


def create_interactive_overlay_html(
    image: Image.Image,
    items: list[OCRItem],
    alpha: float = 0.25,
    line_width: int = 2,
    show_badges: bool = True,
) -> str:
    """Generate a self-contained responsive HTML/SVG interactive viewer.
    
    Renders the document image with overlaid SVG polygons and provides smooth,
    styled floating tooltips displaying extracted text and confidence scores on mouse hover.
    """
    orig_w, orig_h = image.size

    # Convert image to JPEG base64 for fast in-memory embedding
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    polygons_svg = []
    badges_svg = []

    for item in items:
        color_idx = (item.index - 1) % len(PALETTE)
        stroke_color, rgb_str = PALETTE[color_idx]
        fill_color = f"rgba({rgb_str}, {alpha})"
        hover_fill = f"rgba({rgb_str}, {min(0.65, alpha + 0.35)})"

        points_str = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in item.box)
        esc_text = html.escape(item.text, quote=True)
        rec_pct = f"{item.rec_score * 100:.1f}%"
        det_pct = f"{item.det_score * 100:.1f}%"

        poly_svg = (
            f'<polygon points="{points_str}" '
            f'class="ocr-poly" '
            f'data-index="{item.index}" '
            f'data-text="{esc_text}" '
            f'data-rec="{rec_pct}" '
            f'data-det="{det_pct}" '
            f'data-stroke="{stroke_color}" '
            f'data-fill="{fill_color}" '
            f'data-hover-fill="{hover_fill}" '
            f'fill="{fill_color}" '
            f'stroke="{stroke_color}" '
            f'stroke-width="{line_width}">'
            f'<title>#{item.index}: {esc_text} (Rec: {rec_pct}, Det: {det_pct})</title>'
            f'</polygon>'
        )
        polygons_svg.append(poly_svg)

        if show_badges and len(item.box) >= 1:
            tl = item.box[0]
            bx = max(2.0, float(tl[0]))
            by = max(2.0, float(tl[1]) - 18.0)
            label = str(item.index)
            bw = max(18.0, len(label) * 8.0 + 8.0)
            bh = 16.0
            tx = bx + 4.0
            ty = by + 12.0
            badge = (
                f'<g class="badge-group" pointer-events="none">'
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'rx="3" fill="rgba(15, 23, 42, 0.92)" stroke="{stroke_color}" stroke-width="1"/>'
                f'<text x="{tx:.1f}" y="{ty:.1f}" fill="#ffffff" font-size="11" font-weight="700" '
                f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif">{label}</text>'
                f'</g>'
            )
            badges_svg.append(badge)

    polys_joined = "\n    ".join(polygons_svg)
    badges_joined = "\n    ".join(badges_svg)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      background: transparent;
      overflow-x: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .ocr-viewer-container {{
      position: relative;
      width: 100%;
      border-radius: 8px;
      overflow: hidden;
      background: #0f172a;
      border: 1px solid #334155;
      user-select: none;
    }}
    .ocr-svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .ocr-poly {{
      cursor: pointer;
      transition: fill 0.15s ease, stroke-width 0.15s ease, filter 0.15s ease;
    }}
    .ocr-poly:hover {{
      stroke-width: {max(line_width + 1.5, 3.5)}px !important;
      stroke: #ffffff !important;
      filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.85));
    }}
    #ocr-tooltip {{
      position: absolute;
      display: none;
      pointer-events: none;
      background: rgba(15, 23, 42, 0.96);
      color: #f8fafc;
      border: 1px solid #38bdf8;
      border-radius: 8px;
      padding: 9px 13px;
      font-size: 13px;
      line-height: 1.4;
      box-shadow: 0 10px 25px -4px rgba(0, 0, 0, 0.7), 0 0 14px rgba(56, 189, 248, 0.35);
      backdrop-filter: blur(8px);
      z-index: 9999;
      min-width: 190px;
      max-width: 320px;
      transition: opacity 0.1s ease;
    }}
  </style>
</head>
<body>
  <div id="ocr-container" class="ocr-viewer-container">
    <svg viewBox="0 0 {orig_w} {orig_h}" class="ocr-svg" xmlns="http://www.w3.org/2000/svg">
      <image href="data:image/jpeg;base64,{img_b64}" width="{orig_w}" height="{orig_h}" />
      {polys_joined}
      {badges_joined}
    </svg>
    <div id="ocr-tooltip"></div>
  </div>

  <script>
    (function() {{
      const container = document.getElementById('ocr-container');
      const tooltip = document.getElementById('ocr-tooltip');
      const polys = document.querySelectorAll('.ocr-poly');

      polys.forEach(poly => {{
        poly.addEventListener('mouseenter', (e) => {{
          const idx = poly.getAttribute('data-index');
          const text = poly.getAttribute('data-text');
          const rec = poly.getAttribute('data-rec');
          const det = poly.getAttribute('data-det');
          const stroke = poly.getAttribute('data-stroke') || '#38bdf8';
          const hoverFill = poly.getAttribute('data-hover-fill');
          
          if (hoverFill) {{
            poly.setAttribute('fill', hoverFill);
          }}

          tooltip.style.borderColor = stroke;
          tooltip.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; gap:8px;">
              <span style="background:${{stroke}}; color:#0f172a; font-size:11px; font-weight:800; padding:2px 7px; border-radius:4px;">#${{idx}}</span>
              <span style="font-size:11px; color:#94a3b8; font-weight:500;">Text Segment</span>
            </div>
            <div style="font-weight:700; font-size:14px; color:#ffffff; margin-bottom:7px; word-break:break-word;">
              ${{text ? text : '<em style="color:#64748b;">(No text)</em>'}}
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:11px; border-top:1px solid #334155; padding-top:5px;">
              <div>
                <span style="color:#94a3b8; display:block;">Rec Conf</span>
                <strong style="color:#38bdf8; font-size:12px;">${{rec}}</strong>
              </div>
              <div>
                <span style="color:#94a3b8; display:block;">Det Score</span>
                <strong style="color:#34d399; font-size:12px;">${{det}}</strong>
              </div>
            </div>
          `;
          tooltip.style.display = 'block';
        }});

        poly.addEventListener('mousemove', (e) => {{
          const rect = container.getBoundingClientRect();
          let x = e.clientX - rect.left + 16;
          let y = e.clientY - rect.top + 16;
          
          const tooltipWidth = 240;
          const tooltipHeight = 90;

          if (x + tooltipWidth > rect.width) {{
            x = e.clientX - rect.left - tooltipWidth - 12;
          }}
          if (y + tooltipHeight > rect.height) {{
            y = e.clientY - rect.top - tooltipHeight - 12;
          }}
          tooltip.style.left = Math.max(6, x) + 'px';
          tooltip.style.top = Math.max(6, y) + 'px';
        }});

        poly.addEventListener('mouseleave', () => {{
          const origFill = poly.getAttribute('data-fill');
          if (origFill) {{
            poly.setAttribute('fill', origFill);
          }}
          tooltip.style.display = 'none';
        }});
      }});
    }})();
  </script>
</body>
</html>
"""
    return html_content
