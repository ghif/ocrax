#!/usr/bin/env python3
"""OCRax Studio: Streamlit interactive web UI for pure JAX OCR."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

# Ensure JAX runs reliably on CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from ocrax.application.interactive_view import create_interactive_overlay_html
from ocrax.application.pdf_utils import get_pdf_page_count, is_pdf, load_pdf_page
from ocrax.application.pipeline import (
    DEFAULT_DET_CHECKPOINT,
    DEFAULT_MODEL_DIR,
    DEFAULT_REC_CHECKPOINT,
    InteractiveOCRPipeline,
)

REPO_ROOT = Path(__file__).resolve().parent

CHECKPOINT_OPTIONS = {
    "PP-OCRv6 RxHandBD Converged (Fine-Tuned - Recommended)": REPO_ROOT
    / "artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_converged",
    "PP-OCRv6 RxHandBD Fine-Tuned (Epoch 14)": REPO_ROOT
    / "artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_finetuned",
    "PP-OCRv6 Pretrained Zero-Shot": REPO_ROOT
    / "artifacts/checkpoints/ppocrv6_medium_rec_orbax",
}

SAMPLE_DOCUMENTS = {
    "None (Upload your own document)": None,
    "Prescription Crop 1: P1116 ('Dexter')": Path(
        "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD/Train_Set/P1116.jpg"
    ),
    "Prescription Crop 2: P1123 ('Paloxi')": Path(
        "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD/Train_Set/P1123.jpg"
    ),
    "Document Overview: Multi-Text Diagram (JPG)": REPO_ROOT
    / "docs/images/ppocrv6-system-overview.jpg",
    "Research Article: Page 1 (PDF)": REPO_ROOT
    / "docs/blog.pdf",
}


@st.cache_resource(show_spinner="Loading JAX detection and recognition models into memory...")
def get_pipeline(rec_checkpoint_str: str) -> InteractiveOCRPipeline:
    """Cache and load the pipeline in memory."""
    rec_ckpt = Path(rec_checkpoint_str)
    return InteractiveOCRPipeline(
        det_checkpoint=DEFAULT_DET_CHECKPOINT,
        rec_checkpoint=rec_ckpt,
        model_dir=DEFAULT_MODEL_DIR,
    )


def main():
    st.set_page_config(
        page_title="OCRax Studio",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 12px 16px;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">🔬 OCRax Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Interactive Clinical Document OCR powered entirely by <b>JAX / Flax NNX</b> '
        '(PP-OCRv6 Detection & Fine-Tuned Handwriting Recognition).'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Model & Detection Settings")

    rec_choice = st.sidebar.selectbox(
        "Recognition Model Checkpoint",
        options=list(CHECKPOINT_OPTIONS.keys()),
        index=0,
    )
    rec_checkpoint = CHECKPOINT_OPTIONS[rec_choice]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Text Detection Parameters")
    box_thresh = st.sidebar.slider(
        "Box Score Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Minimum confidence required for a detected polygon to be kept.",
    )
    det_thresh = st.sidebar.slider(
        "Segmentation Threshold",
        min_value=0.05,
        max_value=0.7,
        value=0.20,
        step=0.05,
        help="Probability threshold to binarize detection heatmaps.",
    )
    unclip_ratio = st.sidebar.slider(
        "Unclip Expansion Ratio",
        min_value=1.0,
        max_value=2.5,
        value=1.4,
        step=0.1,
        help="Expansion ratio for text polygon contours.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visualization Options")
    alpha = st.sidebar.slider(
        "Overlay Opacity", min_value=0.1, max_value=0.6, value=0.25, step=0.05
    )
    line_width = st.sidebar.slider(
        "Border Width", min_value=1, max_value=5, value=2, step=1
    )
    show_badges = st.sidebar.checkbox("Show Numbered Badges", value=True)

    st.sidebar.markdown("---")
    sample_choice = st.sidebar.selectbox(
        "📁 Load Sample Document",
        options=list(SAMPLE_DOCUMENTS.keys()),
        index=0,
    )

    # Load Pipeline
    try:
        pipeline = get_pipeline(str(rec_checkpoint))
    except Exception as exc:
        st.error(f"Failed to load OCRax pipeline: {exc}")
        return

    # Document input handling (Image or PDF)
    uploaded_file = st.file_uploader(
        "Upload a document, handwriting image, or PDF (PNG, JPG, TIFF, PDF)",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"],
    )

    target_image: Image.Image | None = None
    pdf_source: bytes | Path | None = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if is_pdf(uploaded_file.name) or is_pdf(file_bytes):
            pdf_source = file_bytes
        else:
            try:
                target_image = Image.open(io.BytesIO(file_bytes))
            except Exception as e:
                st.error(f"Failed to open uploaded image: {e}")
                return
    elif sample_choice != "None (Upload your own document)":
        sample_path = SAMPLE_DOCUMENTS[sample_choice]
        if sample_path and sample_path.is_file():
            if is_pdf(sample_path):
                pdf_source = sample_path
            else:
                target_image = Image.open(sample_path)

    # Handle PDF document rendering if source is a PDF
    if pdf_source is not None:
        try:
            total_pages = get_pdf_page_count(pdf_source)
            if total_pages > 1:
                col_p1, col_p2 = st.columns([1, 3])
                with col_p1:
                    selected_page = st.number_input(
                        "Select PDF Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1,
                    )
                with col_p2:
                    st.caption(
                        f"📄 Multi-page PDF detected ({total_pages} pages). "
                        f"Rendering Page {selected_page} at 144 DPI for OCR."
                    )
                page_idx = int(selected_page) - 1
            else:
                page_idx = 0

            target_image, _ = load_pdf_page(pdf_source, page_index=page_idx, scale=2.0)
        except Exception as exc:
            st.error(f"Error reading PDF document: {exc}")
            return

    if target_image is None:
        st.info("👆 Please upload an image/PDF or select a sample document from the sidebar to begin.")
        return

    # Process OCR
    with st.spinner("⚡ Running pure JAX detection and recognition..."):
        result = pipeline.process_image(
            target_image,
            box_thresh=box_thresh,
            det_thresh=det_thresh,
            unclip_ratio=unclip_ratio,
        )

    # Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Detected Regions", f"{len(result.items)}")
    with m2:
        avg_rec_conf = (
            sum(item.rec_score for item in result.items) / len(result.items)
            if result.items
            else 0.0
        )
        st.metric("Avg Recognition Conf", f"{avg_rec_conf * 100:.1f}%")
    with m3:
        st.metric("JAX Detection Latency", f"{result.t_det_ms:.1f} ms")
    with m4:
        st.metric("JAX Recognition Latency", f"{result.t_rec_ms:.1f} ms")
    with m5:
        st.metric("Total End-to-End Latency", f"{result.t_total_ms / 1000.0:.2f} s")

    st.markdown("---")

    # Split Panel Layout
    col_left, col_right = st.columns([1.1, 0.9])

    # Left Panel: Document View with Overlaid Text Detection / Segmentation
    with col_left:
        st.subheader("📄 Document View & Segmentation")

        orig_w, orig_h = result.original_size
        aspect_ratio = orig_h / max(1, orig_w)
        viewer_height = max(420, min(850, int(640 * aspect_ratio) + 30))

        interactive_html = create_interactive_overlay_html(
            target_image,
            result.items,
            alpha=alpha,
            line_width=line_width,
            show_badges=show_badges,
        )

        view_tab1, view_tab2, view_tab3 = st.tabs(
            [
                "🎯 Interactive (Hover Tooltips)",
                "🖼️ Static Overlay",
                "📄 Original Document",
            ]
        )
        with view_tab1:
            st.caption(
                "💡 *Hover your mouse over any text segment polygon to view its extracted text and confidence scores.*"
            )
            components.html(interactive_html, height=viewer_height, scrolling=True)
        with view_tab2:
            overlay_view = pipeline.draw_overlay(
                target_image,
                result.items,
                alpha=alpha,
                line_width=line_width,
                show_badges=show_badges,
            )
            st.image(
                overlay_view,
                caption=f"Overlaid View ({len(result.items)} text segments detected in reading order)",
                use_container_width=True,
            )
        with view_tab3:
            st.image(
                target_image,
                caption=f"Original Document ({orig_w}×{orig_h})",
                use_container_width=True,
            )

    # Right Panel: Extracted Text
    with col_right:
        st.subheader("📝 Extracted Text & Transcription")
        tab_text, tab_table, tab_json = st.tabs(
            ["📜 Transcribed Document", "📊 Region Breakdown", "💾 Export JSON"]
        )

        with tab_text:
            st.text_area(
                "Reading Order Transcription",
                value=result.full_text if result.full_text else "(No text detected)",
                height=320,
            )
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "📥 Download Text (.txt)",
                    data=result.full_text,
                    file_name="ocrax_transcription.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with tab_table:
            table_rows = []
            for item in result.items:
                table_rows.append({
                    "Index": item.index,
                    "Transcribed Text": item.text,
                    "Rec Confidence": f"{item.rec_score * 100:.1f}%",
                    "Det Confidence": f"{item.det_score * 100:.1f}%",
                    "Bounding Box (TL, TR, BR, BL)": str(item.box.astype(int).tolist()),
                })
            st.dataframe(table_rows, use_container_width=True, height=360)

            # Crop previews
            with st.expander("🔍 Inspect Individual Text Crops"):
                for item in result.items:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        st.image(item.crop, caption=f"Crop #{item.index}")
                    with c2:
                        st.markdown(f"**Text:** `{item.text}`")
                        st.caption(f"Recognition: {item.rec_score * 100:.1f}% | Detection: {item.det_score * 100:.1f}%")
                    with c3:
                        st.code(str(item.box.astype(int).tolist()), language="json")

        with tab_json:
            export_data = {
                "document_size": list(result.original_size),
                "timing_ms": {
                    "detection": result.t_det_ms,
                    "recognition": result.t_rec_ms,
                    "total": result.t_total_ms,
                },
                "items": [
                    {
                        "index": item.index,
                        "text": item.text,
                        "recognition_confidence": item.rec_score,
                        "detection_confidence": item.det_score,
                        "polygon": item.box.tolist(),
                    }
                    for item in result.items
                ],
                "full_text": result.full_text,
            }
            json_str = json.dumps(export_data, indent=2)
            st.json(export_data)
            st.download_button(
                "📥 Download Results (.json)",
                data=json_str,
                file_name="ocrax_ocr_results.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
