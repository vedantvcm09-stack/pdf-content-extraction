# app.py
# Streamlit UI for the Docling PDF Table + Figure Extractor pipeline.
# Run with: streamlit run app.py
# Wraps all pipeline logic from main.py — no LLMs, no APIs, pure Python.

# QUERY behaviour:
# - A single "Query" input box is used for BOTH table and figure extraction.
# - Get CSV : matches query against table context -> downloads best-match CSV.
# - Get Image: matches query against figure captions -> shows best-match PNG.
# - If no query is given for Get Image, ALL figures are shown (dump mode).
# region Warning Suppression
import warnings
import logging

# Suppress Hugging Face transformers/lazy import warnings about __path__
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", message=".*Behavior may be different and this alias.*")

class TransformersWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Accessing `__path__` from" in msg or "this alias will be removed" in msg:
            return False
        return True

logging.getLogger("transformers").addFilter(TransformersWarningFilter())
# endregion

import io
import re
import time
import zipfile
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

import store
import vlm_helper
from main import (
    parse_pdf,
    preprocess_pdf_orientation,
    build_all_items,
    extract_table_contexts,
    extract_figure_contexts,
    rank_tables,
    find_cross_page_chain,
    export_table_smart,
    deduplicate_columns,
    collapse_to_data_columns,
    debug_document_labels,
    extract_figures_to_memory,
    extract_best_figure_to_memory,
    merge_and_export,
    export_table_to_csv,
    render_pdf_pages_as_images,
    group_figures_by_page,
    group_tables_by_page,
    crop_table_to_image_bytes,
    _table_cell_union_bbox,
    DEFAULT_TOP_N,
    DEFAULT_THRESHOLD,
    DEFAULT_IMAGE_SCALE,
)

# ---------------------------------------------------------------------------
# Module logger — writes to both console and app.log file
# ---------------------------------------------------------------------------

logger = logging.getLogger("app")
if not logger.handlers:
    _fh = logging.FileHandler(
        Path(__file__).resolve().parent / "app.log", encoding="utf-8"
    )
    _ch = logging.StreamHandler()
    _fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _fh.setFormatter(_fmt)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF Table & Figure Extractor",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📊 PDF Table & Figure Extractor")
st.caption("Powered by **Docling** — no LLMs, no APIs, pure Python.")
st.markdown(
    "Upload a PDF, enter a query, and extract a matching **table as CSV** "
    "or a matching **figure as PNG**. Leave the query blank to browse all tables "
    "or dump all figures."
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — Advanced Options
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Options")

    use_ocr = st.toggle(
        "Enable OCR",
        value=False,
        help="Turn ON only for scanned/image-based PDFs. Much slower.",
    )

    extract_images = st.toggle(
        "Extract images (figures)",
        value=False,
        help=(
            "Must be ON to use the Get Image button. "
            "Docling renders figure bounding boxes as PNGs at parse time."
        ),
    )

    image_scale = st.slider(
        "Image resolution scale",
        min_value=1.0,
        max_value=4.0,
        value=float(DEFAULT_IMAGE_SCALE),
        step=0.5,
        help=(
            "Controls Docling render scale for extracted figures. "
            "Higher values produce larger, sharper images but take more time and memory."
        ),
        disabled=not extract_images,
    )

    top_n = st.slider(
        "Top N matches to show",
        min_value=1,
        max_value=20,
        value=DEFAULT_TOP_N,
        help="How many candidate tables/figures to display.",
    )

    threshold = st.slider(
        "Low-confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Warn when the best match score falls below this value.",
    )

    list_only = st.toggle(
        "List-only mode (tables)",
        value=False,
        help="Show all detected tables with their context — no export.",
    )

    show_debug = st.toggle(
        "Show document labels (debug)",
        value=False,
        help="Print every Docling item label and page number.",
    )

    accurate = st.toggle(
        "Accurate table mode (slow)",
        value=False,
        help=(
            "Uses TableFormer ACCURATE mode — better row detection for borderless tables "
            "with multi-line cell content (e.g. tables with no horizontal separators). "
            "Expect ~2-4x slower parsing. Leave OFF for standard tables."
        ),
    )

    st.divider()
    st.markdown("### 🤖 Local Ollama VLM Settings")
    ollama_url = st.text_input(
        "Ollama Server URL",
        value="http://localhost:11434",
        help="Endpoint where your local Ollama instance is running."
    )
    
    ollama_info = vlm_helper.check_ollama_status(ollama_url)
    
    if ollama_info["status"] == "connected":
        st.caption("🟢 Connected to Ollama")
        available_models = ollama_info["models"]
        
        default_model = "llama3.2-vision"
        # If llama3.2-vision isn't pulled, look for other vision-like models or select the first model
        best_model_idx = 0
        for i, m in enumerate(available_models):
            if default_model in m or "vision" in m or "vl" in m:
                best_model_idx = i
                break
                
        vlm_model = st.selectbox(
            "VLM Model",
            options=available_models,
            index=best_model_idx if available_models else 0,
            help="Select the local model to run vision/text tasks."
        )
        if vlm_model:
            model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
            if model_info["has_vision"]:
                st.caption("📷 Model supports vision (multimodal)")
            else:
                st.warning("⚠️ Model does not support vision. Text-only fallback will be used for table summaries. Figure captions and visual structure enhancements will be disabled.")
    else:
        st.caption("🔴 Disconnected from Ollama")
        st.info("Make sure Ollama is running and has vision models pulled (e.g. `llama3.2-vision`).")
        vlm_model = None

    vlm_auto_run = st.toggle(
        "Auto-run VLM on Parse",
        value=False,
        help="Automatically describe figures and summarize tables on parse. May be slow on CPU.",
        disabled=(ollama_info["status"] != "connected")
    )

    vlm_logging = st.toggle(
        "Enable VLM Request Logging",
        value=True,
        help="Log all VLM HTTP requests/responses to console and local log file."
    )
    vlm_helper.set_logging_enabled(vlm_logging)

    st.divider()
    st.markdown(
        "**Tables:** Run list-only mode first to see context text near each "
        "table, then use that wording as your query.\n\n"
        "**Figures:** The query is matched against figure captions. "
        "Leave the query blank to get all figures as a ZIP.\n\n"
        f"**Current image scale:** `{image_scale:.1f}`"
    )

# ---------------------------------------------------------------------------
# Main UI — Upload + Query + Action buttons
# ---------------------------------------------------------------------------

# Initialize session state for active view
if "active_action" not in st.session_state:
    st.session_state.active_action = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader(
    "📂 Upload a PDF",
    type=["pdf"],
    help="Upload any typed/digital PDF. For scanned PDFs enable OCR in the sidebar.",
)

# Reset active view if new file is uploaded
current_file_name = uploaded_file.name if uploaded_file is not None else None
if current_file_name != st.session_state.uploaded_file_name:
    st.session_state.uploaded_file_name = current_file_name
    st.session_state.active_action = None

query = st.text_input(
    "🔍 Query",
    placeholder='e.g. "state-wise funds" or "rainfall distribution map"',
    help=(
        "Used for BOTH table and figure extraction. "
        "For tables: match against section heading / surrounding text. "
        "For figures: match against figure captions (Fig. X / text above-below image). "
        "Leave blank to list all tables or dump all figures."
    ),
    disabled=list_only,
)

col_csv, col_img, col_split, col_table_split = st.columns(4)
with col_csv:
    if st.button(
        "📥 Get CSV" if not list_only else "📋 List Tables",
        type="primary",
        disabled=(uploaded_file is None),
        width='stretch',
    ):
        st.session_state.active_action = "csv"

with col_img:
    if st.button(
        "🖼️ Get Image",
        type="secondary",
        disabled=(uploaded_file is None),
        width='stretch',
        help=(
            "With a query: returns the best-matched figure. "
            "Without a query: returns all figures as a ZIP."
        ),
    ):
        st.session_state.active_action = "img"

with col_split:
    if st.button(
        "📄 Split View",
        type="secondary",
        disabled=(uploaded_file is None),
        width='stretch',
        help="Show PDF pages with extracted figures side-by-side.",
    ):
        st.session_state.active_action = "split"

with col_table_split:
    if st.button(
        "📊 Table Split View",
        type="secondary",
        disabled=(uploaded_file is None),
        width='stretch',
        help="Show PDF pages with extracted tables side-by-side.",
    ):
        st.session_state.active_action = "table_split"

# ---------------------------------------------------------------------------
# Shared: cache the Docling parse result to avoid re-parsing on every click
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_parse(
    file_bytes: bytes,
    filename: str,
    use_ocr: bool,
    extract_images: bool,
    image_scale: float,
    accurate: bool,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    safe_pdf_path = preprocess_pdf_orientation(tmp_path)
    return parse_pdf(
        safe_pdf_path,
        use_ocr=use_ocr,
        extract_images=extract_images,
        image_scale=image_scale,
        accurate=accurate,
    ), safe_pdf_path

def get_conv_result():
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return cached_parse(file_bytes, uploaded_file.name, use_ocr, extract_images, image_scale, accurate)

def ensure_registered(conv_result=None) -> str:
    """Register the currently uploaded PDF in the persistent store and return pdf_id."""
    file_bytes = uploaded_file.getvalue()
    page_count = 0
    if conv_result is not None:
        try:
            page_count = len(getattr(conv_result.document, "pages", []) or [])
        except Exception as e:
            logger.warning(f"[ensure_registered] Failed to read page count: {e}")
            page_count = 0
    manifest = store.register_pdf(file_bytes, uploaded_file.name, page_count=page_count)
    return manifest["pdf_id"]

def _pic_page(conv_result, pic_idx: int) -> int:
    try:
        pic = conv_result.document.pictures[pic_idx]
        if pic.prov:
            return pic.prov[0].page_no
    except Exception as e:
        logger.debug(f"[_pic_page] Could not get page for pic_idx={pic_idx}: {e}")
    return 0

def _table_page(conv_result, table_idx: int) -> int:
    try:
        tbl = conv_result.document.tables[table_idx]
        if tbl.prov:
            return tbl.prov[0].page_no
    except Exception as e:
        logger.debug(f"[_table_page] Could not get page for table_idx={table_idx}: {e}")
    return 0

# ---------------------------------------------------------------------------
# ACTION: Get Image
# ---------------------------------------------------------------------------

if st.session_state.active_action == "img" and uploaded_file is not None:

    if not extract_images:
        st.warning(
            "**'Extract images' is OFF in the sidebar.** "
            "Enable it and click **Get Image** again — Docling needs to "
            "render figure bounding boxes at parse time."
        )
        st.stop()

    with st.spinner(f"Parsing PDF with Docling at image scale {image_scale:.1f}..."):
        try:
            t0 = time.time()
            conv_result, tmp_path = get_conv_result()
            logger.info(f"[Get Image] PDF parsed in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"[Get Image] Docling parse failed: {e}")
            st.error(f"Docling failed to parse the PDF: {e}")
            st.stop()

    pdf_stem = Path(uploaded_file.name).stem
    n_pics = len(conv_result.document.pictures)

    if n_pics == 0:
        st.error(
            "No figures were detected in this PDF.\n\n"
            "Possible reasons: text-only document, or figures use vector "
            "drawing commands that Docling could not isolate as PictureItems."
        )
        st.stop()

    all_items = build_all_items(conv_result)
    fig_contexts = extract_figure_contexts(conv_result, all_items)

    if query.strip():
        with st.spinner("Matching query against figure captions..."):
            result = extract_best_figure_to_memory(
                conv_result, query, fig_contexts, threshold, top_n
            )

        if not result or result.get("png_bytes") is None:
            st.error(
                "Could not render the best-matched figure. "
                "The figure may have no image data in this PDF."
            )
            st.stop()

        st.subheader(f"🏆 Top {min(top_n, len(result['ranked']))} figure matches")
        for rank, (score, ctx) in enumerate(result["ranked"][:top_n], 1):
            label = "BEST MATCH" if rank == 1 else f"#{rank}"
            cap = ctx.get("caption", "")[:80] or "none"
            sec = ctx.get("section_header", "")[:60] or "none"
            st.markdown(
                f"**{label}** — Figure {ctx['pic_idx'] + 1} | "
                f"Score: `{score:.2f}` | "
                f"Caption: _{cap}_ | "
                f"Section: _{sec}_"
            )

        if result["score"] < threshold:
            st.warning(
                f"**Low confidence** — Score {result['score']:.2f} < threshold {threshold:.2f}. "
                "The query may not closely match any figure caption. "
                "Try words from the caption text shown above."
            )

        st.divider()
        st.subheader("🖼️ Best-Match Figure")
        st.caption(f"Rendered at image scale `{image_scale:.1f}`")

        best_caption = result.get("caption") or result.get("nearest_text") or ""
        st.image(result["png_bytes"], caption=best_caption or None, width='stretch')

        # Persist to library
        saved_name = None
        try:
            pdf_id = ensure_registered(conv_result)
            saved_name = store.save_image(
                pdf_id,
                result["png_bytes"],
                idx=result["pic_idx"] + 1,
                page=_pic_page(conv_result, result["pic_idx"]),
                caption=best_caption,
            )
            st.caption(f"💾 Saved to library as `{saved_name}`")
        except Exception as _e:
            logger.warning(f"[Get Image] Library save failed for best-match figure: {_e}")
            st.warning(f"Library save failed: {_e}")

        # VLM captioning
        if saved_name:
            manifest = store.load_manifest(pdf_id)
            img_entry = next((e for e in manifest.get("images", []) if e["filename"] == saved_name), None)
            vlm_desc = img_entry.get("vlm_description", "") if img_entry else ""

            # Auto-run description removed to optimize speed; use manual description button below.

            if vlm_desc:
                st.info(f"🤖 **Ollama VLM Caption:**\n\n{vlm_desc}")
            elif vlm_model:
                model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                if st.button("🤖 Describe Figure with local VLM", key=f"vlm_desc_{result['pic_idx']}", disabled=not model_info["has_vision"]):
                    with st.spinner("🤖 Generating figure caption via local Ollama..."):
                        try:
                            desc_text = vlm_helper.describe_figure(result["png_bytes"], vlm_model, ollama_url)
                            store.save_image_vlm_description(pdf_id, saved_name, desc_text)
                            st.success("Caption generated!")
                            st.rerun()
                        except Exception as ve:
                            st.error(f"VLM caption generation failed: {ve}")

        q_slug = re.sub(r"[^\w]+", "_", query.strip()[:40]).strip("_")
        file_name = f"{pdf_stem}-figure-{result['pic_idx'] + 1}-{q_slug}.png"

        st.download_button(
            label="⬇️ Download this figure as PNG",
            data=result["png_bytes"],
            file_name=file_name,
            mime="image/png",
            type="primary",
        )

        st.success(f"Done! **{file_name}** is ready.")

    else:
        with st.spinner("Extracting all figures..."):
            figures = extract_figures_to_memory(conv_result)

        if not figures:
            st.warning(
                "No renderable figures found. "
                "The PDF may have figures as pure vector drawings with no raster data."
            )
            st.stop()

        n_figs = len(figures)
        st.success(f"Found **{n_figs} figure{'s' if n_figs != 1 else ''}** in this PDF.")

        ctx_map = {c["pic_idx"]: c for c in fig_contexts}

        # Persist all figures to the library
        try:
            pdf_id = ensure_registered(conv_result)
            for i, (_fname, png_bytes) in enumerate(figures):
                cap = ctx_map.get(i, {}).get("caption", "") or ctx_map.get(i, {}).get("nearest_text", "")
                store.save_image(
                    pdf_id,
                    png_bytes,
                    idx=i + 1,
                    page=_pic_page(conv_result, i),
                    caption=cap,
                )
            st.caption(f"💾 Saved {n_figs} figures to library.")
        except Exception as _e:
            logger.warning(f"[Get Image] Bulk library save failed: {_e}")
            st.warning(f"Library save failed: {_e}")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, png_bytes in figures:
                zf.writestr(f"{pdf_stem}-{fname}", png_bytes)
        zip_buffer.seek(0)

        st.download_button(
            label=f"⬇️ Download all {n_figs} figures as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{pdf_stem}-figures.zip",
            mime="application/zip",
            type="primary",
        )

        st.divider()
        st.subheader("🖼️ All Figures")
        st.caption(
            "Enter a query above and click **Get Image** to get just the most "
            "relevant figure instead of the full set."
        )
        st.caption(f"Rendered at image scale `{image_scale:.1f}`")

        cols = st.columns(2)
        for i, (fname, png_bytes) in enumerate(figures):
            with cols[i % 2]:
                pic_idx = i
                cap_text = ctx_map.get(pic_idx, {}).get("caption", "") or ctx_map.get(pic_idx, {}).get("nearest_text", "")
                display_name = f"{pdf_stem}-{fname}"
                st.image(png_bytes, caption=cap_text or display_name, width='stretch')
                st.download_button(
                    label=f"⬇️ {display_name}",
                    data=png_bytes,
                    file_name=display_name,
                    mime="image/png",
                    key=f"dl_fig_{i}",
                )

    st.stop()

# ---------------------------------------------------------------------------
# ACTION: Split View
# ---------------------------------------------------------------------------

if st.session_state.active_action == "split" and uploaded_file is not None:

    if not extract_images:
        st.warning(
            "**'Extract images' is OFF in the sidebar.** "
            "Enable it and click **Split View** again — Docling needs to "
            "render figure bounding boxes at parse time."
        )
        st.stop()

    with st.spinner(f"Parsing PDF with Docling at image scale {image_scale:.1f}..."):
        try:
            t0 = time.time()
            conv_result, tmp_path = get_conv_result()
            logger.info(f"[Split View] PDF parsed in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"[Split View] Docling parse failed: {e}")
            st.error(f"Docling failed to parse the PDF: {e}")
            st.stop()

    n_pics = len(conv_result.document.pictures)

    if n_pics == 0:
        st.error(
            "No figures were detected in this PDF.\n\n"
            "Possible reasons: text-only document, or figures use vector "
            "drawing commands that Docling could not isolate as PictureItems."
        )
        st.stop()

    with st.spinner("Rendering PDF pages and grouping figures..."):
        try:
            # Render PDF pages as images
            page_images = render_pdf_pages_as_images(tmp_path, scale=2.0)
            
            # Group figures by page
            page_figures = group_figures_by_page(conv_result)
        except Exception as e:
            logger.error(f"[Split View] Failed to render PDF pages or group figures: {e}")
            st.error(f"Failed to render PDF pages or group figures: {e}")
            st.stop()

    # Filter to only pages that have extracted figures
    pages_with_figures = [(pn, pi) for pn, pi in page_images if page_figures.get(pn)]

    st.success(
        f"Found **{n_pics} figures** across **{len(pages_with_figures)} page(s)** "
        f"(of {len(page_images)} total)."
    )
    st.divider()

    if not pages_with_figures:
        st.info("No pages with extracted figures to display.")
        st.stop()

    # Display each page with its figures in split view
    for page_num, pil_page in pages_with_figures:
        st.subheader(f"📄 Page {page_num}")
        
        # Create split layout
        col_page, col_figs = st.columns([1, 1])
        
        with col_page:
            st.markdown("**PDF Page**")
            buf = io.BytesIO()
            pil_page.save(buf, format="PNG")
            buf.seek(0)
            st.image(buf, width='stretch')
        
        with col_figs:
            st.markdown("**Extracted Figures**")
            figures_on_page = page_figures.get(page_num, [])
            for pic_idx, pil_fig, caption in figures_on_page:
                fig_buf = io.BytesIO()
                pil_fig.save(fig_buf, format="PNG")
                png_bytes = fig_buf.getvalue()
                st.image(png_bytes, caption=caption or f"Figure {pic_idx + 1}", width='stretch')
                
                saved_name = None
                pdf_id = None
                try:
                    pdf_id = ensure_registered(conv_result)
                    saved_name = store.save_image(pdf_id, png_bytes, idx=pic_idx + 1, page=page_num, caption=caption)
                except Exception as e:
                    logger.warning(f"[Split View] Figure save failed (page={page_num}, pic_idx={pic_idx}): {e}")
                
                vlm_desc = ""
                if saved_name and pdf_id:
                    try:
                        manifest = store.load_manifest(pdf_id)
                        img_entry = next((e for e in manifest.get("images", []) if e["filename"] == saved_name), None)
                        if img_entry:
                            vlm_desc = img_entry.get("vlm_description", "")
                    except Exception as e:
                        logger.warning(f"[Split View] VLM description lookup failed for {saved_name}: {e}")
                
                if vlm_desc:
                    st.info(f"🤖 **Ollama VLM Caption:**\n\n{vlm_desc}")
                elif vlm_model:
                    model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                    if st.button("🤖 Describe with local VLM", key=f"split_vlm_desc_{page_num}_{pic_idx}", disabled=not model_info["has_vision"]):
                        with st.spinner("🤖 Generating figure description..."):
                            try:
                                desc = vlm_helper.describe_figure(png_bytes, vlm_model, ollama_url)
                                store.save_image_vlm_description(pdf_id, saved_name, desc)
                                st.success("Description generated!")
                                st.rerun()
                            except Exception as ve:
                                st.error(f"VLM failed: {ve}")
        
        st.divider()

    st.success("Split view complete!")

# ---------------------------------------------------------------------------
# ACTION: Table Split View
# ---------------------------------------------------------------------------

if st.session_state.active_action == "table_split" and uploaded_file is not None:

    with st.spinner("Parsing PDF with Docling..."):
        try:
            t0 = time.time()
            conv_result, tmp_path = get_conv_result()
            logger.info(f"[Table Split View] PDF parsed in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"[Table Split View] Docling parse failed: {e}")
            st.error(f"Docling failed to parse the PDF: {e}")
            st.stop()

    n_tables = len(conv_result.document.tables)

    if n_tables == 0:
        st.error("No tables were detected in this PDF.")
        if not use_ocr:
            st.info("For scanned PDFs enable **OCR** in the sidebar and try again.")
        st.stop()

    with st.spinner("Rendering PDF pages and grouping tables..."):
        try:
            # Render PDF pages as images
            page_images = render_pdf_pages_as_images(tmp_path, scale=2.0)
            
            # Group tables by page
            page_tables = group_tables_by_page(conv_result)
            
            # Pre-compute contexts and populate page fields
            all_items = build_all_items(conv_result)
            contexts = extract_table_contexts(conv_result, all_items)
            for ctx in contexts:
                ctx["page"] = _table_page(conv_result, ctx["table_idx"])
        except Exception as e:
            logger.error(f"[Table Split View] Failed to render pages or group tables: {e}")
            st.error(f"Failed to render PDF pages or group tables: {e}")
            st.stop()

    # Filter to only pages that have extracted tables
    pages_with_tables = [(pn, pi) for pn, pi in page_images if page_tables.get(pn)]

    st.success(
        f"Found **{n_tables} tables** across **{len(pages_with_tables)} page(s)** "
        f"(of {len(page_images)} total)."
    )
    st.divider()

    if not pages_with_tables:
        st.info("No pages with extracted tables to display.")
        st.stop()

    # Calculate overall progress
    pdf_id = ensure_registered(conv_result)
    manifest = store.load_manifest(pdf_id)
    
    total_to_process = 0
    completed = 0
    for pn, _ in pages_with_tables:
        for t_idx, _, t_ctx in page_tables.get(pn, []):
            chain = find_cross_page_chain(t_ctx, contexts, conv_result, debug=False)
            if len(chain) > 1 and chain[0]["table_idx"] != t_idx:
                continue # Skip continuation tables
            
            total_to_process += 1
            saved_name = store.table_filename(manifest["stem"], t_idx + 1, pn)
            tab_entry = next((e for e in manifest.get("tables", []) if e["filename"] == saved_name), None)
            
            if tab_entry:
                is_merged = tab_entry.get("is_merged", False)
                is_vlm_refined = tab_entry.get("is_vlm_refined", False)
                has_enhanced = bool(tab_entry.get("vlm_enhanced_filename", ""))
                
                if len(chain) > 1:
                    # Cross page
                    if is_merged and (not vlm_auto_run or not vlm_model or is_vlm_refined):
                        completed += 1
                else:
                    # Single page
                    if not vlm_auto_run or not vlm_model or has_enhanced:
                        completed += 1

    if total_to_process > 0:
        pct = int(completed / total_to_process * 100)
        st.progress(completed / total_to_process, text=f"Processing Tables: {completed} / {total_to_process} ({pct}%)")

    # Display each page with its tables in split view
    for page_num, pil_page in pages_with_tables:
        st.subheader(f"📄 Page {page_num}")
        
        # Create split layout
        col_page, col_tables = st.columns([1, 1])
        
        with col_page:
            st.markdown("**PDF Page**")
            buf = io.BytesIO()
            pil_page.save(buf, format="PNG")
            buf.seek(0)
            st.image(buf, width='stretch')
        
        with col_tables:
            st.markdown("**Extracted Tables**")
            tables_on_page = page_tables.get(page_num, [])
            for table_idx, table_item, table_context in tables_on_page:
                # Find the cross-page chain
                chain = find_cross_page_chain(table_context, contexts, conv_result, debug=show_debug)
                is_cross_page = len(chain) > 1
                
                if is_cross_page:
                    is_anchor = (chain[0]["table_idx"] == table_idx)
                    if not is_anchor:
                        st.info(f"🔗 Continuation of Table {chain[0]['table_idx'] + 1} (displayed on Page {chain[0]['page']})")
                        continue

                title = f"Table {table_idx + 1}"
                if is_cross_page:
                    chain_ids = [c["table_idx"] + 1 for c in chain]
                    title += f" 🔗 CROSS-PAGE (merged Tables {chain_ids})"

                with st.expander(title, expanded=True):
                    if table_context:
                        st.markdown(f"**Section header:** {table_context['section_header'] or '_none_'}")
                        st.markdown(f"**Nearest text:** {table_context['nearest_text'] or '_none_'}")
                    
                    try:
                        temp_output_path = str(Path(tempfile.gettempdir()) / f"temp_table_{table_idx}.csv")
                        pdf_id = ensure_registered(conv_result)
                        manifest = store.load_manifest(pdf_id)
                        
                        # Generate expected filename using the safe stem from manifest
                        saved_name = store.table_filename(manifest["stem"], table_idx + 1, page_num)
                        
                        tab_entry = next((e for e in manifest.get("tables", []) if e["filename"] == saved_name), None)
                        
                        is_merged_in_manifest = tab_entry.get("is_merged", False) if tab_entry else False
                        is_vlm_refined = tab_entry.get("is_vlm_refined", False) if tab_entry else False
                        vlm_sum = tab_entry.get("vlm_summary", "") if tab_entry else ""
                        vlm_enhanced_fname = tab_entry.get("vlm_enhanced_filename", "") if tab_entry else ""

                        if is_cross_page:
                            # Cross-page merged table flow
                            if not is_merged_in_manifest:
                                # Not merged yet: either auto-run VLM enhanced merge or do heuristic merge by default
                                if vlm_auto_run and vlm_model:
                                    st.toast(f"🤖 Refining Table {table_idx + 1} cross-page merge...", icon="⏳")
                                    with st.spinner("🤖 Auto-merging cross-page table and refining with VLM..."):
                                        vlm_model_to_use = None
                                        try:
                                            model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                                            if model_info["has_vision"]:
                                                vlm_model_to_use = vlm_model
                                        except Exception as e:
                                            logger.warning(f"[Table Split View] VLM capability check failed, skipping VLM merge: {e}")
                                        
                                        df = merge_and_export(
                                            conv_result,
                                            chain,
                                            temp_output_path,
                                            vlm_model=vlm_model_to_use,
                                            ollama_url=ollama_url,
                                            pdf_path=tmp_path,
                                        )
                                        
                                        tb_bbox = _table_cell_union_bbox(table_item)
                                        saved_name = store.save_table(
                                            pdf_id,
                                            df,
                                            idx=table_idx + 1,
                                            page=page_num,
                                            section_header=(table_context or {}).get("section_header", ""),
                                            nearest_text=(table_context or {}).get("nearest_text", ""),
                                            bbox=list(tb_bbox) if tb_bbox else None,
                                        )
                                        
                                        manifest = store.load_manifest(pdf_id)
                                        for e in manifest.get("tables", []):
                                            if e["filename"] == saved_name:
                                                e["is_merged"] = True
                                                e["is_vlm_refined"] = (vlm_model_to_use is not None)
                                                break
                                        store.save_manifest(manifest)
                                        
                                        # Cross-page VLM refinement complete. Auto-run summary is skipped to optimize speed.
                                        st.rerun()
                                else:
                                    # Fallback to auto heuristic merge so user always sees combined output at anchor page
                                    with st.spinner("🔗 Merging cross-page table (heuristic)..."):
                                        df = merge_and_export(
                                            conv_result,
                                            chain,
                                            temp_output_path,
                                            vlm_model=None,
                                            ollama_url=None,
                                            pdf_path=None,
                                        )
                                        tb_bbox = _table_cell_union_bbox(table_item)
                                        saved_name = store.save_table(
                                            pdf_id,
                                            df,
                                            idx=table_idx + 1,
                                            page=page_num,
                                            section_header=(table_context or {}).get("section_header", ""),
                                            nearest_text=(table_context or {}).get("nearest_text", ""),
                                            bbox=list(tb_bbox) if tb_bbox else None,
                                        )
                                        manifest = store.load_manifest(pdf_id)
                                        for e in manifest.get("tables", []):
                                            if e["filename"] == saved_name:
                                                e["is_merged"] = True
                                                e["is_vlm_refined"] = False
                                                break
                                        store.save_manifest(manifest)
                                        st.rerun()
                            else:
                                out_path = store.table_path(pdf_id, saved_name)
                                try:
                                    df = pd.read_csv(out_path)
                                except Exception as e:
                                    logger.warning(f"[Table Split View] Cached merged CSV missing at {out_path}: {e}. Re-merging...")
                                    # Re-merge if file is missing
                                    df = merge_and_export(
                                        conv_result, chain,
                                        str(Path(tempfile.gettempdir()) / f"temp_table_{table_idx}.csv"),
                                        vlm_model=None, ollama_url=None, pdf_path=None,
                                    )

                            st.dataframe(df, width='stretch')
                            st.caption(f"Combined Table: {df.shape[0]} rows × {df.shape[1]} columns")

                            # VLM Assistant Tools for Cross-Page Tables
                            if vlm_model:
                                st.markdown("---")
                                st.caption("🤖 VLM Assistant Tools (Combined Table)")
                                col_sum_btn, col_enh_btn = st.columns(2)
                                
                                with col_sum_btn:
                                    if vlm_sum:
                                        st.info(f"**VLM Combined Table Summary:**\n\n{vlm_sum}")
                                    else:
                                        if st.button("🤖 Summarize Combined", key=f"split_vlm_sum_combined_{page_num}_{table_idx}"):
                                            with st.spinner("🤖 Summarizing combined table..."):
                                                try:
                                                    first_crop = None
                                                    try:
                                                        first_t_item = conv_result.document.tables[chain[0]["table_idx"]]
                                                        first_crop = crop_table_to_image_bytes(tmp_path, chain[0]["page"], first_t_item)
                                                    except Exception as e:
                                                        logger.warning(f"[Table Split View] Table crop for summary failed: {e}")
                                                    csv_str = df.to_csv(index=False)
                                                    summary_text = vlm_helper.summarize_table(csv_str, first_crop, vlm_model, ollama_url)
                                                    store.save_table_vlm_summary(pdf_id, saved_name, summary_text)
                                                    st.success("Summary generated!")
                                                    st.rerun()
                                                except Exception as ve:
                                                    st.error(f"VLM failed: {ve}")
                                                    
                                with col_enh_btn:
                                    if is_vlm_refined:
                                        st.success("✨ VLM Refinement Complete!")
                                    else:
                                        model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                                        if st.button("🤖 Refine Merge with VLM", key=f"split_vlm_enh_combined_{page_num}_{table_idx}", disabled=not model_info["has_vision"]):
                                            with st.spinner("🤖 Refining merged structure with VLM..."):
                                                try:
                                                    tb_bbox = _table_cell_union_bbox(table_item)
                                                    refined_df = merge_and_export(
                                                        conv_result,
                                                        chain,
                                                        temp_output_path,
                                                        vlm_model=vlm_model,
                                                        ollama_url=ollama_url,
                                                        pdf_path=tmp_path,
                                                    )
                                                    store.save_table(
                                                        pdf_id,
                                                        refined_df,
                                                        idx=table_idx + 1,
                                                        page=page_num,
                                                        section_header=(table_context or {}).get("section_header", ""),
                                                        nearest_text=(table_context or {}).get("nearest_text", ""),
                                                        bbox=list(tb_bbox) if tb_bbox else None,
                                                    )
                                                    manifest = store.load_manifest(pdf_id)
                                                    for e in manifest.get("tables", []):
                                                        if e["filename"] == saved_name:
                                                            e["is_merged"] = True
                                                            e["is_vlm_refined"] = True
                                                            break
                                                    store.save_manifest(manifest)
                                                    st.success("Refined structure complete!")
                                                    st.rerun()
                                                except Exception as ve:
                                                    st.error(f"VLM failed: {ve}")
                        else:
                            # Single table flow
                            df = export_table_to_csv(conv_result, table_idx, temp_output_path)
                            
                            try:
                                tb_bbox = _table_cell_union_bbox(table_item)
                                saved_name = store.save_table(
                                    pdf_id,
                                    df,
                                    idx=table_idx + 1,
                                    page=page_num,
                                    section_header=(table_context or {}).get("section_header", ""),
                                    nearest_text=(table_context or {}).get("nearest_text", ""),
                                    bbox=list(tb_bbox) if tb_bbox else None,
                                )
                            except Exception as e:
                                logger.warning(f"[Table Split View] Single table save failed (table_idx={table_idx}): {e}")

                            vlm_sum = ""
                            vlm_enhanced_fname = ""
                            if saved_name and pdf_id:
                                try:
                                    manifest = store.load_manifest(pdf_id)
                                    tab_entry = next((e for e in manifest.get("tables", []) if e["filename"] == saved_name), None)
                                    if tab_entry:
                                        vlm_sum = tab_entry.get("vlm_summary", "")
                                        vlm_enhanced_fname = tab_entry.get("vlm_enhanced_filename", "")
                                except Exception as e:
                                    logger.warning(f"[Table Split View] VLM data lookup failed for {saved_name}: {e}")

                            # Auto-run VLM structure enhancement if enabled and not already enhanced for single tables
                            if not vlm_enhanced_fname and vlm_model and vlm_auto_run:
                                model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                                if model_info["has_vision"]:
                                    with st.spinner("🤖 Auto-enhancing table structure via Ollama VLM..."):
                                        try:
                                            table_img_bytes = crop_table_to_image_bytes(tmp_path, page_num, table_item)
                                            if table_img_bytes:
                                                csv_str = df.to_csv(index=False)
                                                enhanced_csv = vlm_helper.enhance_table_structure(csv_str, table_img_bytes, vlm_model, ollama_url)
                                                enhanced_df = pd.read_csv(io.StringIO(enhanced_csv))
                                                store.save_table_vlm_enhanced(pdf_id, saved_name, enhanced_df)
                                                st.rerun()
                                        except Exception as ve:
                                            st.warning(f"VLM structure auto-run failed: {ve}")

                            if vlm_enhanced_fname:
                                t_orig, t_enh = st.tabs(["Original Extraction", "🤖 VLM Enhanced"])
                                with t_orig:
                                    st.dataframe(df, width='stretch')
                                    st.caption(f"Original: {df.shape[0]} rows × {df.shape[1]} columns")
                                with t_enh:
                                    try:
                                        enh_path = store.table_path(pdf_id, vlm_enhanced_fname)
                                        enh_df = pd.read_csv(enh_path)
                                        st.dataframe(enh_df, width='stretch')
                                        st.caption(f"VLM Enhanced: {enh_df.shape[0]} rows × {enh_df.shape[1]} columns")
                                    except Exception as e:
                                        st.error(f"Failed to load enhanced CSV: {e}")
                            else:
                                st.dataframe(df, width='stretch')
                                st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

                            # VLM Assistant Tools in Split View
                            if vlm_model:
                                st.markdown("---")
                                st.caption("🤖 VLM Assistant Tools")
                                col_sum_btn, col_enh_btn = st.columns(2)
                                
                                with col_sum_btn:
                                    if vlm_sum:
                                        st.info(f"**VLM Table Summary:**\n\n{vlm_sum}")
                                    else:
                                        if st.button("🤖 Summarize", key=f"split_vlm_sum_{page_num}_{table_idx}"):
                                            with st.spinner("🤖 Summarizing table..."):
                                                try:
                                                    table_img_bytes = crop_table_to_image_bytes(tmp_path, page_num, table_item)
                                                    csv_str = df.to_csv(index=False)
                                                    summary_text = vlm_helper.summarize_table(csv_str, table_img_bytes, vlm_model, ollama_url)
                                                    store.save_table_vlm_summary(pdf_id, saved_name, summary_text)
                                                    st.success("Summary generated!")
                                                    st.rerun()
                                                except Exception as ve:
                                                    st.error(f"VLM failed: {ve}")
                                                    
                                with col_enh_btn:
                                    if vlm_enhanced_fname:
                                        st.success("✨ Enhanced!")
                                    else:
                                        model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                                        if st.button("✨ Enhance Structure", key=f"split_vlm_enh_{page_num}_{table_idx}", disabled=not model_info["has_vision"]):
                                            with st.spinner("🤖 Enhancing structure..."):
                                                try:
                                                    table_img_bytes = crop_table_to_image_bytes(tmp_path, page_num, table_item)
                                                    if not table_img_bytes:
                                                        st.error("No visual table crop coordinates available.")
                                                    else:
                                                        csv_str = df.to_csv(index=False)
                                                        enhanced_csv = vlm_helper.enhance_table_structure(csv_str, table_img_bytes, vlm_model, ollama_url)
                                                        enhanced_df = pd.read_csv(io.StringIO(enhanced_csv))
                                                        store.save_table_vlm_enhanced(pdf_id, saved_name, enhanced_df)
                                                        st.success("Reconstruction complete!")
                                                        st.rerun()
                                                except Exception as ve:
                                                    st.error(f"VLM failed: {ve}")
                    except Exception as e:
                        logger.error(f"[Table Split View] Table render failed (table_idx={table_idx}): {e}")
                        st.error(f"Failed to render table: {e}")
        
        st.divider()

    st.success("Table split view complete!")

# ---------------------------------------------------------------------------
# ACTION: Get CSV / List Tables
# ---------------------------------------------------------------------------

if st.session_state.active_action == "csv" and uploaded_file is not None:

    if not list_only and not query.strip():
        st.warning(
            "Please enter a query, or enable **List-only mode** in the sidebar "
            "to browse all detected tables."
        )
        st.stop()

    with st.spinner("Parsing PDF with Docling... (may take 20-60 s for large files)"):
        try:
            t0 = time.time()
            conv_result, tmp_path = get_conv_result()
            logger.info(f"[Get CSV] PDF parsed in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"[Get CSV] Docling parse failed: {e}")
            st.error(f"Docling failed to parse the PDF: {e}")
            st.stop()

    n_tables = len(conv_result.document.tables)
    n_pics = len(conv_result.document.pictures)

    if n_tables == 0:
        st.error("No tables were detected in this PDF.")
        if not use_ocr:
            st.info("For scanned PDFs enable **OCR** in the sidebar and try again.")
        st.stop()

    st.success(
        f"Docling detected **{n_tables} table{'s' if n_tables != 1 else ''}** "
        f"and **{n_pics} picture{'s' if n_pics != 1 else ''}** in this PDF."
    )

    if show_debug:
        with st.expander("📋 Debug — All document item labels"):
            debug_lines = []
            for item, level in conv_result.document.iterate_items():
                page_no = 0
                try:
                    if item.prov:
                        page_no = item.prov[0].page_no
                except Exception as e:
                    logger.debug(f"[Get CSV] Debug: failed to get page_no for item: {e}")
                text_preview = getattr(item, "text", "")[:80] or ""
                debug_lines.append(f"[p{page_no}] {str(item.label):<30} {text_preview}")
            st.code("\n".join(debug_lines), language="text")

    all_items = build_all_items(conv_result)

    with st.spinner("Extracting table contexts..."):
        contexts = extract_table_contexts(conv_result, all_items)

    if list_only:
        st.subheader("📋 All detected tables")
        for ctx in contexts:
            chain = find_cross_page_chain(ctx, contexts, conv_result, debug=show_debug)
            chain_ids = [c["table_idx"] + 1 for c in chain if c["table_idx"] != ctx["table_idx"]]
            cross_tag = (
                f" 🔗 CROSS-PAGE (merges with Tables {chain_ids})"
                if len(chain) > 1 and chain[0]["table_idx"] == ctx["table_idx"]
                else ""
            )

            with st.expander(f"Table {ctx['table_idx'] + 1}{cross_tag}"):
                st.markdown(f"**Section header:** {ctx['section_header'] or '_none_'}")
                st.markdown(f"**Nearest text:** {ctx['nearest_text'] or '_none_'}")
                if ctx["full_context"]:
                    st.markdown(f"**Full context:** {ctx['full_context'][:300]}...")
        st.info(
            "Copy the **Nearest text** of your target table and use it as your query. "
            "Then disable List-only mode and click **Get CSV**."
        )
        st.stop()

    with st.spinner("Matching query against all tables..."):
        ranked = rank_tables(query, contexts)

    st.subheader(f"🏆 Top {min(top_n, len(ranked))} table matches")
    for rank, (score, ctx) in enumerate(ranked[:top_n], 1):
        label = "BEST MATCH" if rank == 1 else f"#{rank}"
        st.markdown(
            f"**{label}** — Table {ctx['table_idx'] + 1} | "
            f"Score: `{score:.2f}` | "
            f"Section: _{ctx['section_header'][:60] or 'none'}_ | "
            f"Nearest: _{ctx['nearest_text'][:60] or 'none'}_"
        )

    best_score, best_ctx = ranked[0]

    if best_score < threshold:
        st.warning(
            f"**Low confidence** — Score {best_score:.2f} < threshold {threshold:.2f}. "
            "Try **List-only mode** to see actual context text, then refine your query."
        )

    pdf_stem = Path(uploaded_file.name).stem
    q_slug = re.sub(r"[^\w]+", "_", query.strip()[:40]).strip("_")
    csv_name = f"{pdf_stem}_{q_slug}.csv"
    temp_output_path = str(Path(tempfile.gettempdir()) / csv_name)

    chain = find_cross_page_chain(best_ctx, contexts, conv_result, debug=show_debug)

    with st.spinner("Exporting table..."):
        try:
            if len(chain) > 1:
                ids = [c["table_idx"] + 1 for c in chain]
                st.info(f"🔗 Cross-page table — merging {len(chain)} fragments (Tables {ids}) into one CSV.")
                
                vlm_model_to_use = None
                if vlm_model:
                    try:
                        model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                        if model_info["has_vision"]:
                            vlm_model_to_use = vlm_model
                    except Exception as e:
                        logger.warning(f"[Get CSV] VLM capability check failed, skipping VLM for merge: {e}")

                result_df = merge_and_export(
                    conv_result,
                    chain,
                    temp_output_path,
                    vlm_model=vlm_model_to_use,
                    ollama_url=ollama_url,
                    pdf_path=tmp_path,
                )
            else:
                result_df = export_table_to_csv(conv_result, best_ctx["table_idx"], temp_output_path)
        except Exception as e:
            logger.error(f"[Get CSV] Export failed: {e}")
            st.error(f"Export failed: {e}")
            st.stop()

    # Persist to library
    saved_name = None
    try:
        pdf_id = ensure_registered(conv_result)
        table_item = conv_result.document.tables[best_ctx["table_idx"]]
        tb_bbox = _table_cell_union_bbox(table_item)
        saved_name = store.save_table(
            pdf_id,
            result_df,
            idx=best_ctx["table_idx"] + 1,
            page=_table_page(conv_result, best_ctx["table_idx"]),
            section_header=best_ctx.get("section_header", ""),
            nearest_text=best_ctx.get("nearest_text", ""),
            bbox=list(tb_bbox) if tb_bbox else None,
        )
        st.caption(f"💾 Saved to library as `{saved_name}`")
    except Exception as _e:
        logger.warning(f"[Get CSV] Library save failed: {_e}")
        st.warning(f"Library save failed: {_e}")

    # VLM table features
    vlm_sum = ""
    vlm_enhanced_fname = ""
    if saved_name:
        manifest = store.load_manifest(pdf_id)
        tab_entry = next((e for e in manifest.get("tables", []) if e["filename"] == saved_name), None)
        if tab_entry:
            vlm_sum = tab_entry.get("vlm_summary", "")
            vlm_enhanced_fname = tab_entry.get("vlm_enhanced_filename", "")

        # Auto-run VLM structure enhancement if enabled and not already saved
        if not vlm_enhanced_fname and vlm_model and vlm_auto_run:
            model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
            if model_info["has_vision"]:
                with st.spinner("🤖 Auto-enhancing table structure via Ollama VLM..."):
                    try:
                        table_item = conv_result.document.tables[best_ctx["table_idx"]]
                        table_page_no = _table_page(conv_result, best_ctx["table_idx"])
                        table_img_bytes = crop_table_to_image_bytes(tmp_path, table_page_no, table_item)
                        if table_img_bytes:
                            csv_str = result_df.to_csv(index=False)
                            enhanced_csv = vlm_helper.enhance_table_structure(csv_str, table_img_bytes, vlm_model, ollama_url)
                            enhanced_df = pd.read_csv(io.StringIO(enhanced_csv))
                            vlm_enhanced_fname = store.save_table_vlm_enhanced(pdf_id, saved_name, enhanced_df)
                            st.rerun()
                    except Exception as ve:
                        st.warning(f"VLM structure auto-run failed: {ve}")

    st.subheader("📄 Extracted Table")
    if vlm_enhanced_fname:
        t_orig, t_enh = st.tabs(["Original Extraction", "🤖 VLM Enhanced"])
        with t_orig:
            st.dataframe(result_df, width='stretch')
            st.caption(f"Original: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
        with t_enh:
            try:
                enh_path = store.table_path(pdf_id, vlm_enhanced_fname)
                enh_df = pd.read_csv(enh_path)
                st.dataframe(enh_df, width='stretch')
                st.caption(f"VLM Enhanced: {enh_df.shape[0]} rows × {enh_df.shape[1]} columns")
                
                # Provide download for enhanced CSV
                csv_buf_enh = io.BytesIO()
                enh_df.to_csv(csv_buf_enh, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="⬇️ Download Enhanced CSV",
                    data=csv_buf_enh.getvalue(),
                    file_name=vlm_enhanced_fname,
                    mime="text/csv",
                    key="dl_enhanced_btn_main",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Failed to load enhanced CSV: {e}")
    else:
        st.dataframe(result_df, width='stretch')
        st.caption(f"{result_df.shape[0]} rows × {result_df.shape[1]} columns")

    # VLM Actions row
    if saved_name and vlm_model:
        st.divider()
        st.markdown("### 🤖 VLM Assistant Tools")
        col_sum, col_enh = st.columns(2)
        
        with col_sum:
            if vlm_sum:
                st.info(f"**Ollama VLM Table Summary:**\n\n{vlm_sum}")
            else:
                if st.button("🤖 Summarize Table with local VLM", key="btn_vlm_sum_main"):
                    with st.spinner("🤖 Summarizing table via local Ollama..."):
                        try:
                            table_item = conv_result.document.tables[best_ctx["table_idx"]]
                            table_page_no = _table_page(conv_result, best_ctx["table_idx"])
                            table_img_bytes = crop_table_to_image_bytes(tmp_path, table_page_no, table_item)
                            csv_str = result_df.to_csv(index=False)
                            summary_text = vlm_helper.summarize_table(csv_str, table_img_bytes, vlm_model, ollama_url)
                            store.save_table_vlm_summary(pdf_id, saved_name, summary_text)
                            st.success("Summary generated!")
                            st.rerun()
                        except Exception as ve:
                            st.error(f"Table summarization failed: {ve}")
                            
        with col_enh:
            if vlm_enhanced_fname:
                st.success("✨ Table structure has been enhanced via local VLM!")
            else:
                model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                if st.button("✨ Enhance Table Structure via local VLM", key="btn_vlm_enh_main", disabled=not model_info["has_vision"]):
                    with st.spinner("🤖 Correcting structure and layout (this may take up to a minute)..."):
                        try:
                            table_item = conv_result.document.tables[best_ctx["table_idx"]]
                            table_page_no = _table_page(conv_result, best_ctx["table_idx"])
                            table_img_bytes = crop_table_to_image_bytes(tmp_path, table_page_no, table_item)
                            
                            if not table_img_bytes:
                                st.error("Could not obtain a visual crop of the table.")
                            else:
                                csv_str = result_df.to_csv(index=False)
                                enhanced_csv = vlm_helper.enhance_table_structure(csv_str, table_img_bytes, vlm_model, ollama_url)
                                enhanced_df = pd.read_csv(io.StringIO(enhanced_csv))
                                store.save_table_vlm_enhanced(pdf_id, saved_name, enhanced_df)
                                st.success("Reconstruction complete!")
                                st.rerun()
                        except Exception as ve:
                            st.error(f"Reconstruction failed: {ve}")

    st.divider()
    csv_buffer = io.BytesIO()
    result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download CSV" if not vlm_enhanced_fname else "⬇️ Download Original CSV",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
        type="secondary" if vlm_enhanced_fname else "primary",
        key="dl_original_btn_main"
    )

    st.success(
        f"Done! **{csv_name}** is ready — "
        f"{result_df.shape[0]} rows × {result_df.shape[1]} cols."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "Docling by IBM Research · "
    "No LLMs · No external APIs",
    unsafe_allow_html=True,
)
