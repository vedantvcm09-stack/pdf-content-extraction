# app.py
# Streamlit UI for the Docling PDF Table + Figure Extractor pipeline.
# Run with: streamlit run app.py
# Wraps all pipeline logic from main.py — no LLMs, no APIs, pure Python.

# QUERY behaviour:
# - A single "Query" input box is used for BOTH table and figure extraction.
# - Get CSV : matches query against table context -> downloads best-match CSV.
# - Get Image: matches query against figure captions -> shows best-match PNG.
# - If no query is given for Get Image, ALL figures are shown (dump mode).


import io
import re
import zipfile
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

import store
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
    DEFAULT_TOP_N,
    DEFAULT_THRESHOLD,
    DEFAULT_IMAGE_SCALE,
)

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

uploaded_file = st.file_uploader(
    "📂 Upload a PDF",
    type=["pdf"],
    help="Upload any typed/digital PDF. For scanned PDFs enable OCR in the sidebar.",
)

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
    btn_csv = st.button(
        "📥 Get CSV" if not list_only else "📋 List Tables",
        type="primary",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

with col_img:
    btn_img = st.button(
        "🖼️ Get Image",
        type="secondary",
        disabled=(uploaded_file is None),
        use_container_width=True,
        help=(
            "With a query: returns the best-matched figure. "
            "Without a query: returns all figures as a ZIP."
        ),
    )

with col_split:
    btn_split = st.button(
        "📄 Split View",
        type="secondary",
        disabled=(uploaded_file is None),
        use_container_width=True,
        help="Show PDF pages with extracted figures side-by-side.",
    )

with col_table_split:
    btn_table_split = st.button(
        "📊 Table Split View",
        type="secondary",
        disabled=(uploaded_file is None),
        use_container_width=True,
        help="Show PDF pages with extracted tables side-by-side.",
    )

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
        except Exception:
            page_count = 0
    manifest = store.register_pdf(file_bytes, uploaded_file.name, page_count=page_count)
    return manifest["pdf_id"]

def _pic_page(conv_result, pic_idx: int) -> int:
    try:
        pic = conv_result.document.pictures[pic_idx]
        if pic.prov:
            return pic.prov[0].page_no
    except Exception:
        pass
    return 0

def _table_page(conv_result, table_idx: int) -> int:
    try:
        tbl = conv_result.document.tables[table_idx]
        if tbl.prov:
            return tbl.prov[0].page_no
    except Exception:
        pass
    return 0

# ---------------------------------------------------------------------------
# ACTION: Get Image
# ---------------------------------------------------------------------------

if btn_img and uploaded_file is not None:

    if not extract_images:
        st.warning(
            "**'Extract images' is OFF in the sidebar.** "
            "Enable it and click **Get Image** again — Docling needs to "
            "render figure bounding boxes at parse time."
        )
        st.stop()

    with st.spinner(f"Parsing PDF with Docling at image scale {image_scale:.1f}..."):
        try:
            conv_result, tmp_path = get_conv_result()
        except Exception as e:
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
        st.image(result["png_bytes"], caption=best_caption or None, use_container_width=True)

        # Persist to library
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
            st.warning(f"Library save failed: {_e}")

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
                st.image(png_bytes, caption=cap_text or display_name, use_container_width=True)
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

if btn_split and uploaded_file is not None:

    if not extract_images:
        st.warning(
            "**'Extract images' is OFF in the sidebar.** "
            "Enable it and click **Split View** again — Docling needs to "
            "render figure bounding boxes at parse time."
        )
        st.stop()

    with st.spinner(f"Parsing PDF with Docling at image scale {image_scale:.1f}..."):
        try:
            conv_result, tmp_path = get_conv_result()
        except Exception as e:
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
            st.image(buf, use_container_width=True)
        
        with col_figs:
            st.markdown("**Extracted Figures**")
            figures_on_page = page_figures.get(page_num, [])
            for pic_idx, pil_fig, caption in figures_on_page:
                fig_buf = io.BytesIO()
                pil_fig.save(fig_buf, format="PNG")
                png_bytes = fig_buf.getvalue()
                st.image(png_bytes, caption=caption or f"Figure {pic_idx + 1}", use_container_width=True)
                try:
                    pdf_id = ensure_registered(conv_result)
                    store.save_image(pdf_id, png_bytes, idx=pic_idx + 1, page=page_num, caption=caption)
                except Exception:
                    pass
        
        st.divider()

    st.success("Split view complete!")

# ---------------------------------------------------------------------------
# ACTION: Table Split View
# ---------------------------------------------------------------------------

if btn_table_split and uploaded_file is not None:

    with st.spinner("Parsing PDF with Docling..."):
        try:
            conv_result, tmp_path = get_conv_result()
        except Exception as e:
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
        except Exception as e:
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
            st.image(buf, use_container_width=True)
        
        with col_tables:
            st.markdown("**Extracted Tables**")
            tables_on_page = page_tables.get(page_num, [])
            for table_idx, table_item, table_context in tables_on_page:
                with st.expander(f"Table {table_idx + 1}", expanded=True):
                    if table_context:
                        st.markdown(f"**Section header:** {table_context['section_header'] or '_none_'}")
                        st.markdown(f"**Nearest text:** {table_context['nearest_text'] or '_none_'}")
                    
                    try:
                        temp_output_path = str(Path(tempfile.gettempdir()) / f"temp_table_{table_idx}.csv")
                        df = export_table_to_csv(conv_result, table_idx, temp_output_path)
                        st.dataframe(df, use_container_width=True)
                        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
                        try:
                            pdf_id = ensure_registered(conv_result)
                            store.save_table(
                                pdf_id,
                                df,
                                idx=table_idx + 1,
                                page=page_num,
                                section_header=(table_context or {}).get("section_header", ""),
                                nearest_text=(table_context or {}).get("nearest_text", ""),
                            )
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Failed to render table: {e}")
        
        st.divider()

    st.success("Table split view complete!")

# ---------------------------------------------------------------------------
# ACTION: Get CSV / List Tables
# ---------------------------------------------------------------------------

if btn_csv and uploaded_file is not None:

    if not list_only and not query.strip():
        st.warning(
            "Please enter a query, or enable **List-only mode** in the sidebar "
            "to browse all detected tables."
        )
        st.stop()

    with st.spinner("Parsing PDF with Docling... (may take 20-60 s for large files)"):
        try:
            conv_result, tmp_path = get_conv_result()
        except Exception as e:
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
                except Exception:
                    pass
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
                result_df = merge_and_export(conv_result, chain, temp_output_path)
            else:
                result_df = export_table_to_csv(conv_result, best_ctx["table_idx"], temp_output_path)
        except Exception as e:
            st.error(f"Export failed: {e}")
            st.stop()

    st.subheader("📄 Extracted Table")
    st.dataframe(result_df, use_container_width=True)
    st.caption(f"{result_df.shape[0]} rows × {result_df.shape[1]} columns")

    # Persist to library
    try:
        pdf_id = ensure_registered(conv_result)
        saved_name = store.save_table(
            pdf_id,
            result_df,
            idx=best_ctx["table_idx"] + 1,
            page=_table_page(conv_result, best_ctx["table_idx"]),
            section_header=best_ctx.get("section_header", ""),
            nearest_text=best_ctx.get("nearest_text", ""),
        )
        st.caption(f"💾 Saved to library as `{saved_name}`")
    except Exception as _e:
        st.warning(f"Library save failed: {_e}")

    csv_buffer = io.BytesIO()
    result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
        type="primary",
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
