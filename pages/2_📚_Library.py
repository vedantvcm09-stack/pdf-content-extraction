# pages/2_📚_Library.py
# Browse-and-inspect view for all PDFs previously extracted via the
# main "Extract" page. Reads everything from the local store/ directory.
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
from pathlib import Path

import streamlit as st
import pandas as pd
import pymupdf

import store
import vlm_helper

# ---------------------------------------------------------------------------
# Module logger — writes to both console and app.log file
# ---------------------------------------------------------------------------

logger = logging.getLogger("library")
if not logger.handlers:
    _fh = logging.FileHandler(
        Path(__file__).resolve().parent.parent / "app.log", encoding="utf-8"
    )
    _ch = logging.StreamHandler()
    _fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _fh.setFormatter(_fmt)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)



st.set_page_config(
    page_title="Library — PDF Extractions",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Extraction Library")
st.caption(
    "Browse every PDF you've processed. Each entry stores the original PDF "
    "plus all extracted figures and tables on disk."
)

search_query = st.text_input("🔍 Global Search", placeholder="Search across all parsed PDFs (tables, summaries, metadata)...")
if search_query.strip():
    st.subheader(f"Search Results for '{search_query}'")
    query_lower = search_query.lower()
    results = []
    
    # Note: store.list_pdfs() returns manifests, but we haven't loaded it yet.
    # It's loaded below. I'll just call it here locally.
    _manifests = store.list_pdfs()
    for m in _manifests:
        index_p = store.extraction_dir(m["pdf_id"]) / "search_index.json"
        if not index_p.exists():
            continue
        try:
            import json
            index_data = json.loads(index_p.read_text(encoding="utf-8"))
        except:
            continue
            
        # check filename
        if query_lower in index_data.get("filename", "").lower():
            results.append({"pdf_id": m["pdf_id"], "filename": m["filename"], "type": "pdf", "snippet": "Filename match", "page": None})
            
        # check images
        for img in index_data.get("images", []):
            cap = img.get("vlm_description") or img.get("caption", "")
            if cap and query_lower in cap.lower():
                results.append({"pdf_id": m["pdf_id"], "filename": m["filename"], "type": "image", "snippet": cap, "page": img.get("page")})
                
        # check tables
        for tbl in index_data.get("tables", []):
            matched = False
            snippet = ""
            if query_lower in (tbl.get("section_header") or "").lower():
                matched = True; snippet = tbl.get("section_header")
            elif query_lower in (tbl.get("nearest_text") or "").lower():
                matched = True; snippet = tbl.get("nearest_text")
            elif query_lower in (tbl.get("vlm_summary") or "").lower():
                matched = True; snippet = tbl.get("vlm_summary")
            else:
                # search table data
                for row in tbl.get("data", []):
                    for val in row.values():
                        val_str = str(val)
                        if query_lower in val_str.lower():
                            matched = True
                            snippet = f"Row match: {val_str}"
                            break
                    if matched: break
            if matched:
                results.append({"pdf_id": m["pdf_id"], "filename": m["filename"], "type": "table", "snippet": snippet, "page": tbl.get("page"), "tbl_idx": tbl.get("idx")})
                
    if results:
        for r in results:
            with st.container(border=True):
                pg_str = f" (Page {r['page']})" if r['page'] else ""
                st.markdown(f"**{r['type'].title()}** in `{r['filename']}`{pg_str}")
                st.caption(r['snippet'][:500] + ("..." if len(r['snippet']) > 500 else ""))
    else:
        st.info("No matches found.")
    st.divider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _render_page(pdf_path: str, page_num: int, scale: float = 2.0) -> bytes:
    """Render a single PDF page to PNG bytes (1-based page_num)."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
        return pix.tobytes("png")
    finally:
        doc.close()


def _human_count(n: int, singular: str, plural: str = None) -> str:
    plural = plural or (singular + "s")
    return f"{n} {singular if n == 1 else plural}"


# ---------------------------------------------------------------------------
# Top: list of PDFs
# ---------------------------------------------------------------------------

manifests = store.list_pdfs()

if not manifests:
    st.info(
        "No extractions yet. Go to the **Extract** page, upload a PDF and run "
        "any of the extraction actions — results will appear here automatically."
    )
    st.stop()

# Sidebar: pick a PDF
labels = [
    f"{m['filename']}  ·  {_human_count(len(m['images']), 'image')}, "
    f"{_human_count(len(m['tables']), 'table')}"
    for m in manifests
]
id_by_label = {labels[i]: manifests[i]["pdf_id"] for i in range(len(manifests))}

with st.sidebar:
    st.header("Library")
    selected_label = st.radio(
        "Select a PDF",
        labels,
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 Refresh list", width='stretch'):
        st.rerun()

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
            
            vlm_logging = st.toggle(
                "Enable VLM Request Logging",
                value=True,
                help="Log all VLM HTTP requests/responses to console and local log file."
            )
            vlm_helper.set_logging_enabled(vlm_logging)
    else:
        st.caption("🔴 Disconnected from Ollama")
        st.info("Make sure Ollama is running and has vision models pulled (e.g. `llama3.2-vision`).")
        vlm_model = None

selected_id = id_by_label[selected_label]
manifest = store.load_manifest(selected_id)

if manifest is None:
    st.error("Manifest could not be loaded. The store may have been modified.")
    st.stop()

# ---------------------------------------------------------------------------
# Header block for selected PDF
# ---------------------------------------------------------------------------

h_left, h_right = st.columns([3, 1])
with h_left:
    st.subheader(f"📄 {manifest['filename']}")
    st.caption(
        f"`{manifest['pdf_id']}`  ·  uploaded {manifest.get('uploaded_at', 'unknown')}  ·  "
        f"{_human_count(len(manifest['images']), 'image')}, "
        f"{_human_count(len(manifest['tables']), 'table')}"
    )

with h_right:
    src_pdf = store.source_pdf_path(selected_id)
    if src_pdf.exists():
        st.download_button(
            "⬇️ Original PDF",
            data=src_pdf.read_bytes(),
            file_name=manifest["filename"],
            mime="application/pdf",
            width='stretch',
        )
    if st.button("🗑️ Delete this entry", width='stretch', type="secondary"):
        store.delete_pdf(selected_id)
        st.success("Deleted. Refreshing…")
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Tabs: Overview / Figures / Tables / Split View
# ---------------------------------------------------------------------------

tab_overview, tab_figs, tab_tables, tab_split = st.tabs(
    ["Overview", "🖼️ Figures", "📊 Tables", "📄 Split View"]
)

# ---- Overview --------------------------------------------------------------

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages", manifest.get("page_count") or "—")
    c2.metric("Figures", len(manifest["images"]))
    c3.metric("Tables", len(manifest["tables"]))

    # Quick file listing
    st.markdown("**Files on disk**")
    rows = []
    for e in manifest["images"]:
        rows.append({"type": "image", "page": e["page"], "filename": e["filename"], "info": (e.get("caption") or "")[:80]})
    for e in manifest["tables"]:
        rows.append({"type": "table", "page": e["page"], "filename": e["filename"],
                     "info": f"{e['rows']}×{e['cols']}  ·  {(e.get('section_header') or '')[:60]}"})
    if rows:
        rows.sort(key=lambda r: (r["page"], r["type"], r["filename"]))
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    else:
        st.info("No extracted artifacts yet for this PDF.")

# ---- Figures ---------------------------------------------------------------

with tab_figs:
    if not manifest["images"]:
        st.info("No figures saved for this PDF.")
    else:
        cols = st.columns(2)
        for i, entry in enumerate(manifest["images"]):
            img_path = store.image_path(selected_id, entry["filename"])
            if not img_path.exists():
                continue
            with cols[i % 2]:
                st.markdown(
                    f"**Page {entry['page']}  ·  Figure {entry['idx']}**  "
                    f"`{entry['filename']}`"
                )
                st.image(img_path.read_bytes(), width='stretch',
                         caption=entry.get("caption") or None)
                st.download_button(
                    f"⬇️ {entry['filename']}",
                    data=img_path.read_bytes(),
                    file_name=entry["filename"],
                    mime="image/png",
                    key=f"dl_img_{entry['filename']}",
                )
                
                # VLM Caption
                vlm_desc = entry.get("vlm_description", "")
                if vlm_desc:
                    st.info(f"🤖 **Ollama VLM Caption:**\n\n{vlm_desc}")
                elif vlm_model:
                    model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                    if st.button("🤖 Describe with local VLM", key=f"btn_desc_{entry['filename']}", disabled=not model_info["has_vision"]):
                        with st.spinner("🤖 Generating figure description..."):
                            try:
                                img_bytes = img_path.read_bytes()
                                desc = vlm_helper.describe_figure(img_bytes, vlm_model, ollama_url)
                                store.save_image_vlm_description(selected_id, entry["filename"], desc)
                                st.success("Caption generated!")
                                st.rerun()
                            except Exception as ve:
                                logger.error(f"[Library] VLM generation failed: {ve}")
                                st.error(f"VLM generation failed: {ve}")

# ---- Tables ----------------------------------------------------------------

with tab_tables:
    if not manifest["tables"]:
        st.info("No tables saved for this PDF.")
    else:
        for entry in manifest["tables"]:
            tbl_path = store.table_path(selected_id, entry["filename"])
            if not tbl_path.exists():
                continue
                
            if entry.get("is_continuation"):
                with st.expander(
                    f"Page {entry['page']}  ·  Table {entry['idx']} (Continuation of Table {entry.get('anchor_idx')})",
                    expanded=False,
                ):
                    st.info(f"🔗 This table merges with Table {entry.get('anchor_idx')} (displayed on Page {entry.get('anchor_page')})")
                continue

            title = f"Page {entry['page']}  ·  Table {entry['idx']}  ·  {entry['rows']}×{entry['cols']}  ·  `{entry['filename']}`"
            if entry.get("is_merged"):
                title = f"Page {entry['page']}  ·  Table {entry['idx']} 🔗 CROSS-PAGE  ·  {entry['rows']}×{entry['cols']}  ·  `{entry['filename']}`"

            with st.expander(title, expanded=False):
                if entry.get("section_header"):
                    st.markdown(f"**Section header:** {entry['section_header']}")
                if entry.get("nearest_text"):
                    st.markdown(f"**Nearest text:** {entry['nearest_text']}")
                vlm_enhanced_fname = entry.get("vlm_enhanced_filename", "")
                vlm_sum = entry.get("vlm_summary", "")

                try:
                    df = pd.read_csv(tbl_path)
                    if vlm_enhanced_fname:
                        t_orig, t_enh = st.tabs(["Original Extraction", "🤖 VLM Enhanced"])
                        with t_orig:
                            st.dataframe(df, width='stretch')
                            st.caption(f"Original: {df.shape[0]} rows × {df.shape[1]} columns")
                        with t_enh:
                            try:
                                enh_path = store.table_path(selected_id, vlm_enhanced_fname)
                                enh_df = pd.read_csv(enh_path)
                                st.dataframe(enh_df, width='stretch')
                                st.caption(f"VLM Enhanced: {enh_df.shape[0]} rows × {enh_df.shape[1]} columns")
                                
                                # Download enhanced CSV button
                                csv_buf_enh = io.BytesIO()
                                enh_df.to_csv(csv_buf_enh, index=False, encoding="utf-8-sig")
                                st.download_button(
                                    label="⬇️ Download Enhanced CSV",
                                    data=csv_buf_enh.getvalue(),
                                    file_name=vlm_enhanced_fname,
                                    mime="text/csv",
                                    key=f"dl_enh_tbl_{entry['filename']}",
                                    type="primary"
                                )
                            except Exception as e:
                                logger.error(f"[Library] Failed to load enhanced CSV: {e}")
                                st.error(f"Failed to load enhanced CSV: {e}")
                    else:
                        st.dataframe(df, width='stretch')
                        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
                except Exception as e:
                    logger.error(f"[Library] Could not read CSV: {e}")
                    st.error(f"Could not read CSV: {e}")

                # Download original CSV button
                st.download_button(
                    label=f"⬇️ Download CSV" if not vlm_enhanced_fname else f"⬇️ Download Original CSV",
                    data=tbl_path.read_bytes(),
                    file_name=entry["filename"],
                    mime="text/csv",
                    key=f"dl_tbl_{entry['filename']}",
                    type="secondary" if vlm_enhanced_fname else "primary"
                )

                # VLM assistant tools
                if vlm_model:
                    st.divider()
                    st.markdown("##### 🤖 VLM Assistant Tools")
                    col_sum, col_enh = st.columns(2)
                    
                    with col_sum:
                        if vlm_sum:
                            st.info(f"**Ollama VLM Table Summary:**\n\n{vlm_sum}")
                        else:
                            if st.button("🤖 Summarize Table with local VLM", key=f"sum_btn_{entry['filename']}"):
                                with st.spinner("🤖 Summarizing table via local Ollama..."):
                                    try:
                                        # Visual table crop inside library
                                        table_img_bytes = None
                                        bbox = entry.get("bbox")
                                        pdf_path_str = str(store.source_pdf_path(selected_id))
                                        if bbox and Path(pdf_path_str).exists():
                                            try:
                                                import fitz
                                                doc = fitz.open(pdf_path_str)
                                                page = doc[entry["page"] - 1]
                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3.0, 3.0))
                                                table_img_bytes = pix.tobytes("png")
                                                doc.close()
                                            except Exception as crop_err:
                                                logger.warning(f"[Library] Crop table failed: {crop_err}")
                                                
                                        csv_str = df.to_csv(index=False)
                                        summary_text = vlm_helper.summarize_table(csv_str, table_img_bytes, vlm_model, ollama_url)
                                        store.save_table_vlm_summary(selected_id, entry["filename"], summary_text)
                                        st.success("Summary generated!")
                                        st.rerun()
                                    except Exception as ve:
                                        logger.error(f"[Library] VLM table summarization failed: {ve}")
                                        st.error(f"VLM table summarization failed: {ve}")
                                        
                    with col_enh:
                        if vlm_enhanced_fname:
                            st.success("✨ Table structure has been enhanced via local VLM!")
                        else:
                            model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                            if st.button("✨ Enhance Table Structure via local VLM", key=f"enh_btn_{entry['filename']}", disabled=not model_info["has_vision"]):
                                with st.spinner("🤖 Correcting structure and layout (this may take up to a minute)..."):
                                    try:
                                        # Visual table crop inside library
                                        table_img_bytes = None
                                        bbox = entry.get("bbox")
                                        pdf_path_str = str(store.source_pdf_path(selected_id))
                                        if bbox and Path(pdf_path_str).exists():
                                            try:
                                                import fitz
                                                doc = fitz.open(pdf_path_str)
                                                page = doc[entry["page"] - 1]
                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3.0, 3.0))
                                                table_img_bytes = pix.tobytes("png")
                                                doc.close()
                                            except Exception as crop_err:
                                                logger.warning(f"[Library] Crop table failed: {crop_err}")
                                                
                                        if not table_img_bytes:
                                            st.error("Table visual crop coordinates not available in manifest. Make sure this table was parsed with the updated code.")
                                        else:
                                            csv_str = df.to_csv(index=False)
                                            enhanced_csv = vlm_helper.enhance_table_structure(csv_str, table_img_bytes, vlm_model, ollama_url)
                                            enhanced_df = pd.read_csv(io.StringIO(enhanced_csv))
                                            store.save_table_vlm_enhanced(selected_id, entry["filename"], enhanced_df)
                                            st.success("Reconstruction complete!")
                                            st.rerun()
                                    except Exception as ve:
                                        logger.error(f"[Library] Reconstruction failed: {ve}")
                                        st.error(f"Reconstruction failed: {ve}")

# ---- Split View ------------------------------------------------------------

with tab_split:
    # Build {page: {"images": [...], "tables": [...]}}
    pages_with_content: dict[int, dict] = {}
    for e in manifest["images"]:
        pages_with_content.setdefault(e["page"], {"images": [], "tables": []})["images"].append(e)
    for e in manifest["tables"]:
        pages_with_content.setdefault(e["page"], {"images": [], "tables": []})["tables"].append(e)

    if not pages_with_content:
        st.info("Nothing to split-view yet — extract figures or tables first.")
    elif not store.source_pdf_path(selected_id).exists():
        st.warning("Original PDF is missing from the store; cannot render pages.")
    else:
        sorted_pages = sorted(p for p in pages_with_content.keys() if p > 0)
        st.caption(
            f"Showing {len(sorted_pages)} page(s) with extracted content "
            f"(out of {manifest.get('page_count') or '?'} total)."
        )
        pdf_path_str = str(store.source_pdf_path(selected_id))

        for page_num in sorted_pages:
            st.markdown(f"### 📄 Page {page_num}")
            col_page, col_items = st.columns([1, 1])

            with col_page:
                st.markdown("**PDF Page**")
                try:
                    png = _render_page(pdf_path_str, page_num)
                    st.image(png, width='stretch')
                except Exception as e:
                    logger.error(f"[Library] Failed to render page {page_num}: {e}")
                    st.error(f"Failed to render page {page_num}: {e}")

            with col_items:
                items = pages_with_content[page_num]

                if items["images"]:
                    st.markdown("**Figures**")
                    for entry in items["images"]:
                        ip = store.image_path(selected_id, entry["filename"])
                        if ip.exists():
                            st.image(
                                ip.read_bytes(),
                                width='stretch',
                                caption=entry.get("caption") or entry["filename"],
                            )
                            
                            # VLM Caption in Library Split View
                            vlm_desc = entry.get("vlm_description", "")
                            if vlm_desc:
                                st.info(f"🤖 **Ollama VLM Caption:**\n\n{vlm_desc}")
                            elif vlm_model:
                                model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                                if st.button("🤖 Describe with local VLM", key=f"lib_split_desc_{entry['filename']}", disabled=not model_info["has_vision"]):
                                    with st.spinner("🤖 Generating figure description..."):
                                        try:
                                            img_bytes = ip.read_bytes()
                                            desc = vlm_helper.describe_figure(img_bytes, vlm_model, ollama_url)
                                            store.save_image_vlm_description(selected_id, entry["filename"], desc)
                                            st.success("Caption generated!")
                                            st.rerun()
                                        except Exception as ve:
                                            logger.error(f"[Library] VLM failed for figure description: {ve}")
                                            st.error(f"VLM failed: {ve}")

                if items["tables"]:
                    st.markdown("**Tables**")
                    for entry in items["tables"]:
                        if entry.get("is_continuation"):
                            st.info(f"🔗 Continuation of Table {entry.get('anchor_idx')} (displayed on Page {entry.get('anchor_page')})")
                            continue

                        tp = store.table_path(selected_id, entry["filename"])
                        if not tp.exists():
                            continue

                        title = f"Table {entry['idx']} · {entry['rows']}×{entry['cols']} · `{entry['filename']}`"
                        if entry.get("is_merged"):
                            title = f"Table {entry['idx']} 🔗 CROSS-PAGE · {entry['rows']}×{entry['cols']} · `{entry['filename']}`"

                        with st.expander(title, expanded=True):
                            try:
                                df = pd.read_csv(tp)
                                vlm_enhanced_fname = entry.get("vlm_enhanced_filename", "")
                                vlm_sum = entry.get("vlm_summary", "")
                                
                                if vlm_enhanced_fname:
                                    t_orig, t_enh = st.tabs(["Original Extraction", "🤖 VLM Enhanced"])
                                    with t_orig:
                                        st.dataframe(df, width='stretch')
                                        st.caption(f"Original: {df.shape[0]} rows × {df.shape[1]} columns")
                                    with t_enh:
                                        try:
                                            enh_path = store.table_path(selected_id, vlm_enhanced_fname)
                                            enh_df = pd.read_csv(enh_path)
                                            st.dataframe(enh_df, width='stretch')
                                            st.caption(f"VLM Enhanced: {enh_df.shape[0]} rows × {enh_df.shape[1]} columns")
                                        except Exception as e:
                                            logger.error(f"[Library] Failed to load enhanced CSV: {e}")
                                            st.error(f"Failed to load enhanced CSV: {e}")
                                else:
                                    st.dataframe(df, width='stretch')
                                
                                # VLM Assistant Tools in Library Split View
                                if vlm_model:
                                    st.markdown("---")
                                    st.caption("🤖 VLM Assistant Tools")
                                    col_sum_btn, col_enh_btn = st.columns(2)
                                    
                                    with col_sum_btn:
                                        if vlm_sum:
                                            st.info(f"**VLM Table Summary:**\n\n{vlm_sum}")
                                        else:
                                            if st.button("🤖 Summarize", key=f"lib_split_sum_{entry['filename']}"):
                                                with st.spinner("🤖 Summarizing table..."):
                                                    try:
                                                        table_img_bytes = None
                                                        bbox = entry.get("bbox")
                                                        pdf_path_str = str(store.source_pdf_path(selected_id))
                                                        if bbox and Path(pdf_path_str).exists():
                                                            try:
                                                                import fitz
                                                                doc = fitz.open(pdf_path_str)
                                                                page = doc[entry["page"] - 1]
                                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                                pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3.0, 3.0))
                                                                table_img_bytes = pix.tobytes("png")
                                                                doc.close()
                                                            except Exception as crop_err:
                                                                 logger.warning(f"[Library] Crop failed in split view: {crop_err}")
                                                                 
                                                        csv_str = df.to_csv(index=False)
                                                        summary_text = vlm_helper.summarize_table(csv_str, table_img_bytes, vlm_model, ollama_url)
                                                        store.save_table_vlm_summary(selected_id, entry["filename"], summary_text)
                                                        st.success("Summary generated!")
                                                        st.rerun()
                                                    except Exception as ve:
                                                        logger.error(f"[Library] VLM table summarization failed: {ve}")
                                                        st.error(f"VLM failed: {ve}")
                                                        
                                    with col_enh_btn:
                                        if entry.get("is_merged"):
                                            if entry.get("is_vlm_refined"):
                                                st.success("✨ VLM Refinement Complete!")
                                            else:
                                                st.info("Structure refined during merge.")
                                        elif vlm_enhanced_fname:
                                            st.success("✨ Enhanced!")
                                        else:
                                            model_info = vlm_helper.get_model_capabilities(ollama_url, vlm_model)
                                            if st.button("✨ Enhance Structure", key=f"lib_split_enh_{entry['filename']}", disabled=not model_info["has_vision"]):
                                                with st.spinner("🤖 Enhancing structure..."):
                                                    try:
                                                        table_img_bytes = None
                                                        bbox = entry.get("bbox")
                                                        pdf_path_str = str(store.source_pdf_path(selected_id))
                                                        if bbox and Path(pdf_path_str).exists():
                                                            try:
                                                                import fitz
                                                                doc = fitz.open(pdf_path_str)
                                                                page = doc[entry["page"] - 1]
                                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                                pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3.0, 3.0))
                                                                table_img_bytes = pix.tobytes("png")
                                                                doc.close()
                                                            except Exception as crop_err:
                                                                 logger.warning(f"[Library] Crop failed in split view: {crop_err}")
                                                                 
                                                        if not table_img_bytes:
                                                            st.error("No visual table crop coordinates available.")
                                                        else:
                                                            csv_str = df.to_csv(index=False)
                                                            enhanced_csv = vlm_helper.enhance_table_structure(csv_str, table_img_bytes, vlm_model, ollama_url)
                                                            enhanced_df = pd.read_csv(io.StringIO(enhanced_csv))
                                                            store.save_table_vlm_enhanced(selected_id, entry["filename"], enhanced_df)
                                                            st.success("Reconstruction complete!")
                                                            st.rerun()
                                                    except Exception as ve:
                                                        logger.error(f"[Library] VLM table enhancement failed: {ve}")
                                                        st.error(f"VLM failed: {ve}")
                            except Exception as e:
                                 logger.error(f"[Library] Could not read CSV: {e}")
                                 st.error(f"Could not read CSV: {e}")

            st.divider()
