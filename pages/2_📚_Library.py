# pages/2_📚_Library.py
# Browse-and-inspect view for all PDFs previously extracted via the
# main "Extract" page. Reads everything from the local store/ directory.

import io
from pathlib import Path

import streamlit as st
import pandas as pd
import pymupdf

import store


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
    if st.button("🔄 Refresh list", use_container_width=True):
        st.rerun()

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
            use_container_width=True,
        )
    if st.button("🗑️ Delete this entry", use_container_width=True, type="secondary"):
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
                st.image(img_path.read_bytes(), use_container_width=True,
                         caption=entry.get("caption") or None)
                st.download_button(
                    f"⬇️ {entry['filename']}",
                    data=img_path.read_bytes(),
                    file_name=entry["filename"],
                    mime="image/png",
                    key=f"dl_img_{entry['filename']}",
                )

# ---- Tables ----------------------------------------------------------------

with tab_tables:
    if not manifest["tables"]:
        st.info("No tables saved for this PDF.")
    else:
        for entry in manifest["tables"]:
            tbl_path = store.table_path(selected_id, entry["filename"])
            if not tbl_path.exists():
                continue
            with st.expander(
                f"Page {entry['page']}  ·  Table {entry['idx']}  ·  "
                f"{entry['rows']}×{entry['cols']}  ·  `{entry['filename']}`",
                expanded=False,
            ):
                if entry.get("section_header"):
                    st.markdown(f"**Section header:** {entry['section_header']}")
                if entry.get("nearest_text"):
                    st.markdown(f"**Nearest text:** {entry['nearest_text']}")
                try:
                    df = pd.read_csv(tbl_path)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                st.download_button(
                    f"⬇️ {entry['filename']}",
                    data=tbl_path.read_bytes(),
                    file_name=entry["filename"],
                    mime="text/csv",
                    key=f"dl_tbl_{entry['filename']}",
                )

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
                    st.image(png, use_container_width=True)
                except Exception as e:
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
                                use_container_width=True,
                                caption=entry.get("caption") or entry["filename"],
                            )

                if items["tables"]:
                    st.markdown("**Tables**")
                    for entry in items["tables"]:
                        tp = store.table_path(selected_id, entry["filename"])
                        if not tp.exists():
                            continue
                        with st.expander(
                            f"Table {entry['idx']} · {entry['rows']}×{entry['cols']} · "
                            f"`{entry['filename']}`",
                            expanded=True,
                        ):
                            try:
                                df = pd.read_csv(tp)
                                st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not read CSV: {e}")

            st.divider()
