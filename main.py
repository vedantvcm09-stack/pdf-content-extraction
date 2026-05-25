# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------

import io
import re
import difflib
import argparse
import tempfile
from statistics import median
from pathlib import Path
from PIL import Image

import pandas as pd
import pymupdf
import pdfplumber
import tempfile
from pathlib import Path
from PIL import Image

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)

try:
    from docling_core.types.doc import DocItemLabel  # docling 2.x
except ImportError:
    from docling.datamodel.document import DocItemLabel  # older builds

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

DEFAULT_TOP_N = 3  # number of top matches to display
DEFAULT_THRESHOLD = 0.25  # warn if best score falls below this
DEFAULT_IMAGE_SCALE = 2.0  # Docling render scale for figures/page images
WINDOW_SIZE = 6  # rolling buffer size for preceding text items
_SOURCE_PDF_PATHS = {}

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "for", "on",
    "at", "by", "with", "from", "or", "be", "as", "it", "its", "this",
    "that", "has", "have", "been", "was", "were", "which", "between",
    "into", "not", "also", "up", "will", "further", "following", "about",
    "than", "their", "such", "other",
}

# ---------------------------------------------------------------------------
# 2. Parsing
# ---------------------------------------------------------------------------

def build_converter(
    use_ocr: bool = False,
    extract_images: bool = False,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    accurate: bool = False,
) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = use_ocr
    opts.do_table_structure = True
    if accurate:
        try:
            opts.table_structure_options = TableStructureOptions(
                mode=TableFormerMode.ACCURATE
            )
        except Exception:
            pass
    opts.generate_picture_images = extract_images
    if hasattr(opts, "images_scale"):
        opts.images_scale = image_scale
    elif hasattr(opts, "image_scale"):
        opts.image_scale = image_scale
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def parse_pdf(
    pdf_path: str,
    use_ocr: bool = False,
    extract_images: bool = False,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    accurate: bool = False,
):
    print(f"[docling] File : {Path(pdf_path).name}")
    print(f"[docling] OCR : {'ON (scanned mode)' if use_ocr else 'OFF (typed PDF mode)'}")
    print(f"[docling] Images   : {'ON' if extract_images else 'OFF'}")
    print(f"[docling] Accurate : {'ON (slow — ACCURATE mode)' if accurate else 'OFF (fast mode)'}")
    if extract_images:
        print(f"[docling] Image scale : {image_scale}")
    result = build_converter(use_ocr, extract_images, image_scale, accurate).convert(
        str(Path(pdf_path).resolve())
    )
    try:
        origin = result.document.origin
        if origin and origin.filename:
            _SOURCE_PDF_PATHS[origin.filename] = str(Path(pdf_path).resolve())
    except Exception:
        pass
    n_tables = len(result.document.tables)
    n_pics = len(result.document.pictures)
    print(f"[docling] Done — {n_tables} tables, {n_pics} pictures detected")
    return result

# ---------------------------------------------------------------------------
# 3. Shared: flatten all_items list
# ---------------------------------------------------------------------------

def build_all_items(conv_result) -> list:
    all_items = []
    for item, level in conv_result.document.iterate_items():
        page_no = 0
        try:
            if item.prov:
                page_no = item.prov[0].page_no
        except Exception:
            pass
        all_items.append((item.label, getattr(item, "text", "").strip(), page_no))
    return all_items

# ---------------------------------------------------------------------------
# 4. Table Context Extraction
# ---------------------------------------------------------------------------

def extract_cell_text(table_item) -> str:
    try:
        cells = []
        grid = table_item.data.grid
        for row in grid:
            for cell in row:
                val = str(getattr(cell, "text", "") or "").strip()
                if val:
                    cells.append(val)
        return " ".join(cells)
    except Exception:
        return ""

def extract_table_contexts(conv_result, all_items: list = None) -> list:
    doc = conv_result.document
    if all_items is None:
        all_items = build_all_items(conv_result)

    EXCLUDED = {
        DocItemLabel.TABLE,
        DocItemLabel.PICTURE,
        DocItemLabel.FORMULA,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.SECTION_HEADER,
    }

    contexts = []
    table_count = 0

    for pos, (lbl, _, _) in enumerate(all_items):
        if lbl != DocItemLabel.TABLE:
            continue

        anchor_page = all_items[pos][2]

        last_header = ""
        text_window = []
        for j in range(pos - 1, max(0, pos - WINDOW_SIZE * 3) - 1, -1):
            j_lbl, j_text, j_page = all_items[j]
            if j_page != anchor_page and j_page != 0 and anchor_page != 0:
                break
            if j_lbl == DocItemLabel.TABLE:
                break
            if j_lbl == DocItemLabel.SECTION_HEADER and j_text and not last_header:
                last_header = j_text
            elif j_lbl not in EXCLUDED and j_text:
                text_window.insert(0, j_text)
                if len(text_window) >= WINDOW_SIZE:
                    break

        caption_text = ""
        _IGNORE_PROV = {
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.FOOTNOTE,
        }
        for k in range(pos + 1, min(len(all_items), pos + 4)):
            k_lbl, k_text, k_page = all_items[k]
            if k_lbl not in _IGNORE_PROV:
                if k_page != anchor_page and k_page != 0 and anchor_page != 0:
                    break
            if k_lbl == DocItemLabel.CAPTION and k_text:
                caption_text = k_text
                break
            if k_lbl == DocItemLabel.TABLE:
                break

        nearest = caption_text if caption_text else (text_window[-1] if text_window else last_header)
        full_ctx = " ".join(text_window + ([caption_text] if caption_text else []))

        contexts.append(
            {
                "table_idx": table_count,
                "section_header": last_header,
                "nearest_text": nearest,
                "full_context": full_ctx,
                "cell_text": extract_cell_text(doc.tables[table_count]),
            }
        )
        table_count += 1

    return contexts

# ---------------------------------------------------------------------------
# 5. Figure Context Extraction
# ---------------------------------------------------------------------------

def extract_figure_contexts(conv_result, all_items: list = None) -> list:
    doc = conv_result.document
    if all_items is None:
        all_items = build_all_items(conv_result)

    EXCLUDED = {
        DocItemLabel.TABLE,
        DocItemLabel.PICTURE,
        DocItemLabel.FORMULA,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.SECTION_HEADER,
    }

    contexts = []
    pic_count = 0

    for pos, (lbl, _, _) in enumerate(all_items):
        if lbl != DocItemLabel.PICTURE:
            continue

        anchor_page = all_items[pos][2]

        last_header = ""
        text_window = []
        for j in range(pos - 1, max(0, pos - WINDOW_SIZE * 3) - 1, -1):
            j_lbl, j_text, j_page = all_items[j]
            if j_page != anchor_page and j_page != 0 and anchor_page != 0:
                break
            if j_lbl == DocItemLabel.PICTURE:
                break
            if j_lbl == DocItemLabel.SECTION_HEADER and j_text and not last_header:
                last_header = j_text
            elif j_lbl not in EXCLUDED and j_text:
                text_window.insert(0, j_text)
                if len(text_window) >= WINDOW_SIZE:
                    break

        caption_text = ""
        _IGNORE_PROV = {
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.FOOTNOTE,
        }
        for k in range(pos + 1, min(len(all_items), pos + 6)):
            k_lbl, k_text, k_page = all_items[k]
            if k_lbl not in _IGNORE_PROV:
                if k_page != anchor_page and k_page != 0 and anchor_page != 0:
                    break
            if k_lbl == DocItemLabel.CAPTION and k_text:
                caption_text = k_text
                break
            if k_lbl == DocItemLabel.PICTURE:
                break

        nearest = caption_text if caption_text else (text_window[-1] if text_window else last_header)
        full_ctx = " ".join(text_window + ([caption_text] if caption_text else []))

        contexts.append(
            {
                "pic_idx": pic_count,
                "section_header": last_header,
                "nearest_text": nearest,
                "full_context": full_ctx,
                "caption": caption_text,
            }
        )
        pic_count += 1

    return contexts

# ---------------------------------------------------------------------------
# 6. Matching & Scoring
# ---------------------------------------------------------------------------

def tokenize(text: str) -> set:
    text = "" if text is None else str(text)
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 1 and w not in STOPWORDS
    }

def soft_token_overlap(q_tokens: set, candidate_text: str) -> float:
    if not q_tokens:
        return 0.0
    c_tokens = tokenize(candidate_text)
    matched = 0.0
    for qt in q_tokens:
        if qt in c_tokens:
            matched += 1.0
        elif len(qt) >= 4 and any(ct.startswith(qt) for ct in c_tokens):
            matched += 0.80
    return matched / len(q_tokens)

def sliding_window(query: str, candidate: str) -> float:
    query = "" if query is None else str(query)
    candidate = "" if candidate is None else str(candidate)
    w_len = len(query) + 30
    if len(candidate) <= w_len:
        return difflib.SequenceMatcher(None, query, candidate).ratio()
    step = max(1, w_len // 4)
    return max(
        difflib.SequenceMatcher(None, query, candidate[i:i + w_len]).ratio()
        for i in range(0, len(candidate) - w_len + 1, step)
    )

def score_match(query: str, ctx: dict) -> float:
    q = query.lower().strip()
    if not q:
        return 0.0
    q_tokens = tokenize(q)

    nt_score = 0.0
    if ctx.get("nearest_text"):
        nt = ctx["nearest_text"].lower()
        if q in nt:
            return 1.0
        nt_score = max(
            soft_token_overlap(q_tokens, nt),
            difflib.SequenceMatcher(None, q, nt).ratio() * 0.80,
            sliding_window(q, nt) * 0.85,
        )

    sh_score = 0.0
    if ctx.get("section_header"):
        sh = ctx["section_header"].lower()
        sh_cap = 0.90 if not ctx.get("nearest_text", "").strip() else 0.50
        if q in sh:
            sh_score = sh_cap
        else:
            sh_score = max(
                soft_token_overlap(q_tokens, sh) * sh_cap,
                difflib.SequenceMatcher(None, q, sh).ratio() * (sh_cap * 0.80),
            )

    fallback_score = 0.0
    if max(nt_score, sh_score) < 0.35:
        cell = ctx.get("cell_text", "")
        if cell:
            fallback_score = max(
                fallback_score,
                soft_token_overlap(q_tokens, cell) * 0.65,
                sliding_window(q, cell) * 0.60,
            )

        full = ctx.get("full_context", "")
        if full:
            fallback_score = max(
                fallback_score,
                soft_token_overlap(q_tokens, full) * 0.65,
                sliding_window(q, full) * 0.60,
            )

    return round(max(nt_score, sh_score, fallback_score), 4)

def rank_tables(query: str, contexts: list) -> list:
    return sorted(
        [(score_match(query, ctx), ctx) for ctx in contexts],
        key=lambda x: x[0],
        reverse=True,
    )

def rank_figures(query: str, contexts: list) -> list:
    return sorted(
        [(score_match(query, ctx), ctx) for ctx in contexts],
        key=lambda x: x[0],
        reverse=True,
    )

# ---------------------------------------------------------------------------
# 7. CSV Export helpers
# ---------------------------------------------------------------------------

def collapse_to_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    def col_is_empty(series: pd.Series) -> bool:
        null_vals = {"", "nan", "None", "NaN"}
        return series.astype(str).str.strip().isin(null_vals).all()

    cols = list(df.columns)
    first_data = 0
    last_data = len(cols) - 1

    for i in range(len(cols)):
        if not col_is_empty(df.iloc[:, i]):
            first_data = i
            break

    for i in range(len(cols) - 1, -1, -1):
        if not col_is_empty(df.iloc[:, i]):
            last_data = i
            break

    return df.iloc[:, first_data:last_data + 1].copy()

def make_columns_unique(columns):

    seen = {}
    result = []

    for col in columns:

        col = str(col).strip()

        if col not in seen:
            seen[col] = 0
            result.append(col)

        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")

    return result
def _safe_cell_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)

def _stringify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    safe_df = df.copy()
    safe_df.columns = [_safe_cell_value(col).strip() or f"Col_{idx}" for idx, col in enumerate(safe_df.columns)]
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(lambda value: _safe_cell_value(value).strip())
    return safe_df

def _original_export_table_smart(table_item, doc) -> pd.DataFrame:
    def _safe_export_to_df(ti):
        for kwargs in [{'doc': doc}, {}]:
            try:
                result = ti.export_to_dataframe(**kwargs)
                if result is not None:
                    return result
            except (TypeError, Exception):
                continue
        return pd.DataFrame()

    try:
        grid = table_item.data.grid
    except AttributeError:
        return _safe_export_to_df(table_item)

    if not grid:
        return _safe_export_to_df(table_item)

    num_cols = max(
        (cell.start_col_offset_idx + cell.col_span)
        for row in grid for cell in row
        if hasattr(cell, "col_span")
    ) if grid else 0

    if num_cols == 0:
        return _safe_export_to_df(table_item)

    rowspan_continuations = set()
    for grid_row_idx, row in enumerate(grid):
        for cell in row:
            row_start = getattr(cell, "start_row_offset_idx", grid_row_idx)
            col_start = getattr(cell, "start_col_offset_idx", 0)
            row_span = getattr(cell, "row_span", 1)
            col_span = getattr(cell, "col_span", 1)
            for r in range(row_start + 1, row_start + row_span):
                for c in range(col_start, col_start + col_span):
                    rowspan_continuations.add((r, c))

    raw_rows = []
    for grid_row_idx, row in enumerate(grid):
        if not row:
            continue
        first_cell = row[0]
        is_full_span = (
            hasattr(first_cell, "col_span")
            and first_cell.col_span == num_cols
            and len(row) == 1
        )

        row_data = [""] * num_cols
        if is_full_span:
            row_data[0] = first_cell.text.strip() if first_cell.text else ""
        else:
            for cell in row:
                col = getattr(cell, "start_col_offset_idx", 0)
                if col < num_cols:
                    if (grid_row_idx, col) not in rowspan_continuations:
                        row_data[col] = cell.text.strip() if cell.text else ""
        raw_rows.append(row_data)

    if not raw_rows:
        return _safe_export_to_df(table_item)

    return pd.DataFrame(raw_rows[1:], columns=raw_rows[0])

def _cell_text(cell) -> str:
    return str(getattr(cell, "text", "") or "").strip()

def _cell_bbox(cell):
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
        try:
            if cell.prov:
                bbox = cell.prov[0].bbox
        except Exception:
            return None
    return bbox

def _bbox_coord(bbox, names: tuple[str, ...]):
    for name in names:
        if hasattr(bbox, name):
            return getattr(bbox, name)
    if isinstance(bbox, dict):
        for name in names:
            if name in bbox:
                return bbox[name]
    return None

def _cell_x_center(cell) -> float | None:
    bbox = _cell_bbox(cell)
    if bbox is None:
        return None
    left = _bbox_coord(bbox, ("l", "left", "x0"))
    right = _bbox_coord(bbox, ("r", "right", "x1"))
    if left is None or right is None:
        return None
    try:
        return (float(left) + float(right)) / 2.0
    except (TypeError, ValueError):
        return None

def _numeric_density(values: list[str]) -> float:
    tokens = []
    for value in values:
        tokens.extend(re.findall(r"\(?-?\d+(?:\.\d+)?%?\)?|[A-Za-z][A-Za-z/-]*", str(value)))
    if not tokens:
        return 0.0
    numeric = sum(1 for tok in tokens if re.search(r"\d", tok))
    return numeric / len(tokens)

def _row_profile(row: list[str]) -> dict:
    non_empty = [str(v).strip() for v in row if str(v).strip()]
    numeric_cells = sum(1 for value in non_empty if re.search(r"\d", value))
    return {
        "non_empty": len(non_empty),
        "numeric_cells": numeric_cells,
        "numeric_density": _numeric_density(non_empty),
        "text_density": 0.0 if not non_empty else 1.0 - (numeric_cells / len(non_empty)),
    }

def _looks_like_header_row(row: list[str]) -> bool:
    profile = _row_profile(row)
    if profile["non_empty"] == 0:
        return False
    return profile["numeric_density"] <= 0.35 and profile["text_density"] >= 0.50

def _looks_like_compact_header_band(row: list[str]) -> bool:
    values = [_clean_header_piece(value) for value in row if str(value).strip()]
    if len(values) < 2:
        return False
    avg_len = sum(len(value) for value in values) / len(values)
    long_cell_ratio = sum(1 for value in values if len(value.split()) > 4 or len(value) > 35) / len(values)
    return avg_len <= 24 and long_cell_ratio <= 0.25

def _looks_like_data_row_label(value: str) -> bool:
    """Detect data-row labels using purely structural patterns (no domain keywords)."""
    value = str(value or "").strip()
    if not value:
        return False
    # Structural patterns: numbered items, lettered items, roman numerals
    return bool(
        re.match(r"^\(?\d+[\).\s-]", value)       # "1.", "1)", "(1)", "1 -"
        or re.match(r"^[A-Za-z]\)", value)          # "a)", "A)"
        or re.match(r"^(?:i{1,3}|iv|vi{0,3}|ix|x)\)", value.lower())  # roman: "i)", "ii)", "iii)"
    )

def _row_overlap_ratio(a: list[str], b: list[str]) -> float:
    a_values = {_clean_header_piece(v).lower() for v in a if str(v).strip()}
    b_values = {_clean_header_piece(v).lower() for v in b if str(v).strip()}
    if not a_values or not b_values:
        return 0.0
    return len(a_values & b_values) / max(len(b_values), 1)

def _classify_row(row: list[str]) -> str:
    profile = _row_profile(row)
    if profile["non_empty"] == 0:
        return "empty"
    values = [_clean_header_piece(v) for v in row if str(v).strip()]
    unique_values = {v.lower() for v in values if v}
    if profile["non_empty"] <= 2 and profile["numeric_cells"] == 0:
        return "section_header"
    if profile["numeric_cells"] == 0 and profile["non_empty"] >= 3 and len(unique_values) <= 2:
        return "section_header"
    return "data"

def _detect_header_row_count(raw_rows: list[list[str]]) -> int:
    """Conservative header detection: default to 1 row.
    Only extend to 2+ rows if row is CLEARLY a sub-header band
    (all text, no numeric data, short labels, high overlap with row 0).
    NEVER absorb a row that looks like data."""
    if not raw_rows or len(raw_rows) <= 1:
        return 1

    max_header_rows = min(3, max(1, len(raw_rows) - 1))
    count = 1

    for idx in range(1, max_header_rows):
        row = raw_rows[idx]
        if not row:
            break

        # Hard stop: if this row starts with a data label pattern ("1.", "a)", etc.)
        first_cell = str(row[0]).strip() if row else ""
        if _looks_like_data_row_label(first_cell):
            break

        # Hard stop: if this row is a section header
        if _classify_row(row) == "section_header":
            break

        profile = _row_profile(row)

        compact_header_like = _looks_like_compact_header_band(row)

        # Hard stop for numeric data rows.  Multi-row headers may contain
        # years or units, so low-density numeric text in compact labels can
        # still be header material.
        if profile["numeric_cells"] > 0 and not (
            compact_header_like and profile["numeric_density"] <= 0.35
        ):
            break

        # Hard stop: if the row has substantial populated cells that don't overlap
        # with row 0, it's a data row, not a continuation header
        populated = sum(1 for value in row if str(value).strip())
        if populated >= max(2, len(row) // 2):
            overlap = _row_overlap_ratio(raw_rows[0], row)
            if overlap < 0.20 and not _looks_like_compact_header_band(row):
                break

        # Handle parenthesized units like "(in Km)" or "(Rs. in Cr.)"
        is_parenthesized_units = False
        non_empty_cells = [str(v).strip() for v in row if str(v).strip()]
        if non_empty_cells:
            paren_count = sum(1 for v in non_empty_cells if re.match(r"^\(.*\)$", v))
            if paren_count / len(non_empty_cells) >= 0.5:
                is_parenthesized_units = True

        row0_values = [_clean_header_piece(v) for v in raw_rows[0] if str(v).strip()]
        row0_fills_all = len(row0_values) == len(raw_rows[0])
        row0_unique_ratio = len({v.lower() for v in row0_values if v}) / max(len(row0_values), 1)
        unit_or_date_markers = sum(
            1
            for value in non_empty_cells
            if re.search(r"\(|\)|\b\d{4}\b|%", value)
        )
        row_is_unit_or_date_band = (
            non_empty_cells
            and unit_or_date_markers / len(non_empty_cells) >= 0.50
        )
        if (
            idx == 1
            and row0_fills_all
            and row0_unique_ratio >= 0.80
            and populated == len(row)
            and _row_overlap_ratio(raw_rows[0], row) < 0.20
            and not is_parenthesized_units
            and not row_is_unit_or_date_band
        ):
            break

        # Only accept as header if it's clearly a compact header band or units
        if compact_header_like or is_parenthesized_units:
            count = idx + 1
        else:
            break

    return max(1, count)

def _clean_header_piece(value: str) -> str:
    """Normalize header text: collapse whitespace, strip trailing punctuation."""
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = value.strip("-:;,. ")
    return value

def _decorate_duplicate_header_children(headers: list[list[str]]) -> list[list[str]]:
    """Disambiguate duplicate child-header cells by prefixing with their
    corresponding parent-header text.  Works for any parent labels"""
    if len(headers) < 2:
        return headers

    decorated = [row[:] for row in headers]
    child_row = decorated[-1]
    parent_row = decorated[-2]
    num_cols = len(child_row)
    idx = 0
    while idx < num_cols:
        child = _clean_header_piece(child_row[idx])
        if not child:
            idx += 1
            continue
        # Find run of consecutive identical child values
        end = idx + 1
        while end < num_cols and _clean_header_piece(child_row[end]).lower() == child.lower():
            end += 1
        run_len = end - idx
        if run_len >= 2:
            # Collect the parent text for each position in the run
            parent_texts = []
            for k in range(idx, end):
                pt = _clean_header_piece(parent_row[k]) if k < len(parent_row) else ""
                parent_texts.append(pt)
            # Only decorate if parents are distinct (otherwise prefixing
            # would not help disambiguate)
            distinct_parents = {pt.lower() for pt in parent_texts if pt}
            if len(distinct_parents) >= 2:
                for k in range(idx, end):
                    pt = parent_texts[k - idx]
                    if pt:
                        decorated[-1][k] = f"{pt} {child}"
        idx = end
    return decorated

def _fill_header_blanks(headers: list[list[str]]) -> list[list[str]]:
    filled = [row[:] for row in headers]

    for row_idx in range(len(filled) - 1):
        populated = [
            idx for idx, value in enumerate(filled[row_idx])
            if _clean_header_piece(value)
        ]

        carry = ""

        for col_idx, value in enumerate(filled[row_idx]):
            value = _clean_header_piece(value)

            if value:
                carry = value

            elif carry and col_idx > 0:
                filled[row_idx][col_idx] = carry

    return filled

def _merge_header_rows(header_rows: list[list[str]], num_cols: int) -> list[str]:

    header_rows = _decorate_duplicate_header_children(header_rows)
    header_rows = _fill_header_blanks(header_rows)

    columns = []

    for col_idx in range(num_cols):

        pieces = []

        for row in header_rows:

            piece = _clean_header_piece(
                row[col_idx] if col_idx < len(row) else ""
            )

            if piece and (
                not pieces or piece.lower() != pieces[-1].lower()
            ):
                pieces.append(piece)

        if pieces:
            columns.append(" - ".join(pieces))
        else:
            columns.append(f"Column_{col_idx}")

    return columns

def _looks_like_key_value_without_header(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    num_cols = max((len(row) for row in rows), default=0)
    if num_cols != 2:
        return False
    populated_rows = [
        row for row in rows
        if sum(1 for value in row if str(value).strip()) == 2
    ]
    if len(populated_rows) < 2:
        return False
    first_left = str(populated_rows[0][0]).strip()
    first_right = str(populated_rows[0][1]).strip()
    if not first_left or not first_right:
        return False
    first_row_header_like = (
        re.search(r"[A-Za-z]", first_left)
        and re.search(r"[A-Za-z]", first_right)
        and _numeric_density([first_left, first_right]) <= 0.25
        and not _looks_like_data_row_label(first_left)
    )
    right_values = [str(row[1]).strip() for row in populated_rows]
    right_data_ratio = sum(
        1 for value in right_values
        if _numeric_like(value) or _numeric_density([value]) >= 0.35
    ) / len(right_values)
    left_text_ratio = sum(
        1 for row in populated_rows
        if re.search(r"[A-Za-z]", str(row[0]))
    ) / len(populated_rows)
    if first_row_header_like and right_data_ratio >= 0.50:
        return False
    return right_data_ratio >= 0.80 and left_text_ratio >= 0.80

def _column_centers_from_grid(grid: list) -> dict[int, float]:
    buckets = {}
    for row in grid:
        for cell in row:
            col = getattr(cell, "start_col_offset_idx", None)
            center = _cell_x_center(cell)
            if col is None or center is None:
                continue
            buckets.setdefault(int(col), []).append(center)
    return {col: median(values) for col, values in buckets.items() if values}

def _nearest_col_from_center(center: float | None, column_centers: dict[int, float]) -> int | None:
    if center is None or not column_centers:
        return None
    return min(column_centers, key=lambda col: abs(column_centers[col] - center))

def _safe_col_for_cell(cell, fallback_col: int, column_centers: dict[int, float], num_cols: int) -> int:
    col = getattr(cell, "start_col_offset_idx", None)
    if col is None:
        col = _nearest_col_from_center(_cell_x_center(cell), column_centers)
    if col is None:
        col = fallback_col
    col = int(max(0, min(int(col), num_cols - 1)))
    return col

def _build_offset_enhanced_raw_rows(table_item) -> list[list[str]]:
    grid = table_item.data.grid
    if not grid:
        return []

    offset_cols = [
        getattr(cell, "start_col_offset_idx", idx) + getattr(cell, "col_span", 1)
        for row in grid
        for idx, cell in enumerate(row)
    ]
    max_cells_in_row = max((len(row) for row in grid), default=0)
    num_cols = max(max(offset_cols, default=0), max_cells_in_row)
    if num_cols == 0:
        return []

    column_centers = _column_centers_from_grid(grid)
    rowspan_continuations = set()
    for grid_row_idx, row in enumerate(grid):
        for cell_idx, cell in enumerate(row):
            row_start = getattr(cell, "start_row_offset_idx", grid_row_idx)
            col_start = _safe_col_for_cell(cell, cell_idx, column_centers, num_cols)
            row_span = getattr(cell, "row_span", 1)
            col_span = getattr(cell, "col_span", 1)
            for r in range(row_start + 1, row_start + row_span):
                for c in range(col_start, min(num_cols, col_start + col_span)):
                    rowspan_continuations.add((r, c))

    raw_rows = []
    for grid_row_idx, row in enumerate(grid):
        row_data = [""] * num_cols
        occupied = set()
        for cell_idx, cell in enumerate(row):
            text = _cell_text(cell)
            if not text:
                continue

            col_span = max(1, int(getattr(cell, "col_span", 1)))
            col = _safe_col_for_cell(cell, cell_idx, column_centers, num_cols)
            if col_span >= num_cols:
                col = 0
                if row_data[col].strip() == text:
                    continue
            if col in occupied and _cell_x_center(cell) is not None:
                empties = [idx for idx in range(num_cols) if idx not in occupied]
                if empties:
                    col = min(
                        empties,
                        key=lambda idx: abs(column_centers.get(idx, idx) - (_cell_x_center(cell) or idx)),
                    )

            if (grid_row_idx, col) in rowspan_continuations:
                continue

            repeat_span = grid_row_idx < 3 and 1 < col_span < num_cols
            target_cols = range(col, min(num_cols, col + col_span)) if repeat_span else (col,)

            for target_col in target_cols:
                existing = row_data[target_col].strip()
                if existing == text:
                    continue
                if existing and text in existing.split("  "):
                    continue
                if row_data[target_col]:
                    row_data[target_col] = f"{row_data[target_col]} {text}".strip()
                else:
                    row_data[target_col] = text
                occupied.add(target_col)

        if any(value.strip() for value in row_data):
            raw_rows.append(row_data)
    return raw_rows

def _cell_axis_bounds(cell, axis: str = "x") -> tuple[float, float] | None:
    bbox = _cell_bbox(cell)
    if bbox is None:
        return None
    if axis == "y":
        start = _bbox_coord(bbox, ("t", "top", "y0"))
        end = _bbox_coord(bbox, ("b", "bottom", "y1"))
    else:
        start = _bbox_coord(bbox, ("l", "left", "x0"))
        end = _bbox_coord(bbox, ("r", "right", "x1"))
    if start is None or end is None:
        return None
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        return None
    return (min(start, end), max(start, end))

def _cell_dimensions(cell) -> tuple[float, float] | None:
    x_bounds = _cell_axis_bounds(cell, "x")
    y_bounds = _cell_axis_bounds(cell, "y")
    if x_bounds is None or y_bounds is None:
        return None
    return (x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0])

def _infer_column_axis(grid: list) -> str:
    wide_weight = 0
    tall_weight = 0
    for row in grid:
        for cell in row:
            text = _cell_text(cell)
            if not text:
                continue
            dims = _cell_dimensions(cell)
            if dims is None:
                continue
            width, height = dims
            weight = max(len(text), 1)
            if height > width * 1.20:
                tall_weight += weight
            elif width > height * 1.20:
                wide_weight += weight
    return "y" if tall_weight > wide_weight * 1.25 else "x"

def _axis_center(cell, axis: str) -> float | None:
    bounds = _cell_axis_bounds(cell, axis)
    if bounds is None:
        return None
    return (bounds[0] + bounds[1]) / 2.0

def _dedupe_row_cells(row: list, axis: str) -> list:
    seen = set()
    cells = []
    for cell in row:
        text = _cell_text(cell)
        if not text:
            continue
        x_bounds = _cell_axis_bounds(cell, "x")
        y_bounds = _cell_axis_bounds(cell, "y")
        key = (
            text,
            getattr(cell, "start_col_offset_idx", None),
            getattr(cell, "col_span", 1),
            tuple(round(v, 2) for v in x_bounds) if x_bounds else None,
            tuple(round(v, 2) for v in y_bounds) if y_bounds else None,
        )
        if key in seen:
            continue
        seen.add(key)
        cells.append(cell)
    return sorted(cells, key=lambda c: (_axis_center(c, axis) is None, _axis_center(c, axis) or 0.0))

def _cluster_positions(positions: list[float]) -> list[float]:
    if not positions:
        return []
    values = sorted(float(p) for p in positions)
    if len(values) == 1:
        return values

    gaps = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not gaps:
        return [median(values)]

    positive_gaps = sorted(gaps)
    small_gaps = positive_gaps[:max(1, len(positive_gaps) // 2)]
    threshold = max(2.0, median(small_gaps) * 2.5)
    large_gap_floor = median(positive_gaps) * 0.75
    if large_gap_floor > threshold * 3:
        threshold = min(large_gap_floor, threshold * 4)

    clusters = [[values[0]]]
    for value in values[1:]:
        if value - clusters[-1][-1] <= threshold:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [median(cluster) for cluster in clusters]

def _canonical_columns_from_geometry(grid: list, axis: str) -> list[float]:
    row_positions = []
    for row in grid:
        cells = _dedupe_row_cells(row, axis)
        atomic_positions = []
        for cell in cells:
            col_span = max(1, int(getattr(cell, "col_span", 1)))
            center = _axis_center(cell, axis)
            if center is None:
                continue
            if col_span == 1:
                atomic_positions.append(center)
        if atomic_positions:
            row_positions.append(sorted(atomic_positions))

    if not row_positions:
        all_positions = [
            _axis_center(cell, axis)
            for row in grid
            for cell in _dedupe_row_cells(row, axis)
            if _axis_center(cell, axis) is not None
        ]
        return _cluster_positions([p for p in all_positions if p is not None])

    target_count = max(len(row) for row in row_positions)
    strong_rows = [row for row in row_positions if len(row) == target_count]
    if len(strong_rows) >= 2:
        return [median(row[idx] for row in strong_rows) for idx in range(target_count)]

    clustered = _cluster_positions([pos for row in row_positions for pos in row])
    if len(clustered) >= target_count:
        return clustered

    best_row = max(row_positions, key=len)
    return best_row

def _nearest_column_index(center: float | None, columns: list[float]) -> int | None:
    if center is None or not columns:
        return None
    return min(range(len(columns)), key=lambda idx: abs(columns[idx] - center))

def _columns_covered_by_cell(cell, axis: str, columns: list[float]) -> list[int]:
    if not columns:
        return []

    col_span = max(1, int(getattr(cell, "col_span", 1)))
    bounds = _cell_axis_bounds(cell, axis)
    center = _axis_center(cell, axis)
    nearest = _nearest_column_index(center, columns)
    if nearest is None:
        return []

    if col_span <= 1:
        return [nearest]

    covered = []
    if bounds is not None:
        gaps = [abs(b - a) for a, b in zip(columns, columns[1:])]
        pad = (median(gaps) * 0.35) if gaps else 2.0
        covered = [idx for idx, pos in enumerate(columns) if bounds[0] - pad <= pos <= bounds[1] + pad]

    if 1 < len(covered) <= col_span + 1:
        return covered

    offset = getattr(cell, "start_col_offset_idx", None)
    if offset is not None and 0 <= int(offset) < len(columns):
        return list(range(int(offset), min(len(columns), int(offset) + col_span)))

    start = max(0, min(nearest, len(columns) - col_span))
    return list(range(start, min(len(columns), start + col_span)))

def _build_geometric_raw_rows(table_item) -> list[list[str]]:
    grid = table_item.data.grid
    if not grid:
        return []

    axis = _infer_column_axis(grid)
    columns = _canonical_columns_from_geometry(grid, axis)
    if not columns:
        return []

    raw_rows = []
    for row in grid:
        row_data = [""] * len(columns)
        for cell in _dedupe_row_cells(row, axis):
            text = _cell_text(cell)
            if not text:
                continue
            covered_cols = _columns_covered_by_cell(cell, axis, columns)
            if not covered_cols:
                continue
            col_span = max(1, int(getattr(cell, "col_span", 1)))
            # Full-width spanning cells (subheaders/titles) go to col 0 only
            if col_span >= len(columns):
                covered_cols = [0]
                # Place text only in column 0 — do NOT repeat across columns
                existing = row_data[0].strip()
                if existing != text:
                    row_data[0] = f"{existing} {text}".strip() if existing else text
                continue

            repeat_span = len(covered_cols) > 1 and col_span > 1
            target_cols = covered_cols if repeat_span else [covered_cols[0]]
            for col in target_cols:
                existing = row_data[col].strip()
                if existing == text:
                    continue
                row_data[col] = f"{existing} {text}".strip() if existing else text
        if any(value.strip() for value in row_data):
            raw_rows.append(row_data)
    return raw_rows

def _source_pdf_path_for_doc(doc) -> str | None:
    try:
        origin = doc.origin
        filename = getattr(origin, "filename", "")
    except Exception:
        filename = ""
    if filename and filename in _SOURCE_PDF_PATHS:
        return _SOURCE_PDF_PATHS[filename]
    if filename:
        candidates = [Path(filename), Path.cwd() / filename]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
    return None

def _table_cell_union_bbox(table_item) -> tuple[float, float, float, float] | None:
    boxes = []
    try:
        grid = table_item.data.grid
    except Exception:
        grid = []
    for row in grid:
        for cell in row:
            bbox = _cell_bbox(cell)
            if bbox is None:
                continue
            left = _bbox_coord(bbox, ("l", "left", "x0"))
            top = _bbox_coord(bbox, ("t", "top", "y0"))
            right = _bbox_coord(bbox, ("r", "right", "x1"))
            bottom = _bbox_coord(bbox, ("b", "bottom", "y1"))
            if None in (left, top, right, bottom):
                continue
            boxes.append((float(left), float(top), float(right), float(bottom)))
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )

def _extract_table_text_lines(table_item, doc) -> list[dict]:
    pdf_path = _source_pdf_path_for_doc(doc)
    if not pdf_path:
        return []
    try:
        page_no = table_item.prov[0].page_no if table_item.prov else None
    except Exception:
        page_no = None
    if not page_no:
        return []

    table_bbox = _table_cell_union_bbox(table_item)
    if table_bbox is None:
        return []

    margin = 6.0
    x0, y0, x1, y1 = table_bbox
    clip = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)

    lines = []
    pdf_doc = None
    try:
        pdf_doc = pymupdf.open(pdf_path)
        page = pdf_doc[int(page_no) - 1]
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                bbox = line.get("bbox")
                if not bbox:
                    continue
                lx0, ly0, lx1, ly1 = (float(v) for v in bbox)
                lyc = (ly0 + ly1) / 2.0
                if lx1 < clip[0] or lx0 > clip[2] or lyc < clip[1] or lyc > clip[3]:
                    continue
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                lines.append(
                    {
                        "text": re.sub(r"\s+", " ", text),
                        "x0": lx0,
                        "y0": ly0,
                        "x1": lx1,
                        "y1": ly1,
                        "xc": (lx0 + lx1) / 2.0,
                        "yc": lyc,
                        "height": max(ly1 - ly0, 1.0),
                    }
                )
    except Exception:
        return []
    finally:
        if pdf_doc is not None:
            pdf_doc.close()
    return sorted(lines, key=lambda item: (item["yc"], item["x0"]))

def _cluster_text_lines_into_visual_rows(lines: list[dict]) -> list[list[dict]]:
    if not lines:
        return []
    heights = [line["height"] for line in lines if line.get("height")]
    threshold = max(2.0, median(heights) * 0.55) if heights else 3.0
    rows = []
    for line in sorted(lines, key=lambda item: item["yc"]):
        if not rows:
            rows.append([line])
            continue
        row_center = median(item["yc"] for item in rows[-1])
        if abs(line["yc"] - row_center) <= threshold:
            rows[-1].append(line)
        else:
            rows.append([line])

    # --- Post-clustering validation ---
    # Split visual rows where multiple text fragments start at very similar
    # X-positions (column-0 overlap).  This prevents multiline labels from

    all_x0 = [line["x0"] for line in lines]
    global_min_x = min(all_x0) if all_x0 else 0.0
    global_max_x = max(all_x0) if all_x0 else 1.0
    col0_zone = max(10.0, (global_max_x - global_min_x) * 0.12)

    validated = []
    for row in rows:
        if len(row) <= 1:
            validated.append(row)
            continue
        row.sort(key=lambda item: item["x0"])
        # Find items in the column-0 zone
        col0_items = [item for item in row if item["x0"] - global_min_x < col0_zone]
        if len(col0_items) >= 2:
            # Multiple items map to column 0 — check if they have distinct Y
            col0_items.sort(key=lambda item: item["yc"])
            y_gap = col0_items[-1]["yc"] - col0_items[0]["yc"]
            med_h = median(item["height"] for item in col0_items)
            if y_gap > med_h * 0.6:
                # Split: assign other items to their nearest col0 anchor by Y
                other_items = [item for item in row if item not in col0_items]
                for anchor in col0_items:
                    sub = [anchor]
                    for other in other_items:
                        if abs(other["yc"] - anchor["yc"]) <= threshold:
                            sub.append(other)
                    sub.sort(key=lambda item: item["x0"])
                    if sub:
                        validated.append(sub)
                # Handle any unassigned others
                assigned = {id(item) for sub in validated[-len(col0_items):] for item in sub}
                leftover = [item for item in other_items if id(item) not in assigned]
                if leftover and validated:
                    validated[-1].extend(leftover)
                    validated[-1].sort(key=lambda item: item["x0"])
                continue
        validated.append(row)

    for row in validated:
        row.sort(key=lambda item: item["x0"])
    return validated

def _line_columns(lines: list[dict]) -> list[float]:
    starts = [line["x0"] for line in lines]
    columns = _cluster_positions(starts)
    if len(columns) <= 1:
        columns = _cluster_positions([line["xc"] for line in lines])
    return columns

def _column_starts_from_grid(table_item) -> list[float]:
    try:
        grid = table_item.data.grid
    except Exception:
        return []
    buckets = {}
    for row in grid:
        for cell in row:
            text = _cell_text(cell)
            if not text:
                continue
            col_span = max(1, int(getattr(cell, "col_span", 1)))
            if col_span != 1:
                continue
            col = getattr(cell, "start_col_offset_idx", None)
            bounds = _cell_axis_bounds(cell, "x")
            if col is None or bounds is None:
                continue
            buckets.setdefault(int(col), []).append(bounds[0])
    if len(buckets) >= 2:
        return [median(buckets[col]) for col in sorted(buckets)]
    return []

def _line_to_column(line: dict, columns: list[float]) -> int | None:
    if not columns:
        return None
    if len(columns) == 1:
        return 0

    gaps = [b - a for a, b in zip(columns, columns[1:]) if b > a]
    default_gap = median(gaps) if gaps else 20.0
    boundaries = [columns[0] - default_gap / 2.0]
    boundaries.extend((a + b) / 2.0 for a, b in zip(columns, columns[1:]))
    boundaries.append(columns[-1] + default_gap / 2.0)

    x0 = float(line.get("x0", line.get("xc", 0.0)))
    x1 = float(line.get("x1", x0))

    # Prefer the starting edge for left-aligned text so long labels do not bleed into later numeric columns just because they visually overlap them.
    for idx in range(len(columns)):
        if boundaries[idx] <= x0 <= boundaries[idx + 1]:
            return idx

    overlaps = []
    for idx in range(len(columns)):
        overlap = max(0.0, min(x1, boundaries[idx + 1]) - max(x0, boundaries[idx]))
        overlaps.append(overlap)
    if max(overlaps, default=0.0) > 0:
        return max(range(len(columns)), key=lambda idx: overlaps[idx])

    return min(range(len(columns)), key=lambda idx: abs(columns[idx] - x0))

def _append_cell_text(row: list[str], col: int, text: str):
    text = text.strip()
    if not text:
        return
    existing = row[col].strip()
    if not existing:
        row[col] = text
    elif text not in existing:
        joiner = " " if existing.endswith(("-", "/", ",")) else " "
        row[col] = f"{existing}{joiner}{text}".strip()

def _is_connector_fragment(text: str) -> bool:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if not text or re.search(r"\d", text):
        return False
    words = re.findall(r"[A-Za-z]+", text)
    if not words or len(words) > 3:
        return False
    lowered = " ".join(word.lower() for word in words)
    connectors = {
        "of", "and", "or", "for", "to", "in", "on", "with", "by", "from",
        "incl", "including", "misc", "misc on", "expenditure on",
    }
    return lowered in connectors or all(len(word) <= 4 and word.lower() in connectors for word in words)

def _visual_rows_to_grid(visual_rows: list[list[dict]], columns: list[float]) -> list[list[str]]:
    raw_rows = []
    for visual_idx, visual in enumerate(visual_rows):
        row = [""] * len(columns)
        for line in visual:
            col = _line_to_column(line, columns)
            if col is not None:
                if col > 0 and _is_connector_fragment(line["text"]):
                    previous_populated = [idx for idx in range(col - 1, -1, -1) if row[idx].strip()]
                    if previous_populated:
                        col = previous_populated[0]
                _append_cell_text(row, col, line["text"])
        if not any(cell.strip() for cell in row):
            continue

        populated = [idx for idx, value in enumerate(row) if value.strip()]
        first_col_text = row[0].strip() if row else ""
        populated_values = [row[idx] for idx in populated]
        numeric_density = _numeric_density(populated_values)
        is_sparse_continuation = len(populated) <= max(2, len(columns) // 3)
        has_first_col_only = populated and populated[0] == 0 and len(populated) <= 2
        has_real_row_payload = len(populated) >= 3 or numeric_density >= 0.45

        # --- Strengthened continuation-merge heuristic ---
        # A row is a continuation of the previous row if:
        # 1. Sparse (few populated cells) AND no/low numeric content, OR
        # 2. Only column-0 text (label continuation)
        # Also check vertical gap: tight gap = stronger continuation signal.
        is_vertically_tight = False
        if raw_rows and visual_idx > 0:
            prev_visual = visual_rows[visual_idx - 1]
            gap = min(l["y0"] for l in visual) - max(l["y1"] for l in prev_visual)
            med_height = median(l["height"] for l in visual) if visual else 8.0
            is_vertically_tight = gap < med_height * 1.8

        prev_populated = []
        if raw_rows:
            prev_populated = [idx for idx, value in enumerate(raw_rows[-1]) if str(value).strip()]
        populated_subset_prev = bool(populated) and set(populated).issubset(set(prev_populated))
        has_alpha_fragment = any(re.search(r"[A-Za-z]", value) for value in populated_values)
        has_dangling_neighbor = any(
            idx < len(raw_rows[-1])
            and (
                re.search(r"[\(\[,;:/-]\s*$", str(raw_rows[-1][idx]).strip())
                or re.search(r"^[\)\],;:/-]", str(row[idx]).strip())
            )
            for idx in populated
        ) if raw_rows else False
        short_fragment_row = (
            populated_values
            and max(len(str(value).split()) for value in populated_values) <= 3
            and len(populated) <= max(2, int(len(columns) * 0.40))
        )
        no_first_col_wrap = (
            not first_col_text
            and is_vertically_tight
            and populated_subset_prev
            and (
                has_alpha_fragment
                or has_dangling_neighbor
                or (short_fragment_row and numeric_density < 0.80)
            )
        )

        # More aggressive continuation for sparse non-numeric rows with tight gap
        is_sparse_non_numeric = (
            len(populated) <= max(2, int(len(columns) * 0.40))
            and numeric_density < 0.20
            and not has_real_row_payload
        )

        should_merge_up = (
            raw_rows
            and (
                has_first_col_only
                or no_first_col_wrap
                or (not first_col_text and is_sparse_continuation and not has_real_row_payload)
                or (is_sparse_non_numeric and is_vertically_tight)
            )
        )

        if should_merge_up:
            for idx, value in enumerate(row):
                if value.strip():
                    _append_cell_text(raw_rows[-1], idx, value)
        else:
            raw_rows.append(row)
    return raw_rows

def _build_row_first_raw_rows(table_item, doc) -> list[list[str]]:
    lines = _extract_table_text_lines(table_item, doc)
    if not lines:
        return []
    columns = _column_starts_from_grid(table_item) or _line_columns(lines)
    if len(columns) < 2:
        return []
    visual_rows = _cluster_text_lines_into_visual_rows(lines)
    raw_rows = _visual_rows_to_grid(visual_rows, columns)
    return raw_rows

def _raw_rows_score(raw_rows: list[list[str]]) -> float:
    if not raw_rows:
        return 0.0
    width = max(len(row) for row in raw_rows)
    if width == 0:
        return 0.0
    normalized = [row + [""] * (width - len(row)) for row in raw_rows]
    non_empty_counts = [sum(1 for value in row if str(value).strip()) for row in normalized]
    if not non_empty_counts:
        return 0.0
    populated_ratio = median(non_empty_counts) / width
    ragged_penalty = (max(non_empty_counts) - min(non_empty_counts)) / max(width, 1)
    merged_penalty = 0.0
    for row in normalized[1:]:
        long_cells = sum(1 for value in row if len(str(value).split()) >= 8)
        merged_penalty += long_cells / width
    merged_penalty = merged_penalty / max(len(normalized) - 1, 1)
    return populated_ratio - (ragged_penalty * 0.25) - (merged_penalty * 0.35)

def _first_column_fragment_penalty(df: pd.DataFrame) -> float:
    if df is None or df.empty or df.shape[1] < 3 or len(df) < 2:
        return 0.0

    penalties = 0
    comparisons = max(len(df) - 1, 1)
    for row_idx in range(len(df) - 1):
        current_label = _safe_cell_value(df.iloc[row_idx, 0]).strip()
        next_label = _safe_cell_value(df.iloc[row_idx + 1, 0]).strip()
        if not current_label or not next_label:
            continue

        current_payload = [
            _safe_cell_value(value).strip()
            for value in df.iloc[row_idx, 1:].tolist()
            if _safe_cell_value(value).strip()
        ]
        next_payload = [
            _safe_cell_value(value).strip()
            for value in df.iloc[row_idx + 1, 1:].tolist()
            if _safe_cell_value(value).strip()
        ]
        if _numeric_density(current_payload) < 0.45 or _numeric_density(next_payload) < 0.45:
            continue

        current_words = len(re.findall(r"[A-Za-z0-9]+", current_label))
        next_words = len(re.findall(r"[A-Za-z0-9]+", next_label))
        next_starts_like_fragment = bool(re.match(r"^[a-z)\],;:]", next_label))
        current_unbalanced = current_label.count("(") > current_label.count(")")
        if current_words >= 10 and next_words <= 5 and (next_starts_like_fragment or current_unbalanced):
            penalties += 1

    return penalties / comparisons

def _looks_like_wrapped_text_fragment(value: str) -> bool:
    value = _safe_cell_value(value).strip()
    if not value:
        return False
    if re.match(r"^[a-z(\[;,]", value):
        return True
    words = re.findall(r"[A-Za-z]+", value)
    if not words:
        return False
    connector_endings = {
        "of", "to", "at", "in", "on", "for", "with", "due", "and", "or",
        "as", "by", "from", "than",
    }
    return (
        len(words) <= 4
        and words[-1].lower() in connector_endings
        and not _looks_like_data_row_label(value)
    )

def _short_wrapped_cell_fragment(value: str) -> bool:
    value = _safe_cell_value(value).strip()
    if not value:
        return False
    words = re.findall(r"[A-Za-z]+", value)
    if len(words) > 5:
        return False
    return bool(
        _looks_like_wrapped_text_fragment(value)
        or re.match(r"^\d+[\).]\s*", value)
        or re.search(r"[,;:/-]\s*$", value)
    )

def _wrapped_single_record_penalty(df: pd.DataFrame) -> float:
    if df is None or df.empty or df.shape[1] < 3:
        return 0.0
    if len(df) < 3 or len(df) > 12 or df.shape[1] > 8:
        return 0.0

    values = [
        _safe_cell_value(value).strip()
        for value in df.values.flatten().tolist()
        if _safe_cell_value(value).strip()
    ]
    if not values or _numeric_density(values) > 0.60:
        return 0.0

    row_density = _row_density(df)
    if row_density < 0.45:
        return 0.0

    first_col_values = [
        _safe_cell_value(value).strip()
        for value in df.iloc[1:, 0].tolist()
        if _safe_cell_value(value).strip()
    ]
    if len(first_col_values) < 2:
        return 0.0

    continuation_ratio = sum(
        1 for value in first_col_values
        if _looks_like_wrapped_text_fragment(value)
    ) / len(first_col_values)
    if continuation_ratio >= 0.65:
        return continuation_ratio

    short_fragment_ratio = sum(
        1 for value in values
        if _short_wrapped_cell_fragment(value)
    ) / len(values)
    return short_fragment_ratio if short_fragment_ratio >= 0.68 else 0.0

def _dataframe_quality_score(df: pd.DataFrame, raw_rows: list[list[str]] | None = None) -> float:
    if df is None or df.empty:
        return -999.0
    rows, cols = df.shape
    if cols < 2:
        return -50.0

    total_cells = max(rows * cols, 1)
    empty_cells = int(df.astype(str).apply(lambda col: col.str.strip().eq("")).sum().sum())
    empty_ratio = empty_cells / total_cells

    header_lengths = [len(str(col)) for col in df.columns]
    avg_header_len = sum(header_lengths) / max(len(header_lengths), 1)
    long_header_ratio = sum(1 for length in header_lengths if length > 90) / max(cols, 1)

    values = [str(value) for value in df.values.flatten().tolist()]
    one_word_ratio = sum(1 for value in values if value.strip() and len(value.split()) == 1) / max(
        sum(1 for value in values if value.strip()),
        1,
    )

    row_bonus = min(rows, 12) / 12.0
    col_bonus = min(cols, 10) / 10.0
    raw_score = _raw_rows_score(raw_rows or [])
    label_fragment_penalty = _first_column_fragment_penalty(df)
    wrapped_single_record_penalty = _wrapped_single_record_penalty(df)

    return (
        1.0
        + row_bonus * 0.45
        + col_bonus * 0.25
        + raw_score * 0.35
        - empty_ratio * 0.85
        - long_header_ratio * 0.75
        - max(0.0, avg_header_len - 65.0) / 120.0
        - one_word_ratio * 0.08
        - label_fragment_penalty * 0.30
        - wrapped_single_record_penalty * 0.65
    )

def _build_enhanced_raw_rows(table_item, doc=None) -> list[list[str]]:
    row_first_rows = _build_row_first_raw_rows(table_item, doc) if doc is not None else []
    geometric_rows = _build_geometric_raw_rows(table_item)
    offset_rows = _build_offset_enhanced_raw_rows(table_item)
    if _raw_rows_score(row_first_rows) >= max(0.20, _raw_rows_score(geometric_rows) * 0.85, _raw_rows_score(offset_rows) * 0.85):
        return row_first_rows
    if _raw_rows_score(geometric_rows) >= max(0.20, _raw_rows_score(offset_rows) * 0.85):
        return geometric_rows
    return offset_rows

def _enhance_raw_table(raw_rows: list[list[str]]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame()

    num_cols = max(len(row) for row in raw_rows)
    normalized_rows = [row + [""] * (num_cols - len(row)) for row in raw_rows]
    
    # Strip leading spanning-title rows (1-2 populated cells, very low numeric)
    while len(normalized_rows) > 2:
        row = normalized_rows[0]
        populated = [str(v).strip() for v in row if str(v).strip()]
        row_fills_all_columns = len(populated) == num_cols
        single_long_title = (
            len(populated) == 1
            and (
                len(populated[0]) >= 30
                or re.match(r"^\(?status\b", populated[0].strip(), re.IGNORECASE)
            )
        )
        if (
            not row_fills_all_columns
            and (
                single_long_title
                or (len(populated) <= 2 and _numeric_density(populated) < 0.10)
            )
        ):
            normalized_rows = normalized_rows[1:]
        else:
            break

    header_count = _detect_header_row_count(normalized_rows)
    if _looks_like_key_value_without_header(normalized_rows):
        headers = [f"Col_{idx}" for idx in range(num_cols)]
        data_rows = [
            row for row in normalized_rows
            if _classify_row(row) != "empty"
        ]
        return pd.DataFrame(data_rows, columns=headers)

    headers = _merge_header_rows(normalized_rows[:header_count], num_cols)

    data_rows = []
    for row in normalized_rows[header_count:]:
        row_type = _classify_row(row)
        if row_type == "empty":
            continue
        if row_type == "section_header":
            # Preserve spanning subheaders: text in col 0, empty in rest
            subheader_row = [""] * num_cols
            text_parts = []
            seen_parts = set()
            for value in row:
                text = str(value).strip()
                key = text.lower()
                if text and key not in seen_parts:
                    text_parts.append(text)
                    seen_parts.add(key)
            subheader_row[0] = " ".join(text_parts)
            data_rows.append(subheader_row)
            continue
        data_rows.append(row)

    # Deduplicate subheader rows where same text appears in all columns
    for row in data_rows:
        non_empty = [str(v).strip() for v in row if str(v).strip()]
        unique_vals = set(non_empty)
        if len(unique_vals) == 1 and len(non_empty) >= 3:
            # All cells have same text — it's a spanning subheader
            text = next(iter(unique_vals))
            row[:] = [""] * num_cols
            row[0] = text

    return pd.DataFrame(data_rows, columns=headers)

def _build_pdfplumber_raw_rows(table_item, doc) -> list[list[str]]:
    pdf_path = _source_pdf_path_for_doc(doc)
    page_no = _table_page_no(table_item)
    if not pdf_path or not page_no:
        return []

    bbox = _table_cell_union_bbox(table_item)
    if not bbox:
        return []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_no - 1]
            # Convert Docling bbox to pdfplumber coordinates
            x0, y0, x1, y1 = bbox
            table_region = page.within_bbox((x0, y0, x1, y1))
            tables = table_region.extract_tables()
            if tables:
                return tables[0]
    except Exception as exc:
        print(f"[pdfplumber] Failed to extract: {exc}")
    return []

def export_table_smart(table_item, doc) -> pd.DataFrame:
    try:
        grid = table_item.data.grid
        if not grid:
            return _original_export_table_smart(table_item, doc)
    except Exception as exc:
        print(f"[tables] Could not read table grid; falling back to original logic: {exc}")
        return _original_export_table_smart(table_item, doc)

    candidates = []
    max_cols = max((len(row) for row in grid), default=0) if grid else 0

    builders = [
        ("row-first", lambda ti: _build_row_first_raw_rows(ti, doc)),
        ("geometric", _build_geometric_raw_rows),
        ("offset", _build_offset_enhanced_raw_rows),
    ]

    for label, builder in builders:
        try:
            raw_rows = builder(table_item)
            enhanced = _enhance_raw_table(raw_rows)
            if enhanced is not None and not enhanced.empty:
                candidates.append((label, _dataframe_quality_score(enhanced, raw_rows), enhanced, len(raw_rows)))
        except Exception as exc:
            print(f"[tables] {label} reconstruction failed; trying fallback: {exc}")

    try:
        original = _original_export_table_smart(table_item, doc)
        if original is not None and not original.empty:
            candidates.append(("original", _dataframe_quality_score(original, None), original, len(original) + 1))
    except Exception as exc:
        print(f"[tables] original export failed: {exc}")

    if not candidates:
        return pd.DataFrame()

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_score = candidates[0][1]
    best_raw_len = candidates[0][3]

    if candidates[0][0] == "row-first":
        fuller_candidates = [
            item for item in candidates[1:]
            if item[3] > best_raw_len and item[1] >= best_score * 0.82
        ]
        if fuller_candidates:
            fuller_candidates.sort(key=lambda item: (item[3], item[1]), reverse=True)
            candidates.insert(0, fuller_candidates[0])
            best_score = candidates[0][1]

    # Use pdfplumber if table is dense (>=8 cols) OR current best score is poor (<0.3)
    if max_cols >= 8 or best_score < 0.3:
        try:
            raw_rows = _build_pdfplumber_raw_rows(table_item, doc)
            if raw_rows:
                enhanced = _enhance_raw_table(raw_rows)
                if enhanced is not None and not enhanced.empty:
                    score = _dataframe_quality_score(enhanced, raw_rows)
                    if score > best_score:
                        candidates.insert(0, ("pdfplumber", score, enhanced, len(raw_rows)))
        except Exception as exc:
            print(f"[tables] pdfplumber fallback failed: {exc}")

    return candidates[0][2]

def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, ec = [], 0
    for col in df.columns:
        s = str(col).strip()
        if s in ("", "nan", "None"):
            s = f"Col_{ec}"
            ec += 1
        new_cols.append(s)
    df.columns = new_cols
    return df

def fix_split_cell_text(df: pd.DataFrame) -> pd.DataFrame:
    # Generic repair for cells where words were split by the PDF extraction
    if df.empty:
        return df
    df = df.copy()
    for col_idx in range(df.shape[1]):
        sample = df.iloc[:, col_idx].map(_safe_cell_value).str.strip()
        non_empty = sample[sample != ""]
        if non_empty.empty:
            continue
        # Skip columns where most values are numeric
        numeric_count = non_empty.apply(
            lambda v: bool(re.match(r'^[\d.,%()+\s-]+$', v))
        ).sum()
        if numeric_count / len(non_empty) > 0.5:
            continue

        def repair_cell(val: str) -> str:
            val = str(val).strip()
            if not val:
                return val
            # Fix trailing single-letter splits
            # Pattern: uppercase word (3+ chars) followed by a lone uppercase letter at end-of-string or before a space.
            val = re.sub(
                r'([A-Z]{3,})\s([A-Z])(?=\s|$)',
                lambda m: m.group(1) + m.group(2),
                val,
            )
            # Fix serial-number concatenation
            val = re.sub(
                r'(\d+)([A-Z]{2,})',
                lambda m: m.group(1) + ' ' + m.group(2),
                val,
            )
            # Fix mid-word space
            uppercase_join_stops = {
                "AND", "OR", "OF", "THE", "IN", "ON", "FOR", "TO", "BY", "UT", "NCT",
            }
            val = re.sub(
                r'([A-Z]{2,})\s([A-Z]{1,3})(?=\s|$)',
                lambda m: (
                    f"{m.group(1)} {m.group(2)}"
                    if m.group(2).upper() in uppercase_join_stops
                    else m.group(1) + m.group(2)
                ),
                val,
            )
            return val

        df.iloc[:, col_idx] = df.iloc[:, col_idx].apply(repair_cell)
    return df

def _repair_alpha_prefix_numeric_bleed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] < 2:
        return df
    repaired = df.copy()
    for col_idx in range(1, repaired.shape[1]):
        sample = repaired.iloc[:, col_idx].map(_safe_cell_value).str.strip()
        non_empty = sample[sample != ""]
        if non_empty.empty:
            continue

        def numeric_after_prefix(value: str) -> bool:
            value = str(value).strip()
            if _numeric_like(value) or _numeric_density([value]) >= 0.60:
                return True
            match = re.match(r"^([A-Za-z][A-Za-z\s/&().-]{1,45})\s+([\d,]+(?:\.\d+)?(?:\s*%|\s*[A-Za-z.()/-]+)?)$", value)
            return bool(match and (_numeric_like(match.group(2)) or _numeric_density([match.group(2)]) >= 0.60))

        numericish_ratio = non_empty.map(numeric_after_prefix).sum() / len(non_empty)
        if numericish_ratio < 0.65:
            continue

        for row_pos, value in enumerate(repaired.iloc[:, col_idx].map(_safe_cell_value).tolist()):
            value = value.strip()
            match = re.match(
                r"^([A-Za-z][A-Za-z\s/&().-]{1,45})\s+([\d,]+(?:\.\d+)?(?:\s*%|\s*[A-Za-z.()/-]+)?)$",
                value,
            )
            if not match:
                continue
            prefix, numeric_tail = match.group(1).strip(), match.group(2).strip()
            left_value = _safe_cell_value(repaired.iat[row_pos, col_idx - 1]).strip()
            if not left_value or _numeric_like(left_value) or not re.search(r"[A-Za-z]", left_value):
                continue
            trailing_paren = re.match(r"^(.*?)(\s+\([^)]+\))$", left_value)
            trailing_unit = re.match(r"^(.*?)(\b[A-Za-z][A-Za-z/-]*\s+\([^)]+\))$", left_value)
            if trailing_unit and len(prefix.split()) == 1:
                repaired.iat[row_pos, col_idx - 1] = f"{trailing_unit.group(1).strip()} {prefix} {trailing_unit.group(2)}".strip()
            elif trailing_paren:
                repaired.iat[row_pos, col_idx - 1] = f"{trailing_paren.group(1).strip()} {prefix}{trailing_paren.group(2)}".strip()
            else:
                repaired.iat[row_pos, col_idx - 1] = f"{left_value} {prefix}".strip()
            repaired.iat[row_pos, col_idx] = numeric_tail
    return repaired

def _repair_leaked_next_row_label(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] < 3:
        return df
    repaired = df.copy()

    text_col_idx = 1 if df.shape[1] > 2 else 0
    text_col = repaired.columns[text_col_idx]
    data_cols = list(repaired.columns[text_col_idx + 1:])
    if not data_cols:
        return repaired

    for pos in range(1, len(repaired)):
        row_label = str(repaired.iloc[pos, text_col_idx]).strip()
        prev_label = str(repaired.iloc[pos - 1, text_col_idx]).strip()
        if not row_label or not prev_label:
            continue
        if not re.match(r"^\([^)]+\)$", row_label):
            continue
        row_payload = [
            str(repaired.iloc[pos][col]).strip()
            for col in data_cols
            if str(repaired.iloc[pos][col]).strip()
        ]
        if not row_payload or _numeric_density(row_payload) < 0.50:
            continue
        match = re.match(r"^(.+?\))\s+([A-Z][A-Za-z0-9].+)$", prev_label)
        if not match:
            continue
        repaired.iat[pos - 1, text_col_idx] = match.group(1).strip()
        repaired.iat[pos, text_col_idx] = f"{match.group(2).strip()} {row_label}".strip()
    return repaired


# Keep the old name as an alias so any existing callers still work.
fix_split_state_names = fix_split_cell_text

# ---------------------------------------------------------------------------
# 7b. Generic merged-row repair (post-processing)
# ---------------------------------------------------------------------------

# Pattern: two numeric-like tokens separated by whitespace.

_MULTI_VAL_RE = re.compile(
    r'^[\(\[]?(?:\d+(?:[.,]\d+)?%?|na)[\)\]]?'   # first value
    r'\s+'
    r'[\(\[]?(?:\d+(?:[.,]\d+)?%?|na)[\)\]]?$',   # second value
    re.IGNORECASE,
)

# Pattern: parenthesized number followed by a non-paren number in one cell.
# value leaked from an adjacent row.
_PAREN_STRAY_RE = re.compile(
    r'^(\([\d.,]+\))\s+([\d.,]+)$'
)


def _repair_merged_cell_values(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty or len(df.columns) < 2:
        return df

    values = df.astype(str).values.tolist()
    cols = list(df.columns)
    num_cols = len(cols)

    # --- Pass 1: split fully-merged rows -----------------------------------
    split_rows = []
    for row in values:
        # Count data cells (skip col 0 = row label) with multi-value pattern
        data_cells_count = 0
        multi_count = 0
        for ci in range(1, num_cols):
            v = str(row[ci]).strip()
            if v and v.lower() not in ('', 'nan', 'none'):
                data_cells_count += 1
                if _MULTI_VAL_RE.match(v):
                    multi_count += 1

        if data_cells_count >= 3 and multi_count / data_cells_count >= 0.30:
            # This row is very likely two source rows merged together.
            row1, row2 = [], []
            for ci in range(num_cols):
                v = str(row[ci]).strip()
                if ci == 0:
                    # Try splitting compound labels:

                    lm = re.match(r'^(.+?\([^)]+\))\s+(.+?\([^)]+\))$', v)
                    if lm:
                        row1.append(lm.group(1))
                        row2.append(lm.group(2))
                    else:
                        row1.append(v)
                        row2.append('')
                elif _MULTI_VAL_RE.match(v):
                    parts = v.split()
                    row1.append(parts[0])
                    row2.append(parts[1] if len(parts) > 1 else '')
                else:
                    row1.append(v)
                    row2.append('')
            split_rows.append(row1)
            split_rows.append(row2)
        else:
            split_rows.append(list(row))

    # --- Pass 2: clean stray leaked values ---------------------------------
    for ri in range(len(split_rows)):
        row = split_rows[ri]
        # Count how many data cells are parenthesized
        paren_count = 0
        data_count = 0
        stray_col = None
        for ci in range(1, num_cols):
            v = str(row[ci]).strip()
            if not v or v.lower() in ('nan', 'none'):
                continue
            data_count += 1
            if re.match(r'^\([\d.,]+\)$', v):
                paren_count += 1
            elif _PAREN_STRAY_RE.match(v):
                stray_col = ci

        # If the row is predominantly parenthesized and has exactly one stray cell, clean it.
        if stray_col is not None and data_count >= 3 and paren_count / data_count >= 0.50:
            m = _PAREN_STRAY_RE.match(str(row[stray_col]).strip())
            if m:
                paren_val, plain_val = m.group(1), m.group(2)
                row[stray_col] = paren_val
                # Try to place the plain value in the adjacent row below
                # if that row has an empty cell in the same column.
                for neighbor in (ri + 1, ri - 1):
                    if 0 <= neighbor < len(split_rows):
                        nb_val = str(split_rows[neighbor][stray_col]).strip()
                        if not nb_val or nb_val.lower() in ('', 'nan', 'none'):
                            split_rows[neighbor][stray_col] = plain_val
                            break

    return pd.DataFrame(split_rows, columns=cols)

def _repair_single_row_spanning_label(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[0] != 1 or df.shape[1] < 3:
        return df

    first_header = _safe_cell_value(df.columns[0]).strip()
    first_value = _safe_cell_value(df.iat[0, 0]).strip()
    if not first_header or not first_value:
        return df
    if _numeric_like(first_header) or _numeric_like(first_value):
        return df
    if not re.search(r"[A-Za-z]", first_header) or not re.search(r"[A-Za-z]", first_value):
        return df
    if not _looks_like_wrapped_text_fragment(first_value):
        return df

    other_headers = [_safe_cell_value(col).strip() for col in df.columns[1:]]
    other_values = [_safe_cell_value(value).strip() for value in df.iloc[0, 1:].tolist()]
    header_numeric_ratio = sum(
        1 for value in other_headers
        if _numeric_like(value) or re.search(r"\d", value)
    ) / max(len(other_headers), 1)
    value_numeric_ratio = sum(
        1 for value in other_values
        if _numeric_like(value) or _numeric_density([value]) >= 0.60
    ) / max(len(other_values), 1)
    if header_numeric_ratio < 0.70 or value_numeric_ratio < 0.70:
        return df

    repaired = df.copy()
    new_columns = list(repaired.columns)
    new_columns[0] = _clean_header_piece(f"{first_header} {first_value}")
    repaired.columns = new_columns
    repaired.iat[0, 0] = ""
    return repaired


def export_table_to_csv(conv_result, table_idx: int, output_path: str) -> pd.DataFrame:
    doc = conv_result.document
    table_item = doc.tables[table_idx]
    df = export_table_smart(table_item, doc)
    if df is None or df.empty:
        df = pd.DataFrame()
    else:
        df = collapse_to_data_columns(df)
        df = fix_split_cell_text(df)
        df = _repair_leaked_next_row_label(df)
        df = _repair_alpha_prefix_numeric_bleed(df)
        df = _repair_merged_cell_values(df)
        df = _repair_single_row_spanning_label(df)
    
    df = _stringify_dataframe(df)
    df.columns = make_columns_unique(df.columns)
    df = deduplicate_columns(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df

def auto_csv_name(pdf_path: str, query: str) -> str:
    stem = Path(pdf_path).stem
    slug = re.sub(r"[^\w]+", "_", query.strip()[:40]).strip("_")
    return f"{stem}_{slug}.csv"

# ---------------------------------------------------------------------------
# 8. Cross-page Chain Handling
# ---------------------------------------------------------------------------

MERGE_DEBUG = False

def _context_text(ctx: dict) -> str:
    return " ".join(
        str(ctx.get(key, "") or "").strip()
        for key in ("section_header", "nearest_text", "full_context")
        if str(ctx.get(key, "") or "").strip()
    ).strip()

def _context_table_label(ctx: dict) -> str:
    text = _context_text(ctx)
    match = re.search(r"\b(?:table|tab\.?)\s*[:.-]?\s*(\d+(?:\.\d+)*[a-z]?)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else ""

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _is_generic_header(value: str) -> bool:
    value = _clean_header_piece(value).lower()
    return not value or bool(re.match(r"^(col|column|unnamed)[\s_-]*\d+$", value))

def _header_tokens(columns: list) -> set:
    pieces = []
    for col in columns:
        text = _clean_header_piece(col)
        if _is_generic_header(text):
            continue
        if _numeric_density([text]) > 0.55:
            continue
        pieces.append(text)
    return tokenize(" ".join(pieces))

def _header_similarity(a_cols: list, b_cols: list) -> float:
    a_text = " | ".join(_clean_header_piece(col).lower() for col in a_cols)
    b_text = " | ".join(_clean_header_piece(col).lower() for col in b_cols)
    return max(
        _jaccard(_header_tokens(a_cols), _header_tokens(b_cols)),
        difflib.SequenceMatcher(None, a_text, b_text).ratio() * 0.85,
    )

def _numeric_like(value: str) -> bool:
    value = str(value or "").strip()
    return bool(value and re.match(r"^[\d,./()%+\-\s:]+$", value))

def _data_like_header(columns: list) -> bool:
    values = [_clean_header_piece(col) for col in columns if _clean_header_piece(col)]
    if not values:
        return True
    generic_count = sum(1 for value in values if _is_generic_header(value))
    numeric_count = sum(1 for value in values if _numeric_like(value) or _numeric_density([value]) >= 0.55)
    first = values[0] if values else ""
    first_is_ordinal_label = bool(re.match(r"^\d+$", first)) and len(values) > 1
    return (
        generic_count / max(len(values), 1) >= 0.50
        or numeric_count / max(len(values), 1) >= 0.45
        or first_is_ordinal_label
        or _looks_like_data_row_label(first)
    )

def _table_page_no(table_item) -> int | None:
    try:
        if table_item.prov:
            return int(table_item.prov[0].page_no)
    except Exception:
        pass
    return None

def _page_height_for_table(table_item, doc) -> float:
    page_no = _table_page_no(table_item)
    pdf_path = _source_pdf_path_for_doc(doc)
    if page_no and pdf_path:
        pdf_doc = None
        try:
            pdf_doc = pymupdf.open(pdf_path)
            return float(pdf_doc[page_no - 1].rect.height)
        except Exception:
            pass
        finally:
            if pdf_doc is not None:
                pdf_doc.close()
    return 842.0

def _table_column_positions(table_item) -> list[float]:
    starts = _column_starts_from_grid(table_item)
    if len(starts) >= 2:
        return starts
    try:
        grid = table_item.data.grid
        axis = _infer_column_axis(grid)
        if axis == "x":
            return _canonical_columns_from_geometry(grid, axis)
    except Exception:
        pass
    return []

def _normalize_positions(positions: list[float], bbox: tuple | None) -> list[float]:
    if not positions:
        return []
    if bbox is None:
        left, right = min(positions), max(positions)
    else:
        left, right = float(bbox[0]), float(bbox[2])
    width = max(right - left, 1.0)
    return [(float(pos) - left) / width for pos in positions]

def _column_alignment_similarity(a: dict, b: dict) -> float:
    a_pos = _normalize_positions(a["col_positions"], a["bbox"])
    b_pos = _normalize_positions(b["col_positions"], b["bbox"])
    if not a_pos or not b_pos:
        return 1.0 if a["n_cols"] == b["n_cols"] else 0.0

    if len(a_pos) == len(b_pos):
        distances = [abs(x - y) for x, y in zip(a_pos, b_pos)]
    else:
        longer, shorter = (a_pos, b_pos) if len(a_pos) >= len(b_pos) else (b_pos, a_pos)
        distances = [min(abs(pos - ref) for ref in longer) for pos in shorter]
        count_ratio = len(shorter) / max(len(longer), 1)
        if count_ratio < 0.75:
            return 0.0
    return max(0.0, 1.0 - (median(distances) * 3.0))

def _column_type(values: list[str]) -> str:
    non_empty = [str(value).strip() for value in values if str(value).strip()]
    if not non_empty:
        return "empty"
    numeric_ratio = sum(1 for value in non_empty if _numeric_like(value) or _numeric_density([value]) >= 0.60) / len(non_empty)
    text_ratio = sum(1 for value in non_empty if re.search(r"[A-Za-z]", value)) / len(non_empty)
    if numeric_ratio >= 0.70:
        return "numeric"
    if text_ratio >= 0.70 and numeric_ratio <= 0.30:
        return "text"
    return "mixed"

def _column_types(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    return [_column_type(df.iloc[:, idx].astype(str).head(8).tolist()) for idx in range(df.shape[1])]

def _type_similarity(a_types: list[str], b_types: list[str]) -> float:
    if not a_types or not b_types:
        return 0.5
    count = min(len(a_types), len(b_types))
    if count == 0:
        return 0.0
    matches = 0.0
    for idx in range(count):
        if a_types[idx] == b_types[idx]:
            matches += 1.0
        elif "empty" in {a_types[idx], b_types[idx]}:
            matches += 0.5
        elif "mixed" in {a_types[idx], b_types[idx]}:
            matches += 0.65
    return matches / count

def _row_density(df: pd.DataFrame) -> float:
    if df is None or df.empty or df.shape[1] == 0:
        return 0.0
    densities = []
    for _, row in df.astype(str).head(10).iterrows():
        densities.append(sum(1 for value in row.tolist() if value.strip()) / df.shape[1])
    return median(densities) if densities else 0.0

def _table_signature(ctx: dict, conv_result, cache: dict[int, dict]) -> dict:
    idx = int(ctx["table_idx"])
    if idx in cache:
        return cache[idx]

    doc = conv_result.document
    table_item = doc.tables[idx]
    df = _stringify_dataframe(export_table_smart(table_item, doc))
    df = deduplicate_columns(df) if not df.empty else df
    bbox = _table_cell_union_bbox(table_item)
    columns = list(df.columns)
    sig = {
        "idx": idx,
        "ctx": ctx,
        "table_item": table_item,
        "page": _table_page_no(table_item),
        "page_height": _page_height_for_table(table_item, doc),
        "bbox": bbox,
        "df": df,
        "columns": columns,
        "n_cols": len(columns),
        "col_positions": _table_column_positions(table_item),
        "header_tokens": _header_tokens(columns),
        "data_like_header": _data_like_header(columns),
        "col_types": _column_types(df),
        "row_density": _row_density(df),
        "context_tokens": tokenize(_context_text(ctx)),
    }
    cache[idx] = sig
    return sig

def _bbox_alignment_ok(a: dict, b: dict) -> tuple[bool, str]:
    if a["bbox"] is None or b["bbox"] is None:
        return True, "bbox unavailable"
    a_left, _, a_right, _ = a["bbox"]
    b_left, _, b_right, _ = b["bbox"]
    page_width = max(a_right, b_right, 1.0)
    left_delta = abs(a_left - b_left) / page_width
    right_delta = abs(a_right - b_right) / page_width
    a_width = max(a_right - a_left, 1.0)
    b_width = max(b_right - b_left, 1.0)
    width_ratio = min(a_width, b_width) / max(a_width, b_width)
    overlap_ratio = max(0.0, min(a_right, b_right) - max(a_left, b_left)) / max(min(a_width, b_width), 1.0)
    strict_ok = left_delta <= 0.08 and right_delta <= 0.08 and width_ratio >= 0.78
    overlap_ok = (
        overlap_ratio >= 0.85
        and width_ratio >= 0.45
        and left_delta <= 0.20
        and right_delta <= 0.20
    )
    ok = strict_ok or overlap_ok
    return ok, (
        f"bbox left={left_delta:.2f}, right={right_delta:.2f}, "
        f"width={width_ratio:.2f}, overlap={overlap_ratio:.2f}"
    )

def _vertical_continuity_ok(a: dict, b: dict) -> tuple[bool, str]:
    if a["page"] is None or b["page"] is None:
        return True, "page unavailable"
    page_gap = b["page"] - a["page"]
    if page_gap < 0 or page_gap > 1:
        return False, f"page gap {page_gap}"
    if a["bbox"] is None or b["bbox"] is None:
        return page_gap <= 1, "bbox unavailable"

    if page_gap == 0:
        gap = float(b["bbox"][1]) - float(a["bbox"][3])
        ok = -4.0 <= gap <= a["page_height"] * 0.14
        return ok, f"same-page vertical gap={gap:.1f}"

    prev_bottom_ratio = float(a["bbox"][3]) / max(float(a["page_height"]), 1.0)
    next_top_ratio = float(b["bbox"][1]) / max(float(b["page_height"]), 1.0)
    ok = prev_bottom_ratio >= 0.58 or next_top_ratio <= 0.25
    return ok, f"page-break positions bottom={prev_bottom_ratio:.2f}, top={next_top_ratio:.2f}"

def _semantic_continuity_ok(a: dict, b: dict) -> tuple[bool, str]:
    b_ctx = _context_text(b["ctx"])
    if not b_ctx:
        return True, "candidate has no new context"
    a_ctx = _context_text(a["ctx"])
    if not a_ctx:
        return False, "candidate introduces context after orphan base"
    overlap = _jaccard(tokenize(a_ctx), tokenize(b_ctx))
    seq = difflib.SequenceMatcher(None, a_ctx.lower(), b_ctx.lower()).ratio()
    ok = max(overlap, seq * 0.80) >= 0.35
    return ok, f"context similarity={max(overlap, seq * 0.80):.2f}"

def _header_continuity_ok(a: dict, b: dict) -> tuple[bool, str]:
    sim = _header_similarity(a["columns"], b["columns"])
    if sim >= 0.50:
        return True, f"header similarity={sim:.2f}"
    if b["data_like_header"]:
        return True, f"candidate starts with data-like row/header similarity={sim:.2f}"
    if not b["header_tokens"] and a["n_cols"] == b["n_cols"]:
        return True, f"candidate header is generic/header similarity={sim:.2f}"
    # Relaxed: if column count matches and B is on the very next page,
    # garbled headers on continuation pages should not block merge.
    if a["n_cols"] == b["n_cols"] and a["page"] is not None and b["page"] is not None:
        if b["page"] - a["page"] == 1:
            return True, f"consecutive-page same col count/header similarity={sim:.2f}"
    return False, f"header mismatch={sim:.2f}"

def _column_continuity_ok(a: dict, b: dict) -> tuple[bool, str]:
    align = _column_alignment_similarity(a, b)
    same_count = a["n_cols"] == b["n_cols"] and a["n_cols"] > 0
    count_ratio = min(a["n_cols"], b["n_cols"]) / max(a["n_cols"], b["n_cols"], 1)
    ok = same_count or (align >= 0.82 and count_ratio >= 0.75)
    return ok, f"columns {a['n_cols']} vs {b['n_cols']}, alignment={align:.2f}"

def _row_pattern_ok(a: dict, b: dict) -> tuple[bool, str]:
    density_delta = abs(a["row_density"] - b["row_density"])
    type_sim = _type_similarity(a["col_types"], b["col_types"])
    ok = density_delta <= 0.45 and type_sim >= 0.45
    return ok, f"row density delta={density_delta:.2f}, type similarity={type_sim:.2f}"

def _tables_mergeable(a: dict, b: dict) -> tuple[bool, list[str]]:
    checks = [
        _vertical_continuity_ok(a, b),
        _bbox_alignment_ok(a, b),
        _column_continuity_ok(a, b),
        _header_continuity_ok(a, b),
        _semantic_continuity_ok(a, b),
        _row_pattern_ok(a, b),
    ]
    failed = [reason for ok, reason in checks if not ok]
    if failed:
        return False, failed

    a_table_label = _context_table_label(a["ctx"])
    b_table_label = _context_table_label(b["ctx"])
    if a_table_label and b_table_label and a_table_label != b_table_label:
        return False, [f"anti-over-merge: distinct table labels {a_table_label} vs {b_table_label}"]

    a_section = _clean_header_piece(a["ctx"].get("section_header", "")).lower()
    b_section = _clean_header_piece(b["ctx"].get("section_header", "")).lower()
    if (
        a_section
        and b_section
        and a_section != b_section
        and _header_similarity(a["columns"], b["columns"]) >= 0.75
    ):
        return False, ["anti-over-merge: repeated schema under different section titles"]

    # Anti-over-merge: if BOTH tables have real (non-data-like) headers that
    # are very similar, they are likely two DISTINCT tables that happen to share
    # the same structure. Reject merge.
    # Exception: if B is on the immediately next page AND B starts near the top,
    # this is almost certainly a continuation, not a new table.
    if (not a["data_like_header"] and not b["data_like_header"]
            and _header_similarity(a["columns"], b["columns"]) >= 0.80):
        is_consecutive_page = (
            a["page"] is not None and b["page"] is not None
            and b["page"] - a["page"] == 1
        )
        b_starts_at_top = (
            b["bbox"] is not None
            and float(b["bbox"][1]) / max(float(b["page_height"]), 1.0) <= 0.20
        )
        if not (is_consecutive_page and b_starts_at_top):
            return False, ["anti-over-merge: both have real identical headers"]

    return True, [reason for _, reason in checks]

def _log_merge_decision(a_idx: int, b_idx: int, ok: bool, reasons: list[str], debug: bool = False):
    if not (debug or MERGE_DEBUG):
        return
    status = "MERGE" if ok else "SKIP"
    print(f"[merge] Tables {a_idx + 1}->{b_idx + 1}: {status} ({'; '.join(reasons)})")

def build_merge_groups(all_contexts: list, conv_result, debug: bool = False) -> list[list[dict]]:
    contexts = sorted(all_contexts, key=lambda ctx: ctx["table_idx"])
    idx_to_ctx = {ctx["table_idx"]: ctx for ctx in contexts}
    parent = {ctx["table_idx"]: ctx["table_idx"] for ctx in contexts}
    cache: dict[int, dict] = {}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a_idx: int, b_idx: int):
        a_root, b_root = find(a_idx), find(b_idx)
        if a_root != b_root:
            parent[b_root] = a_root

    for left, right in zip(contexts, contexts[1:]):
        if right["table_idx"] != left["table_idx"] + 1:
            continue
        a_sig = _table_signature(left, conv_result, cache)
        b_sig = _table_signature(right, conv_result, cache)
        ok, reasons = _tables_mergeable(a_sig, b_sig)
        _log_merge_decision(left["table_idx"], right["table_idx"], ok, reasons, debug)
        if ok:
            union(left["table_idx"], right["table_idx"])

    groups = {}
    for idx in idx_to_ctx:
        groups.setdefault(find(idx), []).append(idx_to_ctx[idx])
    return [sorted(group, key=lambda ctx: ctx["table_idx"]) for group in groups.values()]

def find_cross_page_chain(anchor_ctx: dict, all_contexts: list, conv_result=None, debug: bool = False) -> list:
    if conv_result is None:
        return [anchor_ctx]
    groups = build_merge_groups(all_contexts, conv_result, debug=debug)
    anchor_idx = anchor_ctx["table_idx"]
    for group in groups:
        if any(ctx["table_idx"] == anchor_idx for ctx in group):
            return group
    return [anchor_ctx]

def _drop_repeated_header_rows(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    base_tokens = _header_tokens(base_cols)
    # Also build a set of cleaned header strings for direct comparison
    base_header_strings = {_clean_header_piece(col).lower() for col in base_cols
                          if _clean_header_piece(col)}
    keep_rows = []
    for _, row in df.iterrows():
        row_values = [str(value) for value in row.tolist()]
        row_tokens = tokenize(" ".join(row_values))
        # Token-based match
        if base_tokens and _jaccard(base_tokens, row_tokens) >= 0.55:
            continue
        # Direct string match: if most non-empty cells match header strings
        non_empty = [_clean_header_piece(v).lower() for v in row_values if _clean_header_piece(v)]
        if non_empty and base_header_strings:
            match_ratio = sum(1 for v in non_empty if v in base_header_strings) / len(non_empty)
            if match_ratio >= 0.50 and len(non_empty) >= 3:
                continue
        keep_rows.append(row_values)
    return pd.DataFrame(keep_rows, columns=list(df.columns))

def _align_fragment_to_base(base_sig: dict, frag_sig: dict) -> pd.DataFrame:
    base_cols = list(base_sig["columns"])
    frag = _stringify_dataframe(frag_sig["df"])
    if frag.empty:
        return pd.DataFrame(columns=base_cols)

    header_sim = _header_similarity(base_cols, list(frag.columns))
    if len(frag.columns) == len(base_cols):
        if frag_sig["data_like_header"] and header_sim < 0.50:
            recovered = pd.DataFrame([list(frag.columns)], columns=base_cols)
            frag.columns = base_cols
            frag = pd.concat([recovered, frag], ignore_index=True)
        else:
            frag.columns = base_cols
        return _drop_repeated_header_rows(frag, base_cols)

    base_pos = _normalize_positions(base_sig["col_positions"], base_sig["bbox"])
    frag_pos = _normalize_positions(frag_sig["col_positions"], frag_sig["bbox"])
    if base_pos and frag_pos:
        aligned = pd.DataFrame("", index=range(len(frag)), columns=base_cols)
        for frag_idx, col in enumerate(frag.columns):
            if frag_idx >= len(frag_pos):
                continue
            base_idx = min(range(len(base_pos)), key=lambda idx: abs(base_pos[idx] - frag_pos[frag_idx]))
            aligned.iloc[:, base_idx] = frag.iloc[:, frag_idx].values
        return _drop_repeated_header_rows(aligned, base_cols)

    try:
        frag = frag.iloc[:, :len(base_cols)].copy()
        frag.columns = base_cols[:frag.shape[1]]
        for col in base_cols[frag.shape[1]:]:
            frag[col] = ""
        return _drop_repeated_header_rows(frag[base_cols], base_cols)
    except ValueError:
        return pd.DataFrame(columns=base_cols)

def merge_and_export(conv_result, chain: list, output_path: str) -> pd.DataFrame:
    doc = conv_result.document
    chain = sorted(chain, key=lambda c: c["table_idx"])
    if not chain:
        merged = pd.DataFrame()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False, encoding="utf-8-sig")
        return merged

    cache: dict[int, dict] = {}
    signatures = [_table_signature(ctx, conv_result, cache) for ctx in chain]
    base_sig = signatures[0]
    aligned = [_stringify_dataframe(base_sig["df"])]
    aligned[0] = deduplicate_columns(aligned[0])
    base_sig["columns"] = list(aligned[0].columns)

    for frag_sig in signatures[1:]:
        aligned.append(_align_fragment_to_base(base_sig, frag_sig))

    merged = pd.concat(aligned, ignore_index=True)
    merged = merged.drop_duplicates()
    merged = fix_split_cell_text(merged)
    merged = _repair_leaked_next_row_label(merged)
    merged = _repair_alpha_prefix_numeric_bleed(merged)
    merged = _repair_merged_cell_values(merged)
    merged = _repair_single_row_spanning_label(merged)
    merged = _stringify_dataframe(merged)
    merged.columns = make_columns_unique(merged.columns)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged

# ---------------------------------------------------------------------------
# 9. Figure / Image Extraction
# ---------------------------------------------------------------------------

def _render_picture(pic, conv_result):
    try:
        img = pic.image.pil_image
        if img is not None:
            return img
    except AttributeError:
        pass
    try:
        img = pic.get_image(conv_result.document)
        if img is not None:
            return img
    except Exception:
        pass
    return None

def extract_figures(conv_result, pdf_path: str, output_dir: str = None) -> list:
    doc = conv_result.document
    pdf_stem = Path(pdf_path).stem
    out_dir = Path(output_dir) if output_dir else Path(pdf_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pictures = list(doc.pictures)
    saved_paths = []

    if not pictures:
        print("[figures] No PictureItem elements found.")
        print("[figures] Ensure --images flag was passed at parse time.")
        return []

    print(f"[figures] {len(pictures)} picture(s) found — saving to {out_dir}")

    for idx, pic in enumerate(pictures, start=1):
        pil_img = _render_picture(pic, conv_result)
        out_path = out_dir / f"{pdf_stem}-figure-{idx}.png"
        if pil_img is not None:
            pil_img.save(str(out_path), format="PNG")
            saved_paths.append(str(out_path))
            print(f"[figures] Saved: {out_path.name}")
        else:
            print(f"[figures] SKIP figure {idx}: no image data available.")

    print(f"[figures] Done — {len(saved_paths)} PNG(s) saved.")
    return saved_paths

def extract_figures_to_memory(conv_result) -> list:
    pictures = list(conv_result.document.pictures)
    results = []
    for idx, pic in enumerate(pictures, start=1):
        pil_img = _render_picture(pic, conv_result)
        if pil_img is None:
            continue
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        results.append((f"figure-{idx}.png", buf.getvalue()))
    return results

def extract_best_figure_to_memory(
    conv_result,
    query: str,
    fig_contexts: list,
    threshold: float = DEFAULT_THRESHOLD,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    if not fig_contexts:
        return {}

    ranked = rank_figures(query, fig_contexts)
    best_score, best_ctx = ranked[0]
    pic_idx = best_ctx["pic_idx"]

    pictures = list(conv_result.document.pictures)
    pil_img = None
    if pic_idx < len(pictures):
        pil_img = _render_picture(pictures[pic_idx], conv_result)

    png_bytes = None
    if pil_img is not None:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    return {
        "pic_idx": pic_idx,
        "score": best_score,
        "caption": best_ctx.get("caption", ""),
        "section_header": best_ctx.get("section_header", ""),
        "nearest_text": best_ctx.get("nearest_text", ""),
        "filename": f"figure-{pic_idx + 1}.png",
        "png_bytes": png_bytes,
        "ranked": ranked,
    }

# ---------------------------------------------------------------------------
# 10. Debug Tool
# ---------------------------------------------------------------------------

def debug_document_labels(conv_result):
    print("\n--- DEBUG: All document item labels ---")
    for item, level in conv_result.document.iterate_items():
        page_no = 0
        try:
            if item.prov:
                page_no = item.prov[0].page_no
        except Exception:
            pass
        text_preview = getattr(item, "text", "")[:80] or ""
        print(f" [p{page_no}] {str(item.label):<30} | {text_preview}")
    print("--- END DEBUG ---\n")

# ---------------------------------------------------------------------------
# 11. Main Pipeline Orchestrator & Pre-flight
# ---------------------------------------------------------------------------

def preprocess_pdf_orientation(pdf_path: str) -> str:
    """
    Pre-flight check for visually rotated pages.
    Uses text-line direction vectors and bounding-box aspect ratios.
    Also checks for pages with predominantly vertical text using pdfplumber
    as a secondary signal. Tries 90° rotation first; if that doesn't help,
    tries 270°.
    """
    doc = pymupdf.open(pdf_path)
    needs_correction = False

    for page in doc:
        text_dict = page.get_text("dict")
        horizontal_weight = 0.0
        vertical_weight = 0.0
        measured_lines = 0

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue

                x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))
                width = abs(float(x1) - float(x0))
                height = abs(float(y1) - float(y0))
                if width == 0 and height == 0:
                    continue

                char_weight = max(len(text), 1)
                dir_x, dir_y = line.get("dir", (1.0, 0.0))

                # Direction vector is the strongest signal for rotation
                if abs(dir_y) > abs(dir_x) * 0.8:  # relaxed threshold
                    vertical_weight += char_weight
                elif abs(dir_x) > abs(dir_y) * 0.8:
                    horizontal_weight += char_weight
                else:
                    # Ambiguous — use bbox aspect ratio as tiebreaker
                    if height > width * 1.2:
                        vertical_weight += char_weight * 0.5
                    else:
                        horizontal_weight += char_weight * 0.5
                measured_lines += 1

        total_weight = horizontal_weight + vertical_weight
        rotated_ratio = 0.0 if total_weight == 0 else vertical_weight / total_weight

        if measured_lines >= 3 and rotated_ratio >= 0.45 and vertical_weight > horizontal_weight:
            print(
                f"  [pre-flight] Page {page.number + 1} appears visually rotated "
                f"({rotated_ratio:.0%} vertical text). Rotating canvas 90°..."
            )
            page.set_rotation((page.rotation + 90) % 360)
            needs_correction = True

    if needs_correction:
        corrected_path = Path(tempfile.gettempdir()) / f"corrected_{Path(pdf_path).name}"
        doc.save(corrected_path)
        doc.close()
        return str(corrected_path)

    doc.close()
    return pdf_path

def run(
    pdf_path: str,
    query: str = "",
    output_csv: str = None,
    top_n: int = DEFAULT_TOP_N,
    threshold: float = DEFAULT_THRESHOLD,
    list_only: bool = False,
    use_ocr: bool = False,
    debug: bool = False,
    images: bool = False,
    output_dir: str = None,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    accurate: bool = False,
) -> pd.DataFrame | None:
    SEP = "-" * 65

    if not Path(pdf_path).exists():
        print(f"ERROR: File not found — {pdf_path}")
        return None

    print(SEP)
    print(" PDF TABLE + FIGURE EXTRACTOR | powered by docling")
    print(SEP)
    print(f" File : {pdf_path}")
    print(f" Query : {query or '(none — list/dump mode)'}")
    print(f" Images  : {'ON' if images else 'OFF'}")
    print(f" Accurate: {'ON (slow — ACCURATE mode)' if accurate else 'OFF (fast mode)'}")
    if images:
        print(f" Image scale : {image_scale}")
    print(SEP)

    # --- NEW HYBRID PIPELINE LOGIC ---
    print(" Running Pre-flight Checks (Orientation)...")
    safe_pdf_path = preprocess_pdf_orientation(pdf_path)

    conv_result = parse_pdf(
        safe_pdf_path, # Feed the safe, checked PDF path here
        use_ocr=use_ocr,
        extract_images=images,
        image_scale=image_scale,
        accurate=accurate,
    )
    all_items = build_all_items(conv_result)

    if debug:
        debug_document_labels(conv_result)

    if images:
        fig_contexts = extract_figure_contexts(conv_result, all_items)
        n_pics = len(conv_result.document.pictures)
        eff_out_dir = output_dir or (str(Path(output_csv).parent) if output_csv else None)

        if n_pics == 0:
            print("[figures] No pictures detected in this PDF.")
        elif query.strip():
            print(f"\n Matching query against {n_pics} figures...")
            result = extract_best_figure_to_memory(
                conv_result, query, fig_contexts, threshold, top_n
            )

            if not result:
                print("[figures] No figure contexts could be built.")
            else:
                print(f"\n Top {min(top_n, len(result['ranked']))} figure matches:")
                for rank, (score, ctx) in enumerate(result["ranked"][:top_n], 1):
                    tag = " <- BEST MATCH" if rank == 1 else ""
                    print(f" {rank}. Score={score:.2f} Figure {ctx['pic_idx'] + 1}{tag}")
                    print(f" Caption : {ctx.get('caption', '')[:80] or 'none'}")
                    print(f" Section : {ctx.get('section_header', '')[:80] or 'none'}")

                if result["score"] < threshold:
                    print(f"\n [LOW CONFIDENCE] Score {result['score']:.2f} < threshold {threshold}.")
                    print(" The query may not match any figure caption closely.")

                if result["png_bytes"] is not None:
                    out_dir = Path(eff_out_dir) if eff_out_dir else Path(pdf_path).parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{Path(pdf_path).stem}-{result['filename']}"
                    out_path.write_bytes(result["png_bytes"])
                    print(f"\n DONE — saved best-match figure: {out_path}")
                else:
                    print("\n [WARNING] Could not render the best-match figure image.")
                    print(" Was --images passed at parse time?")
        else:
            extract_figures(conv_result, pdf_path, eff_out_dir)

        print()

    n_tables = len(conv_result.document.tables)

    if n_tables == 0 and not images:
        print("! No tables detected.")
        print(" If this is a scanned PDF, retry with use_ocr=True.")
        return None

    if n_tables == 0:
        return None

    contexts = extract_table_contexts(conv_result, all_items)

    if list_only or not query.strip():
        print(f"\n {n_tables} tables found\n")
        for ctx in contexts:
            chain = find_cross_page_chain(ctx, contexts, conv_result, debug=debug)
            chain_ids = [c["table_idx"] + 1 for c in chain if c["table_idx"] != ctx["table_idx"]]
            cross_tag = (
                f" [CROSS-PAGE — will auto-merge with Tables {chain_ids}]"
                if len(chain) > 1 and chain[0]["table_idx"] == ctx["table_idx"]
                else ""
            )

            print(f" Table {ctx['table_idx'] + 1}{cross_tag}")
            print(f" Section : {ctx['section_header'][:100] or 'none'}")
            print(f" Nearest : {ctx['nearest_text'][:100] or 'none'}")
            print()
        print(" Tip: use the 'Nearest' text as your query for best matching accuracy.")
        return None

    print(f"\n Matching query against {n_tables} tables...")
    ranked = rank_tables(query, contexts)

    print(f"\n Top {min(top_n, len(ranked))} matches:")
    for rank, (score, ctx) in enumerate(ranked[:top_n], 1):
        tag = " <- BEST MATCH" if rank == 1 else ""
        print(f" {rank}. Score={score:.2f} Table {ctx['table_idx'] + 1}{tag}")
        print(f" Section : {ctx['section_header'][:80] or 'none'}")
        print(f" Nearest : {ctx['nearest_text'][:80] or 'none'}")

    best_score, best_ctx = ranked[0]

    if best_score < threshold:
        print(f"\n [LOW CONFIDENCE] Best score {best_score:.2f} < threshold {threshold}.")
        print(" Run with --list to inspect context text, then refine your query.")

    if not output_csv:
        output_csv = auto_csv_name(pdf_path, query)

    chain = find_cross_page_chain(best_ctx, contexts, conv_result, debug=debug)

    if len(chain) > 1:
        ids = [c["table_idx"] + 1 for c in chain]
        print(f"\n [CROSS-PAGE TABLE DETECTED]")
        print(f" Merging {len(chain)} fragments — Tables {ids} -> {output_csv}")
        df = merge_and_export(conv_result, chain, output_csv)
    else:
        print(f"\n Exporting Table {best_ctx['table_idx'] + 1} -> {output_csv}")
        df = export_table_to_csv(conv_result, best_ctx["table_idx"], output_csv)

    print(f"\n Preview (first 5 rows):")
    print(df.head(5).to_string(index=False))
    print(f"\n DONE — {output_csv} | {df.shape[0]} rows x {df.shape[1]} cols")
    print(SEP)

    return df

# 12. Split View Functions
# ---------------------------------------------------------------------------

def render_pdf_pages_as_images(pdf_path: str, scale: float = 2.0) -> list:
    """
    Render all pages of a PDF as images using PyMuPDF.
    Returns a list of tuples: (page_number, pil_image)
    """
    doc = pymupdf.open(pdf_path)
    page_images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page at specified scale
        mat = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_data))
        page_images.append((page_num + 1, pil_img))  # 1-based page number
    
    doc.close()
    return page_images

def group_figures_by_page(conv_result) -> dict:
    """
    Group extracted figures by their page numbers.
    Returns a dict: {page_number: [(pic_idx, pil_image, caption), ...]}
    """
    pictures = list(conv_result.document.pictures)
    page_figures = {}
    
    for pic_idx, pic in enumerate(pictures):
        # Get page number from provenance
        page_no = 0
        try:
            if pic.prov:
                page_no = pic.prov[0].page_no
        except Exception:
            page_no = 0
        
        # Render the picture
        pil_img = _render_picture(pic, conv_result)
        if pil_img is None:
            continue
        
        # Get caption from the picture item if available
        caption = getattr(pic, "text", "") or ""
        
        if page_no not in page_figures:
            page_figures[page_no] = []
        page_figures[page_no].append((pic_idx, pil_img, caption))
    
    return page_figures

def group_tables_by_page(conv_result) -> dict:
    """
    Group extracted tables by their page numbers.
    Returns a dict: {page_number: [(table_idx, table_item, context), ...]}
    """
    tables = list(conv_result.document.tables)
    page_tables = {}
    all_items = build_all_items(conv_result)
    contexts = extract_table_contexts(conv_result, all_items)
    contexts_by_idx = {ctx["table_idx"]: ctx for ctx in contexts}
    
    for table_idx, table in enumerate(tables):
        page_no = 0
        try:
            if table.prov:
                page_no = table.prov[0].page_no
        except Exception:
            page_no = 0
        
        table_context = contexts_by_idx.get(table_idx)
        
        if page_no not in page_tables:
            page_tables[page_no] = []
        page_tables[page_no].append((table_idx, table, table_context))
    
    return page_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tables (CSV) and/or figures (PNG) from a PDF using docling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("query", nargs="?", default="", help="Query string")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path")
    parser.add_argument("--output-dir", "-d", default=None, help="Directory to save PNGs")
    parser.add_argument("--top", "-n", type=int, default=DEFAULT_TOP_N, help="Top N matches")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Threshold")
    parser.add_argument("--list", "-l", action="store_true", help="List all tables")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR")
    parser.add_argument("--debug", action="store_true", help="Print labels")
    parser.add_argument("--images", action="store_true", help="Extract figures")
    parser.add_argument("--image-scale", type=float, default=DEFAULT_IMAGE_SCALE, help="Image scale")
    parser.add_argument("--accurate", action="store_true", help="TableFormer ACCURATE mode")

    args = parser.parse_args()
    run(
        pdf_path=args.pdf,
        query=args.query,
        output_csv=args.output,
        top_n=args.top,
        threshold=args.threshold,
        list_only=args.list,
        use_ocr=args.ocr,
        debug=args.debug,
        images=args.images,
        output_dir=args.output_dir,
        image_scale=args.image_scale,
        accurate=args.accurate,
    )
