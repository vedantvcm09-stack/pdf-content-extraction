from __future__ import annotations

import io
import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("store")
if not logger.handlers:
    _fh = logging.FileHandler(
        Path(__file__).resolve().parent / "store.log", encoding="utf-8"
    )
    _ch = logging.StreamHandler()
    _fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _fh.setFormatter(_fmt)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Root directory
# ---------------------------------------------------------------------------

# Anchor relative to this file so the store lives next to the code,
# regardless of where streamlit is launched from.
EXTRACTIONS_ROOT = Path(__file__).resolve().parent / "extractions"


# ---------------------------------------------------------------------------
# IDs & filenames
# ---------------------------------------------------------------------------

def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem or "document"


def compute_pdf_id(pdf_bytes: bytes, filename: str) -> str:
    """Stable ID combining a sanitized stem with a short content hash."""
    h = hashlib.md5(pdf_bytes).hexdigest()[:8]
    return f"{_safe_stem(filename)}_{h}"


def image_filename(stem: str, idx: int, page: int) -> str:
    return f"img{idx}_pg{page}_{stem}.png"


def table_filename(stem: str, idx: int, page: int) -> str:
    return f"tab{idx}_pg{page}_{stem}.csv"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def extraction_dir(pdf_id: str) -> Path:
    d = EXTRACTIONS_ROOT / pdf_id
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "summaries").mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(pdf_id: str) -> Path:
    return extraction_dir(pdf_id) / "manifest.json"


def source_pdf_path(pdf_id: str) -> Path:
    return extraction_dir(pdf_id) / "source.pdf"


# ---------------------------------------------------------------------------
# Manifest read / write
# ---------------------------------------------------------------------------

def _new_manifest(pdf_id: str, filename: str, stem: str, md5: str) -> dict:
    return {
        "pdf_id": pdf_id,
        "filename": filename,
        "stem": stem,
        "md5": md5,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "page_count": 0,
        "images": [],
        "tables": [],
    }


def load_manifest(pdf_id: str) -> Optional[dict]:
    p = manifest_path(pdf_id)
    if not p.exists():
        logger.debug(f"[load_manifest] No manifest file at {p}")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[load_manifest] Failed to parse {p}: {e}")
        return None


def save_manifest(manifest: dict) -> None:
    p = manifest_path(manifest["pdf_id"])
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    update_search_index(manifest)

def update_search_index(manifest: dict) -> None:
    """Builds a comprehensive search_index.json for rapid library searching."""
    pdf_id = manifest["pdf_id"]
    search_data = {
        "pdf_id": pdf_id,
        "filename": manifest.get("filename", ""),
        "page_count": manifest.get("page_count", 0),
        "images": manifest.get("images", []),
        "tables": []
    }
    
    for tbl in manifest.get("tables", []):
        tbl_data = dict(tbl)
        csv_fname = tbl.get("vlm_enhanced_filename") or tbl.get("filename")
        if csv_fname:
            csv_p = table_path(pdf_id, csv_fname)
            if csv_p.exists():
                try:
                    df = pd.read_csv(csv_p)
                    # Convert DataFrame to list of dicts, replacing NaNs with empty string
                    tbl_data["data"] = df.fillna("").to_dict(orient="records")
                except Exception as e:
                    logger.warning(f"[update_search_index] Failed to read {csv_p}: {e}")
                    tbl_data["data"] = []
        search_data["tables"].append(tbl_data)
        
    index_path = extraction_dir(pdf_id) / "search_index.json"
    index_path.write_text(json.dumps(search_data, ensure_ascii=False), encoding="utf-8")
    logger.debug(f"[update_search_index] Updated index for {pdf_id}")


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------

def register_pdf(pdf_bytes: bytes, filename: str, page_count: int = 0) -> dict:
    """
    Create (or load) the per-PDF directory and manifest.
    Always writes source.pdf and an up-to-date manifest skeleton.
    Existing image/table entries are preserved if the manifest already exists.
    """
    pdf_id = compute_pdf_id(pdf_bytes, filename)
    stem = _safe_stem(filename)
    md5 = hashlib.md5(pdf_bytes).hexdigest()

    extraction_dir(pdf_id)  # ensure dirs exist

    # Write source.pdf if missing
    sp = source_pdf_path(pdf_id)
    if not sp.exists():
        sp.write_bytes(pdf_bytes)
        logger.info(f"[register_pdf] Wrote source PDF ({len(pdf_bytes) // 1024} KB) → {sp}")

    manifest = load_manifest(pdf_id) or _new_manifest(pdf_id, filename, stem, md5)
    manifest["filename"] = filename
    manifest["stem"] = stem
    manifest["md5"] = md5
    if page_count:
        manifest["page_count"] = page_count
    save_manifest(manifest)
    logger.info(f"[register_pdf] Registered '{filename}' as {pdf_id} ({page_count} pages)")
    return manifest


def save_image(
    pdf_id: str,
    png_bytes: bytes,
    idx: int,
    page: int,
    caption: str = "",
) -> str:
    """Persist a figure PNG and update manifest. Returns relative filename."""
    manifest = load_manifest(pdf_id)
    if manifest is None:
        raise RuntimeError(f"No manifest for pdf_id={pdf_id}; call register_pdf first.")

    fname = image_filename(manifest["stem"], idx, page)
    out_path = extraction_dir(pdf_id) / "images" / fname
    out_path.write_bytes(png_bytes)
    logger.info(f"[save_image] {fname} ({len(png_bytes) // 1024} KB) → {out_path}")

    # Upsert by filename
    manifest["images"] = [e for e in manifest["images"] if e["filename"] != fname]
    manifest["images"].append({
        "filename": fname,
        "page": page,
        "idx": idx,
        "caption": caption or "",
    })
    manifest["images"].sort(key=lambda e: (e["page"], e["idx"]))
    save_manifest(manifest)
    return fname


def save_table(
    pdf_id: str,
    df: pd.DataFrame,
    idx: int,
    page: int,
    section_header: str = "",
    nearest_text: str = "",
    bbox: list[float] | None = None,
) -> str:
    """Persist a table CSV and update manifest. Returns relative filename."""
    manifest = load_manifest(pdf_id)
    if manifest is None:
        raise RuntimeError(f"No manifest for pdf_id={pdf_id}; call register_pdf first.")

    fname = table_filename(manifest["stem"], idx, page)
    out_path = extraction_dir(pdf_id) / "tables" / fname
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[save_table] {fname} ({df.shape[0]}×{df.shape[1]}) → {out_path}")

    manifest["tables"] = [e for e in manifest["tables"] if e["filename"] != fname]
    manifest["tables"].append({
        "filename": fname,
        "page": page,
        "idx": idx,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "section_header": section_header or "",
        "nearest_text": nearest_text or "",
        "bbox": bbox,
    })
    manifest["tables"].sort(key=lambda e: (e["page"], e["idx"]))
    save_manifest(manifest)
    return fname


def save_image_vlm_description(pdf_id: str, filename: str, description: str) -> None:
    """Save generated VLM caption/description for an image and update the manifest."""
    manifest = load_manifest(pdf_id)
    if manifest is None:
        logger.warning(f"[save_image_vlm_description] No manifest for {pdf_id}, skipping.")
        return

    # Update in manifest
    updated = False
    for entry in manifest.get("images", []):
        if entry["filename"] == filename:
            entry["vlm_description"] = description
            updated = True
            break

    if updated:
        save_manifest(manifest)

    # Save as text file in summaries/ directory
    txt_name = f"desc_{Path(filename).stem}.txt"
    out_path = extraction_dir(pdf_id) / "summaries" / txt_name
    out_path.write_text(description, encoding="utf-8")
    logger.info(f"[save_image_vlm_description] {filename} → {out_path} ({len(description)} chars)")


def save_table_vlm_summary(pdf_id: str, filename: str, summary: str) -> None:
    """Save generated VLM summary/explanation for a table and update the manifest."""
    manifest = load_manifest(pdf_id)
    if manifest is None:
        logger.warning(f"[save_table_vlm_summary] No manifest for {pdf_id}, skipping.")
        return

    # Update in manifest
    updated = False
    for entry in manifest.get("tables", []):
        if entry["filename"] == filename:
            entry["vlm_summary"] = summary
            updated = True
            break

    if updated:
        save_manifest(manifest)

    # Save as text file in summaries/ directory
    txt_name = f"desc_{Path(filename).stem}.txt"
    out_path = extraction_dir(pdf_id) / "summaries" / txt_name
    out_path.write_text(summary, encoding="utf-8")
    logger.info(f"[save_table_vlm_summary] {filename} → {out_path} ({len(summary)} chars)")


def save_table_vlm_enhanced(pdf_id: str, filename: str, enhanced_df: pd.DataFrame) -> str:
    """Save the VLM-enhanced table CSV, update the manifest, and return its filename."""
    manifest = load_manifest(pdf_id)
    if manifest is None:
        raise RuntimeError(f"No manifest for pdf_id={pdf_id}")

    stem = Path(filename).stem
    enhanced_fname = f"{stem}_enhanced.csv"
    out_path = extraction_dir(pdf_id) / "tables" / enhanced_fname
    enhanced_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"[save_table_vlm_enhanced] {enhanced_fname} ({enhanced_df.shape[0]}×{enhanced_df.shape[1]}) → {out_path}"
    )

    # Update in manifest
    updated = False
    for entry in manifest.get("tables", []):
        if entry["filename"] == filename:
            entry["vlm_enhanced_filename"] = enhanced_fname
            updated = True
            break

    if updated:
        save_manifest(manifest)

    return enhanced_fname


# ---------------------------------------------------------------------------
# Listing / loading
# ---------------------------------------------------------------------------


def list_pdfs() -> list[dict]:
    """Return all manifests sorted by upload time (newest first)."""
    if not EXTRACTIONS_ROOT.exists():
        return []
    out = []
    for child in EXTRACTIONS_ROOT.iterdir():
        if not child.is_dir():
            continue
        m = load_manifest(child.name)
        if m:
            out.append(m)
    out.sort(key=lambda m: m.get("uploaded_at", ""), reverse=True)
    return out


def image_path(pdf_id: str, filename: str) -> Path:
    return extraction_dir(pdf_id) / "images" / filename


def table_path(pdf_id: str, filename: str) -> Path:
    return extraction_dir(pdf_id) / "tables" / filename


def delete_pdf(pdf_id: str) -> None:
    """Remove an entire extraction directory."""
    import shutil
    d = EXTRACTIONS_ROOT / pdf_id
    if d.exists():
        shutil.rmtree(d)
        logger.info(f"[delete_pdf] Removed extraction directory: {d}")
    else:
        logger.warning(f"[delete_pdf] Directory not found: {d}")
