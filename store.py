from __future__ import annotations

import io
import re
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


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
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_manifest(manifest: dict) -> None:
    p = manifest_path(manifest["pdf_id"])
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


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

    manifest = load_manifest(pdf_id) or _new_manifest(pdf_id, filename, stem, md5)
    manifest["filename"] = filename
    manifest["stem"] = stem
    manifest["md5"] = md5
    if page_count:
        manifest["page_count"] = page_count
    save_manifest(manifest)
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
) -> str:
    """Persist a table CSV and update manifest. Returns relative filename."""
    manifest = load_manifest(pdf_id)
    if manifest is None:
        raise RuntimeError(f"No manifest for pdf_id={pdf_id}; call register_pdf first.")

    fname = table_filename(manifest["stem"], idx, page)
    out_path = extraction_dir(pdf_id) / "tables" / fname
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    manifest["tables"] = [e for e in manifest["tables"] if e["filename"] != fname]
    manifest["tables"].append({
        "filename": fname,
        "page": page,
        "idx": idx,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "section_header": section_header or "",
        "nearest_text": nearest_text or "",
    })
    manifest["tables"].sort(key=lambda e: (e["page"], e["idx"]))
    save_manifest(manifest)
    return fname


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
