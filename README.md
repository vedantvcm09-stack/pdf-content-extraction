# PDF Content Extraction Pipeline

Robust PDF table and figure extraction pipeline built using **Docling**, with advanced reconstruction logic for complex real-world documents.

The system extracts:

- Tables → exported as clean CSV files
- Figures/images → exported as PNG files
- Cross-page tables → automatically merged
- Multi-line headers → reconstructed intelligently
- Rotated PDFs → auto-corrected before parsing

Includes both:

- Command-line pipeline
- Interactive Streamlit UI

---

# Features

## Table Extraction

- Context-aware table matching using semantic query scoring
- Cross-page table detection and auto-merging
- Multi-line header reconstruction
- Duplicate column resolution
- Row/column geometric alignment
- Sparse-row continuation repair
- Merged-cell repair heuristics
- PDFPlumber fallback for difficult tables
- OCR support for scanned PDFs
- Accurate TableFormer mode support

---

## Figure Extraction

- Figure detection using Docling picture items
- Query-based figure retrieval using caption/context matching
- Bulk figure export as ZIP
- Adjustable render resolution scaling

---

## Robust Pre-processing

- Automatic rotated-page detection
- Visual orientation correction
- Text-direction analysis using PyMuPDF

---

# Project Structure

```text
pdf_table_extractor/
│
├── app.py                 # Streamlit UI
├── main.py                # Core extraction pipeline
├── requirements.txt
├── README.md
│
├── Test Folder/           # Local evaluation PDFs (excluded from GitHub)
├── analysis_progress.md   # Local notes (excluded)
└── main_enhanced.py       # Experimental version (excluded)
```

---

# Installation

## 1. Clone repository

```bash
git clone <your-repo-url>
cd pdf_table_extractor
```

---

## 2. Create virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Streamlit UI

Run:

```bash
streamlit run app.py
```

Features available in UI:

- Upload PDF
- Query-based table extraction
- Query-based figure extraction
- OCR toggle
- Accurate extraction mode
- List-only table browsing
- Debug mode
- Adjustable image scale

---

# CLI Usage

## Extract best-matching table

```bash
python main.py sample.pdf "state-wise funds"
```

---

## Export to custom CSV

```bash
python main.py sample.pdf "railway expenditure" -o output.csv
```

---

## List all detected tables

```bash
python main.py sample.pdf --list
```

---

## Enable OCR

```bash
python main.py sample.pdf "population data" --ocr
```

---

## Extract figures

```bash
python main.py sample.pdf "rainfall map" --images
```

---

## Dump all figures

```bash
python main.py sample.pdf --images
```

---

## Accurate TableFormer mode

```bash
python main.py sample.pdf "budget allocation" --accurate
```

---

# How Table Matching Works

The pipeline does not rely on LLMs or embeddings.

Instead, it uses:

- Section headers
- Nearby paragraph text
- Captions
- Cell-text fallback matching
- Token overlap scoring
- Sliding-window fuzzy matching
- Sequence similarity scoring

This keeps the system:

- lightweight
- deterministic
- offline
- reproducible

---

# Cross-Page Table Merging

The pipeline automatically detects continuation tables using:

- Header similarity
- Column geometry alignment
- Bounding-box continuity
- Page-transition heuristics
- Row-density similarity
- Semantic context continuity

Fragments are merged into a single CSV automatically.

---

# Supported PDFs

Best results:

- Government reports
- Research papers
- Financial reports
- Parliamentary documents
- Structured analytical PDFs

Supports:

- Typed PDFs
- Rotated PDFs
- Moderate-complexity scanned PDFs (with OCR)

---

# Limitations

Current limitations include:

- Extremely noisy scans
- Fully image-only tables with poor OCR quality
- Highly irregular nested tables
- Very dense table layouts

---

# Technologies Used

- Docling
- Streamlit
- PyMuPDF
- PDFPlumber
- Pandas

---

# Design Philosophy

This project focuses on:

- deterministic extraction
- generic heuristics
- document-agnostic logic
- reproducibility
- minimal external dependencies

No LLMs or external APIs are used.

---

# Acknowledgements

- IBM Research — Docling
- Streamlit
- PyMuPDF
- PDFPlumber
