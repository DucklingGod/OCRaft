# OCRaft

A modern, modular tool for extracting tables (such as BOMs) from PDFs and images using advanced region management and Typhoon OCR (API or local model). Includes robust post-processing, debug tools, and a PyQt5 UI.

## Features
- PyQt5 dark-themed UI with zoom, pan, and multi-region selection
- Region add/edit/remove/reorder, preview, and export (PNG, CSV, Excel)
- Table preview with Excel-like editing, copy-paste, undo
- Export all regions as multi-sheet Excel or CSVs
- High-accuracy OCR using Typhoon OCR API (manual call for full control)
- Fallback to anchor text extraction if OCR fails
- Optional support for local LLM-based OCR (RolmOCR)
- Debugging tools for prompt and image inspection
- Hugging Face cache location configurable (set `HF_HOME`)

## Quick Start
1. Clone this repo and `cd OCRaft-main`
2. (Recommended) Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download and set up Poppler for PDF rendering (see MIGRATION_GUIDE.md)
5. Set your Typhoon OCR API key (will be prompted on first run, or set `TYPHOON_OCR_API_KEY`)
6. Run the app:
   ```
   python frontend/main_frontend_qt.py
   ```

## Model/API Usage
- Typhoon OCR API is used for best accuracy and full parameter control (see `backend/typhoon_ocr_manual.py`).
- For GPU acceleration, install CUDA-enabled PyTorch (see requirements.txt).
- For local LLM OCR (RolmOCR), see the development guide.

## Folder Structure
- `frontend/` — PyQt5 UI and region selection logic
- `backend/` — Extraction, OCR, and parsing logic
- `assets/` — SVG icons, app images
- `test_output/` — Example outputs
- `debug export/` — Debug images and logs

## Documentation
- `PROJECT_OVERVIEW.md` — Project summary and features
- `DEVELOPMENT_GUIDE.md` — Setup, development, and contribution guide
- `MIGRATION_GUIDE.md` — Step-by-step migration/setup for new PCs
- `workflow_overview.md` — End-to-end workflow

## For Developers
See `DEVELOPMENT_GUIDE.md` for architecture, extension points, and contribution guidelines.
