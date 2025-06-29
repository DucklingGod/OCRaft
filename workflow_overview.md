# OCRaft Workflow Overview

This document describes the end-to-end workflow for extracting tables from PDFs/images using OCRaft.

## 1. Input
- User provides a PDF or image file.
- User selects regions of interest (e.g., BOM tables) via the frontend or notebook.

## 2. Preprocessing
- Selected regions are cropped from the input file.
- Optional image preprocessing (denoising, binarization, etc.) is applied.

## 3. OCR Extraction
- Cropped region images are sent to the Typhoon OCR API using a manual HTTP POST call for maximum control (including `max_tokens`).
- If Typhoon OCR fails or is incomplete, fallback to anchor text extraction or alternative models (e.g., RolmOCR).

## 4. Parsing and Post-processing
- OCR output is parsed into tables (supports markdown, HTML, and plain text formats).
- Post-processing cleans up tables, normalizes columns, and corrects common OCR errors.

## 5. Output
- Extracted tables are returned to the frontend or exported (e.g., to Excel).
- Debug images and prompts are saved for troubleshooting.

## 6. Debugging
- Debug logs and raw API responses are printed to the console.
- Cropped images and prompts are saved for inspection.

## 7. Extensibility
- The workflow supports easy switching between OCR models and parameter tuning.
- Modular codebase for backend/frontend separation.
