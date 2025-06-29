# OCRaft Backend

This module contains all OCR and extraction logic for the OCRaft application.
- Main entry: `extractor_backend.py`
- Integrates Typhoon OCR (scb10x/typhoon-ocr-7b) via Hugging Face Transformers
- Device selection (GPU/CPU) is automatic
- Extend here to add new OCR models or extraction logic
