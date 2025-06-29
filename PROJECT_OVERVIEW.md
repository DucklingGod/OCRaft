# OCRaft Project Overview

OCRaft is a Python-based tool for extracting tables (such as BOMs) from PDFs and images using advanced OCR models, including Typhoon OCR and RolmOCR. It supports region selection, preprocessing, and robust post-processing to maximize extraction quality.

## Features
- Extracts tables from PDFs and images using Typhoon OCR API (manual or package)
- Supports region selection and cropping
- Preprocessing for improved OCR accuracy
- Post-processing to clean and normalize tables
- Fallback to anchor text extraction if OCR fails
- Optional support for local LLM-based OCR (RolmOCR)
- Debugging tools for prompt and image inspection
- Modular backend/frontend structure

## Main Components
- `backend/`: Core extraction, OCR, and parsing logic
- `frontend/`: GUI and user interaction
- `Demo_Typhoon_OCR.ipynb`: Example notebook for Typhoon OCR
- `requirements.txt`: Python dependencies

## Typical Workflow
1. User selects a PDF or image and defines regions of interest
2. Regions are cropped and preprocessed
3. Cropped images are sent to Typhoon OCR API (manual call for full control)
4. OCR output is parsed and post-processed into clean tables
5. Results are exported or displayed

## Supported Models
- Typhoon OCR (API)
- RolmOCR (OpenAI-compatible API)

## License
See `LICENSE` for details.
