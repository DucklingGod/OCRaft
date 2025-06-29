# OCRaft Development Guide

This guide provides instructions for setting up, developing, and contributing to the OCRaft project.

## 1. Setup

### a. Clone the Repository
```bash
git clone <your-github-repo-url>
cd OCRaft-main
```

### b. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### c. Install Dependencies
```bash
pip install -r requirements.txt
```

### d. Download Poppler (for PDF rendering)
- Place Poppler binaries in `poppler/Library/bin` or set the `POPLER_PATH` environment variable.

## 2. Running the Application
- Use the frontend GUI or run the demo notebook (`Demo_Typhoon_OCR.ipynb`) for testing.
- For backend-only extraction, run scripts in the `backend/` directory.

## 3. Project Structure
- `backend/`: Core OCR, extraction, and parsing logic
- `frontend/`: GUI and user interaction
- `assets/`: Icons and images
- `test_output/`: Example outputs

## 4. Key Files
- `extractor_backend.py`: Main backend logic
- `typhoon_ocr_manual.py`: Manual Typhoon OCR API call logic
- `requirements.txt`: Python dependencies

## 5. Contributing
- Follow PEP8 style guidelines.
- Document new functions and modules.
- Add tests or example usage where possible.
- Submit pull requests with clear descriptions.

## 6. Troubleshooting
- If OCR results are empty, check the console for raw API responses and debug images.
- Ensure your Typhoon OCR API key is set and valid.
- For PDF issues, verify Poppler is installed and accessible.

## 7. Deployment
- Push your code to GitHub for version control and collaboration.
- You can continue development on another PC by cloning the repo and following the setup steps above.
