# OCRaft Migration & Setup Guide

This guide explains how to migrate and set up the OCRaft development environment on a new PC.

---

## 1. Prepare the New PC
- Install [Python 3.8+](https://www.python.org/downloads/) (recommended: 3.10 or 3.11).
- (Optional) Install [Git](https://git-scm.com/downloads) for version control.

---

## 2. Clone Your Repository
```sh
git clone <your-github-repo-url>
cd OCRaft-main
```

---

## 3. Set Up a Virtual Environment
```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

---

## 4. Install Python Dependencies
```sh
pip install -r requirements.txt
```

---

## 5. Install Poppler for PDF Rendering
- Download Poppler for your OS:
  - [Windows](http://blog.alivate.com.au/poppler-windows/)
  - [Mac: `brew install poppler`]
  - [Linux: `sudo apt install poppler-utils`]
- Place Poppler binaries in `poppler/Library/bin` (recommended) or set the `POPLER_PATH` environment variable to the Poppler bin directory.

---

## 6. Set Up Typhoon OCR API Key
- On first run, you’ll be prompted for your Typhoon OCR API key.
- Or, set it as an environment variable:
  - Windows:
    ```sh
    set TYPHOON_OCR_API_KEY=your-key-here
    ```
  - Linux/Mac:
    ```sh
    export TYPHOON_OCR_API_KEY=your-key-here
    ```

---

## 7. Run the Application
- Use the frontend GUI, or
- Run the demo notebook:
  ```sh
  jupyter notebook Demo_Typhoon_OCR.ipynb
  ```
- Or run backend scripts directly.

---

## 8. (Optional) Configure VS Code
- Open the folder in VS Code.
- Install recommended extensions (Python, Jupyter, etc.).

---

## 9. Test the Setup
- Try extracting a table from a sample PDF or image.
- Check for any missing dependencies or errors.

---

## 10. Continue Development
- Make changes, commit, and push to GitHub as needed.

---

**You’re ready to develop on your new PC!**
