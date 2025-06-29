from pdf2image import convert_from_path
from PIL import Image
import re
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import os
import sys
from typhoon_ocr import prepare_ocr_messages, ocr_document
from typhoon_ocr_manual import typhoon_ocr_manual_api_call
import time
import random
import json
import base64
import io
import requests
import cv2
import numpy as np
import tempfile
import subprocess
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

print("PYTHON EXECUTABLE:", sys.executable)
print("sys.path:", sys.path)

# Portable Poppler path detection
POPLER_PATH = os.environ.get('POPLER_PATH')
if not POPLER_PATH:
    # Try bundled poppler in project directory
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, '..', 'poppler', 'Library', 'bin')
    if os.path.exists(candidate):
        POPLER_PATH = os.path.abspath(candidate)
    else:
        POPLER_PATH = None

def load_pdf_images(pdf_path):
    if not POPLER_PATH:
        raise RuntimeError("Poppler not found. Set the POPLER_PATH environment variable or place Poppler in ./poppler/Library/bin.")
    return convert_from_path(pdf_path, poppler_path=POPLER_PATH)

def extract_region_images(pdf_images, regions):
    region_images = []
    for reg in regions:
        x1, y1, x2, y2 = reg['coords']
        start_page, end_page = reg['page_range']
        images = []
        for page in range(start_page-1, end_page):
            if page < 0 or page >= len(pdf_images):
                continue
            img = pdf_images[page].crop((x1, y1, x2, y2))
            images.append(img)
        region_images.append(images)
    return region_images

def get_api_key():
    """
    Get the Typhoon OCR API key from environment or prompt the user to enter it on first use.
    Stores the key in a local file for future use.
    """
    key = os.environ.get("TYPHOON_OCR_API_KEY")
    keyfile = os.path.join(os.path.expanduser("~"), ".typhoon_ocr_api_key")
    if not key and os.path.exists(keyfile):
        with open(keyfile, "r") as f:
            key = f.read().strip()
    if not key:
        print("Please enter your Typhoon OCR API key (it will be saved for future use):")
        key = input("API Key: ").strip()
        with open(keyfile, "w") as f:
            f.write(key)
    os.environ["TYPHOON_OCR_API_KEY"] = key
    return key

def ocr_image(img, ocr_task_type='structure', language=None, max_tokens=16384):
    # Use manual Typhoon OCR API call for full control
    return typhoon_ocr_manual_api_call(img, task_type=ocr_task_type, language=language, max_tokens=max_tokens)

def ocr_document_with_backoff(*args, max_retries=5, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return ocr_document(*args, **kwargs)
        except Exception as e:
            # Check for rate limit error (HTTP 429)
            if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429:
                retries += 1
                backoff_time = (2 ** retries) + random.random()
                print(f"Rate limit exceeded. Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded for Typhoon OCR API")

def ocr_image_typhoon_api(img, ocr_task_type='structure', language=None):
    # Deprecated: use ocr_image instead
    return ocr_image(img, ocr_task_type=ocr_task_type, language=language)

def ocr_image_rolmocr(img, api_url=None, api_key="123", model="reducto/RolmOCR-7b"):
    """
    Call RolmOCR via OpenAI-compatible API (vLLM server).
    img: PIL Image
    api_url: base url of vLLM server (default: http://localhost:8000/v1)
    api_key: dummy key (default: 123)
    model: model name (default: reducto/RolmOCR-7b)
    Returns: OCR text
    """
    from openai import OpenAI
    import tempfile
    if api_url is None:
        api_url = "http://localhost:8000/v1"  # Default for portability
    client = OpenAI(api_key=api_key, base_url=api_url)
    # Save image to temp file and encode as base64
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp, format="PNG")
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(tmp_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": "Return the plain text representation of this document as if you were reading it naturally.\n"},
                ],
            }
        ],
        temperature=0.2,
        max_tokens=4096
    )
    return response.choices[0].message.content

def parse_text_to_table(text):
    lines = [line for line in text.splitlines() if line.strip()]
    table = []
    for line in lines:
        if '\t' in line:
            cols = [c.strip() for c in line.split('\t')]
        elif ',' in line:
            cols = [c.strip() for c in line.split(',')]
        else:
            cols = [c.strip() for c in re.split(r'\s{2,}|\s', line) if c.strip()]
        table.append(cols)
    return table

def parse_markdown_table(md_text):
    import re
    table_lines = [line for line in md_text.splitlines() if re.match(r'^\s*\|.*\|\s*$', line)]
    table = []
    for line in table_lines:
        # Remove leading/trailing | and split by |
        cols = [c.strip() for c in line.strip('|').split('|')]
        table.append(cols)
    return table

def preprocess_image(pil_img, settings=None):
    """
    Preprocess the image for Typhoon OCR with adjustable settings.
    settings: dict with keys 'denoise_h', 'binarization', 'kernel_w', 'kernel_h', 'border_cleanup', 'border_thickness'
    """
    import cv2
    import numpy as np
    if settings is None:
        settings = {
            'denoise_h': 15,
            'binarization': 'Otsu',
            'kernel_w': 1,
            'kernel_h': 2,
            'border_cleanup': True,
            'border_thickness': 10
        }
    img = np.array(pil_img.convert('L'))
    # 1. Light denoising (edge-preserving)
    h_val = settings.get('denoise_h', 15)
    if h_val > 0:
        img = cv2.fastNlMeansDenoising(img, h=h_val)
    # 2. Binarization
    bin_method = settings.get('binarization', 'Otsu')
    if bin_method == 'Otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif bin_method == 'Adaptive':
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 11)
    else:
        binary = img
    # 3. Morphological close with adjustable kernel
    k_w = settings.get('kernel_w', 1)
    k_h = settings.get('kernel_h', 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # 4. Remove border artifacts
    if settings.get('border_cleanup', True):
        thickness = settings.get('border_thickness', 10)
        h, w = cleaned.shape
        if thickness > 0:
            cleaned[0:thickness, :] = 255
            cleaned[h-thickness:, :] = 255
            cleaned[:, 0:thickness] = 255
            cleaned[:, w-thickness:] = 255
    return Image.fromarray(cleaned)

def image_to_pdf(image_path):
    """
    Convert an image file to a single-page PDF and return the PDF path.
    """
    img = Image.open(image_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf_path = tmp.name
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(pdf_path, 'PDF')
    return pdf_path

def render_pdf_to_png(pdf_path, page_num=1, target_longest_dim=1800):
    """
    Render a PDF page to PNG at a controlled resolution and return a PIL Image.
    """
    # Get page size using pdfinfo
    cmd = ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", pdf_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdfinfo failed: {result.stderr}")
    for line in result.stdout.splitlines():
        if "MediaBox" in line:
            parts = line.split(":")[1].strip().split()
            w, h = abs(float(parts[0]) - float(parts[2])), abs(float(parts[3]) - float(parts[1]))
            break
    else:
        raise RuntimeError("MediaBox not found in pdfinfo output")
    longest_dim = max(w, h)
    dpi = int(target_longest_dim * 72 / longest_dim)
    # Render with pdftoppm
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        png_path = tmp.name
    cmd = ["pdftoppm", "-png", "-f", str(page_num), "-l", str(page_num), "-r", str(dpi), pdf_path, png_path[:-4]]
    subprocess.run(cmd, check=True)
    img = Image.open(png_path)
    return img

def prepare_image_for_ocr(input_path):
    """
    If input is an image, convert to PDF and render as PNG at high resolution.
    If input is a PDF, render first page as PNG at high resolution.
    Returns a PIL Image.
    """
    if input_path.lower().endswith('.pdf'):
        pdf_path = input_path
    else:
        pdf_path = image_to_pdf(input_path)
    img = render_pdf_to_png(pdf_path, page_num=1, target_longest_dim=1800)
    return img

def postprocess_table(table):
    """
    Clean up OCR table output:
    - Remove empty rows/columns
    - Normalize number of columns
    - Collapse consecutive empty cells
    - Merge description columns if both are mostly non-empty
    - Pad header rows
    """
    # Remove empty rows
    table = [row for row in table if any(cell.strip() for cell in row)]
    if not table:
        return table
    # Find max columns
    max_cols = max(len(row) for row in table)
    # Pad rows to max_cols
    table = [row + [''] * (max_cols - len(row)) for row in table]
    # Remove columns that are empty for all rows
    non_empty_cols = [i for i in range(max_cols) if any(row[i].strip() for row in table)]
    table = [[row[i].strip() for i in non_empty_cols] for row in table]
    # Collapse consecutive empty cells in each row
    def collapse_empty(row):
        new_row = []
        prev_empty = False
        for cell in row:
            if cell.strip() == '':
                if not prev_empty:
                    new_row.append('')
                prev_empty = True
            else:
                new_row.append(cell)
                prev_empty = False
        return new_row
    table = [collapse_empty(row) for row in table]
    # Pad header row(s) to match max columns
    max_cols = max(len(row) for row in table)
    table = [row + [''] * (max_cols - len(row)) for row in table]
    # Merge description columns if both are mostly non-empty
    if len(table) > 0 and max_cols >= 3:
        # Heuristic: if columns 1 and 2 are both non-empty in >50% of rows, merge them
        count_both = sum(1 for row in table if row[1].strip() and row[2].strip())
        if count_both > len(table) // 2:
            for row in table:
                row[1] = (row[1] + ' ' + row[2]).strip()
            table = [row[:1] + [row[1]] + row[3:] for row in table]
    return table

def extract_anchor_text(pdf_path, page_num, region_bbox):
    """
    Extracts text elements (with coordinates) from a PDF page within a region.
    region_bbox: (x1, y1, x2, y2) in normalized [0,1] coordinates
    Returns a formatted string for Typhoon prompt.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mediabox = page.rect
    w, h = mediabox.width, mediabox.height
    x1, y1, x2, y2 = region_bbox
    abs_bbox = fitz.Rect(x1 * w, y1 * h, x2 * w, y2 * h)
    text_blocks = page.get_text("blocks")
    anchor_lines = [f"Page region: {abs_bbox.x0:.1f},{abs_bbox.y0:.1f} to {abs_bbox.x1:.1f},{abs_bbox.y1:.1f}"]
    for block in text_blocks:
        bx0, by0, bx1, by1, text, *_ = block
        block_rect = fitz.Rect(bx0, by0, bx1, by1)
        if abs_bbox.intersects(block_rect) and text.strip():
            cx, cy = bx0, by0
            anchor_lines.append(f"[{int(cx)}x{int(cy)}]{text.strip().replace(chr(10), ' ')}")
    return '\n'.join(anchor_lines)

def extract_tables(pdf_images, regions, ocr_task_type='structure', ocr_model='Typhoon OCR', rolmocr_server_url=None, language=None, debug=False, preprocess_settings=None, pdf_path=None):
    """
    Extract tables using the selected OCR model.
    ocr_model: 'Typhoon OCR' or 'RolmOCR'
    rolmocr_server_url: Optional custom URL for RolmOCR server (advanced users)
    language: Optional language hint for Typhoon OCR
    debug: If True, return additional debug information
    preprocess_settings: dict of preprocessing parameters
    """
    extracted_tables = []
    for reg in regions:
        x1, y1, x2, y2 = reg['coords']
        start_page, end_page = reg['page_range']
        all_rows = []
        images = []
        anchor_texts = []
        for page in range(start_page-1, end_page):
            if page < 0 or page >= len(pdf_images):
                continue
            img_full = pdf_images[page]
            if isinstance(img_full, str):
                img_full = prepare_image_for_ocr(img_full)
            w, h = img_full.width, img_full.height
            left = max(0, min(int(x1 * w), w-1))
            top = max(0, min(int(y1 * h), h-1))
            right = max(left+1, min(int(x2 * w), w))
            bottom = max(top+1, min(int(y2 * h), h))
            crop_box = (left, top, right, bottom)
            img = img_full.crop(crop_box)
            # img = preprocess_image(img, settings=preprocess_settings)  # Bypassed to match demo behavior
            # DEBUG: Save the cropped image before sending to OCR API
            debug_img_path = f"debug_sent_to_api_page{page+1}_region_{reg.get('label','unknown')}.png"
            img.save(debug_img_path)
            images.append(img)
            # Anchor text extraction
            if pdf_path:
                anchor_text = extract_anchor_text(pdf_path, page, (x1, y1, x2, y2))
            else:
                anchor_text = None
            anchor_texts.append(anchor_text)
            # DEBUG: Save the prompt (if used)
            if anchor_text:
                prompt = f"Below is a region of a document page.\n{anchor_text}\nReturn the markdown table."
                with open(f"debug_prompt_page{page+1}_region_{reg.get('label','unknown')}.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
            if ocr_model == 'RolmOCR':
                text = ocr_image_rolmocr(img, api_url=rolmocr_server_url)
            else:
                # Use anchor text in Typhoon prompt if available
                if anchor_text:
                    prompt = f"Below is a region of a document page.\n{anchor_text}\nReturn the markdown table."
                    text = ocr_image(img, ocr_task_type=ocr_task_type, language=language)
                else:
                    text = ocr_image(img, ocr_task_type=ocr_task_type, language=language)
            # --- Robustly handle text being a dict (error or raw response) ---
            if not isinstance(text, str):
                if isinstance(text, dict) and 'text' in text:
                    text = text['text']
                else:
                    text = ''
            # If OCR output is missing, an error, or contains HTML, fallback to anchor text
            ocr_failed = (
                not text or
                text.strip().startswith('<!DOCTYPE html') or
                text.strip().startswith('<html') or
                'for sale' in text.lower() or
                'error' in text.lower()
            )
            if ocr_failed and anchor_text:
                table = extract_table_from_anchor_text(anchor_text)
            elif '<table' in text.lower():
                table = normalize_html_table(text)
            else:
                table = parse_markdown_table(text)
                if not table:
                    table = parse_text_to_table(text)
            # Fallback: if table is too short, try extracting from anchor text
            if (not table or len(table) < 5) and anchor_text:
                anchor_rows = extract_table_from_anchor_text(anchor_text)
                if anchor_rows and len(anchor_rows) > len(table):
                    table = anchor_rows
            table = postprocess_table(table)
            table = correct_item_code_ocr_errors(table)
            table = correct_table_headers(table)
            all_rows.extend(table)
        extracted_tables.append({
            'label': reg['label'],
            'table': all_rows,
            'images': images,
            'page_range': (start_page, end_page),
            'anchor_texts': anchor_texts
        })
    typhoon_input = {
        'images': [img for img in images],
        'regions': regions,
        'ocr_task_type': ocr_task_type,
        'language': language,
    }
    typhoon_output = None
    if ocr_model != 'RolmOCR':
        # Call Typhoon OCR API
        api_key = get_api_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "images": [],
            "regions": regions,
            "ocr_task_type": ocr_task_type,
            "language": language,
        }
        # DEBUG: Log API parameters and model
        with open("debug_api_call.txt", "w", encoding="utf-8") as f:
            f.write(f"Endpoint: https://api.opentyphoon.ai/v1\n")
            f.write(f"Headers: {headers}\n")
            f.write(f"Payload keys: {list(payload.keys())}\n")
            f.write(f"OCR Task Type: {ocr_task_type}\n")
            f.write(f"Language: {language}\n")
        for img in images:
            with io.BytesIO() as buf:
                img.save(buf, format="PNG")
                byte_data = buf.getvalue()
                img_base64 = base64.b64encode(byte_data).decode('utf-8')
                payload["images"].append({"data": img_base64})
        response = requests.post("https://api.opentyphoon.ai/v1", headers=headers, data=json.dumps(payload))
        try:
            response_json = response.json()
        except Exception as e:
            # Save raw text for debug, and optionally error message
            response_json = {
                'error': f'Failed to decode JSON: {e}',
                'raw_response': response.text
            }
        typhoon_output = response_json
    if debug:
        return extracted_tables, typhoon_input, typhoon_output
    return extracted_tables

def get_ocr_device():
    """
    Returns the device used for Typhoon OCR: 'cuda' (GPU) or 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def estimate_tokens(text, lang="thai"):
    if lang.lower() == "thai":
        words = len(text.split())
        return int(words * 2.5)
    else:
        words = len(text.split())
        return int(words * 1.3)

def chunk_text(text, max_tokens=7000, lang="thai"):
    words = text.split()
    chunk_size = int(max_tokens / (2.5 if lang == "thai" else 1.3))
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def check_token_limits(text, max_output_tokens=500, lang="thai"):
    input_tokens = estimate_tokens(text, lang)
    context_limit = 8192  # 8K tokens for Typhoon models
    remaining_tokens = context_limit - input_tokens
    if remaining_tokens <= 0:
        print(f"Warning: Input exceeds context window of {context_limit} tokens.")
        return False
    if remaining_tokens < max_output_tokens:
        print(f"Warning: Only {remaining_tokens} tokens remaining for output (requested {max_output_tokens}).")
        return False
    return True

def save_regions(regions, path):
    """
    Save the list of region selections to a JSON file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(regions, f, ensure_ascii=False, indent=2)

def load_regions(path):
    """
    Load the list of region selections from a JSON file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def debug_region_ocr(pil_img, norm_coords, ocr_model='Typhoon OCR', language=None, rolmocr_server_url=None, preprocess=True, preprocess_settings=None):
    """
    For a given PIL image and normalized region coordinates, return:
      - the original cropped image (PIL)
      - the preprocessed image (PIL, or same as original if preprocess=False)
      - the raw OCR output (text)
    """
    w, h = pil_img.width, pil_img.height
    x1, y1, x2, y2 = norm_coords
    left = max(0, min(int(x1 * w), w-1))
    top = max(0, min(int(y1 * h), h-1))
    right = max(left+1, min(int(x2 * w), w))
    bottom = max(top+1, min(int(y2 * h), h))
    crop_box = (left, top, right, bottom)
    cropped = pil_img.crop(crop_box)
    if preprocess:
        preprocessed = preprocess_image(cropped, settings=preprocess_settings)
    else:
        preprocessed = cropped
    if ocr_model == 'RolmOCR':
        raw_text = ocr_image_rolmocr(preprocessed, api_url=rolmocr_server_url)
    else:
        raw_text = ocr_image(preprocessed, ocr_task_type='structure', language=language)
    return cropped, preprocessed, raw_text

def get_region_anchor_text(pdf_path, page_num, region_bbox, margin=8):
    from typhoon_ocr import get_anchor_text
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mediabox = page.rect
    w, h = mediabox.width, mediabox.height
    x1, y1, x2, y2 = region_bbox
    abs_x1 = x1 * w
    abs_x2 = x2 * w
    abs_y1 = h - y1 * h
    abs_y2 = h - y2 * h
    region_top = min(abs_y1, abs_y2) - margin
    region_bottom = max(abs_y1, abs_y2) + margin
    region_left = min(abs_x1, abs_x2) - margin
    region_right = max(abs_x1, abs_x2) + margin
    full_anchor = get_anchor_text(pdf_path, page_num+1, pdf_engine="pdfreport", target_length=8000)
    filtered_lines = []
    coord_lines = []
    first_non_coord_line_added = False
    for line in full_anchor.splitlines():
        if line.startswith("[") and "]" in line:
            try:
                coord_part = line[1:line.index("]")]
                if 'x' in coord_part:
                    cx, cy = map(float, coord_part.split('x'))
                    # Include if the point is inside or near the region (intersect)
                    if (region_left <= cx <= region_right) and (region_top <= cy <= region_bottom):
                        coord_lines.append((cy, cx, line))
            except Exception:
                continue
        elif not first_non_coord_line_added and line.strip():
            filtered_lines.append(line)
            first_non_coord_line_added = True
    # Sort by y (descending), then x (ascending)
    coord_lines.sort(key=lambda t: (-t[0], t[1]))
    filtered_lines.extend([t[2] for t in coord_lines])
    return '\n'.join(filtered_lines)

def normalize_html_table(html):
    """
    Parse and flatten an HTML table (handle rowspan/colspan) into a 2D list.
    Removes empty rows/columns and normalizes headers.
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return []
    # Build grid
    grid = []
    row_spans = {}
    for row_idx, row in enumerate(table.find_all('tr')):
        cells = []
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            # Skip columns taken by rowspan
            while row_spans.get(col_idx, 0):
                cells.append(row_spans[col_idx][0])
                row_spans[col_idx][1] -= 1
                if row_spans[col_idx][1] == 0:
                    del row_spans[col_idx]
                col_idx += 1
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            text = ' '.join(cell.stripped_strings)
            for i in range(colspan):
                cells.append(text)
                # Handle rowspan for future rows
                if rowspan > 1:
                    row_spans[col_idx] = [text, rowspan - 1]
                col_idx += 1
        # Fill in any remaining row_spans
        while row_spans.get(col_idx, 0):
            cells.append(row_spans[col_idx][0])
            row_spans[col_idx][1] -= 1
            if row_spans[col_idx][1] == 0:
                del row_spans[col_idx]
            col_idx += 1
        grid.append(cells)
    # Remove empty rows
    grid = [row for row in grid if any(cell.strip() for cell in row)]
    # Normalize all rows to same length
    max_cols = max((len(row) for row in grid), default=0)
    grid = [row + [''] * (max_cols - len(row)) for row in grid]
    # Remove empty columns
    non_empty_cols = [i for i in range(max_cols) if any(row[i].strip() for row in grid)]
    grid = [[row[i] for i in non_empty_cols] for row in grid]
    return grid

def extract_table_from_anchor_text(anchor_text, min_cols=5):
    """
    Heuristically extract table rows from anchor text.
    Looks for lines starting with a number and splits columns by whitespace or known delimiters.
    min_cols: minimum number of columns to consider a valid row.
    Returns a list of rows (list of strings).
    """
    import re
    rows = []
    for line in anchor_text.splitlines():
        # Look for lines starting with a number (row number)
        if re.match(r"^\[?\d+", line.strip()):
            # Remove leading [coords] if present
            line = re.sub(r"^\[\d+x\d+\]", "", line).strip()
            # Split by two or more spaces, tab, or known delimiters
            cols = re.split(r"\s{2,}|\t|\|", line)
            # If not enough columns, try splitting by single space
            if len(cols) < min_cols:
                cols = [c for c in line.split(' ') if c.strip()]
            if len(cols) >= min_cols:
                rows.append([c.strip() for c in cols])
    return rows

def correct_item_code_ocr_errors(table, item_code_col=None):
    """
    Post-process the ITEM CODE column to fix common OCR errors (e.g., 7 vs /, G vs 6).
    If item_code_col is None, try to auto-detect the column by header name.
    """
    import re
    if not table or not table[0]:
        return table
    # Try to find the ITEM CODE column by header
    headers = [h.lower() for h in table[0]]
    if item_code_col is None:
        for idx, h in enumerate(headers):
            if 'item code' in h or 'hlm code' in h or 'code' in h:
                item_code_col = idx
                break
    if item_code_col is None:
        return table
    # Correction rules
    for i, row in enumerate(table[1:], 1):
        if len(row) > item_code_col:
            code = row[item_code_col]
            # Replace common OCR errors: G->6, S->5, O->0, etc.
            code_fixed = code
            code_fixed = re.sub(r'/43G', '/436', code_fixed)
            code_fixed = re.sub(r'/43S', '/435', code_fixed)
            code_fixed = re.sub(r'/43O', '/430', code_fixed)
            code_fixed = re.sub(r'/43D', '/439', code_fixed)
            code_fixed = re.sub(r'/104O', '/1040', code_fixed)
            # Replace any G between digits with 6
            code_fixed = re.sub(r'(?<=\d)G(?=\d)', '6', code_fixed)
            # Replace any S between digits with 5
            code_fixed = re.sub(r'(?<=\d)S(?=\d)', '5', code_fixed)
            # Replace any O between digits with 0
            code_fixed = re.sub(r'(?<=\d)O(?=\d)', '0', code_fixed)
            # Replace any D between digits with 9
            code_fixed = re.sub(r'(?<=\d)D(?=\d)', '9', code_fixed)
            # Replace any I between digits with 1
            code_fixed = re.sub(r'(?<=\d)I(?=\d)', '1', code_fixed)
            # Replace any B between digits with 8
            code_fixed = re.sub(r'(?<=\d)B(?=\d)', '8', code_fixed)
            row[item_code_col] = code_fixed
    return table

def correct_table_headers(table):
    """
    Rename misrecognized headers to their correct names.
    E.g., 'S/L/1 (N)' -> 'SIZE (IN)'.
    """
    if not table or not table[0]:
        return table
    header_map = {
        's/l/1 (n)': 'SIZE (IN)',
        's/l/1': 'SIZE (IN)',
        'hlm code': 'ITEM CODE',
        '/': '7',  # Correct common OCR error for '/'
        # Add more mappings as needed
    }
    new_header = []
    for h in table[0]:
        h_lower = h.lower().strip()
        new_header.append(header_map.get(h_lower, h))
    table[0] = new_header
    return table
