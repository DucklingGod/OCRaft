import os
import base64
import requests
import io
from PIL import Image
import json

def get_typhoon_api_key():
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

def typhoon_ocr_manual_api_call(img, task_type="structure", language=None, max_tokens=16384, return_raw=False):
    """
    Call Typhoon OCR API manually with full control over parameters.
    img: PIL Image
    task_type: 'default' or 'structure'
    language: language code (optional)
    max_tokens: output token limit (default 16384)
    return_raw: if True, return full API response dict
    Returns: OCR text (or full response if return_raw)
    """
    api_key = get_typhoon_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with io.BytesIO() as buf:
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "images": [{"data": img_base64}],
        "task_type": task_type,
        "max_tokens": max_tokens,
    }
    if language:
        payload["language"] = language
    # Use the correct Typhoon OCR endpoint
    response = requests.post("https://api.opentyphoon.ai/v1", headers=headers, data=json.dumps(payload))
    try:
        response_json = response.json()
    except Exception as e:
        response_json = {"error": f"Failed to decode JSON: {e}", "raw_response": response.text}
    # --- Debug: print the raw API response for troubleshooting ---
    print("[Typhoon OCR API] Raw response:", json.dumps(response_json, ensure_ascii=False, indent=2))
    if return_raw:
        return response_json
    if isinstance(response_json, dict) and "text" in response_json:
        return response_json["text"]
    return response_json
