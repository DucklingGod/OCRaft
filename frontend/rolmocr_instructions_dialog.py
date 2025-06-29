from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton

class RolmOCRInstructionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('RolmOCR Setup Instructions')
        self.resize(700, 600)
        layout = QVBoxLayout(self)
        instructions = (
            "<b>How to Use RolmOCR (Advanced Users)</b><br><br>"
            "<b>1. Requirements:</b><br>"
            "- A Linux machine (local, WSL2, or cloud) with an NVIDIA GPU<br>"
            "- Python 3.9+<br>"
            "- CUDA drivers installed<br>"
            "<br>"
            "<b>2. Install vLLM and dependencies:</b><br>"
            "<pre>pip install torch --index-url https://download.pytorch.org/whl/cu121\npip install vllm openai triton</pre>"
            "<br>"
            "<b>3. Download the RolmOCR model:</b><br>"
            "- vLLM will download the model automatically on first run.<br>"
            "- Example model: <code>reducto/RolmOCR-7b</code> (from HuggingFace).<br>"
            "<br>"
            "<b>4. Start the vLLM OpenAI-compatible API server:</b><br>"
            "<pre>python -m vllm.entrypoints.openai.api_server --model reducto/RolmOCR-7b --host 0.0.0.0 --port 8000</pre>"
            "- This will serve the API at <code>http://your-server-ip:8000/v1</code><br>"
            "<br>"
            "<b>5. (Cloud only) Open firewall port 8000</b><br>"
            "- Make sure your server allows incoming connections on port 8000.<br>"
            "<br>"
            "<b>6. In OCRaft:</b><br>"
            "- Go to Settings → Set RolmOCR Server URL and enter your server URL (e.g. <code>http://localhost:8000/v1</code> or your remote IP).<br>"
            "- Select 'RolmOCR' in the OCR model dropdown.<br>"
            "- Extract tables as usual.<br>"
            "<br>"
            "<b>Troubleshooting:</b><br>"
            "- If you see connection errors, check that the vLLM server is running and accessible.<br>"
            "- vLLM does <b>not</b> work natively on Windows. Use WSL2 or a Linux server.<br>"
            "- For more help, see the vLLM documentation: <a href='https://vllm.readthedocs.io/en/latest/'>https://vllm.readthedocs.io/en/latest/</a><br>"
        )
        text = QTextBrowser()
        text.setHtml(instructions)
        text.setOpenExternalLinks(True)
        layout.addWidget(text)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
