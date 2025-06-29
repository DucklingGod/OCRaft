from PyQt5.QtCore import QThread, pyqtSignal

class TableExtractionWorker(QThread):
    tables_extracted = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pdf_images, region_dicts, ocr_task_type='structure', parent=None, preprocess_settings=None, **kwargs):
        super().__init__(parent)
        self.pdf_images = pdf_images
        self.region_dicts = region_dicts
        self.ocr_task_type = ocr_task_type
        self.language = 'English'  # Default, can be set externally
        self.rolmocr_url = None
        self.debug_info = None  # Store debug info
        self.preprocess_settings = preprocess_settings or {
            'denoise_h': 15,
            'binarization': 'Otsu',
            'kernel_w': 1,
            'kernel_h': 2,
            'border_cleanup': True,
            'border_thickness': 10
        }

    def run(self):
        try:
            import extractor_backend as backend
            debug_info = {
                'region_dicts': self.region_dicts,
                'ocr_task_type': self.ocr_task_type,
                'language': self.language,
                'rolmocr_server_url': self.rolmocr_url,
                'preprocess_settings': self.preprocess_settings,
            }
            tables, typhoon_input, typhoon_output = backend.extract_tables(
                self.pdf_images,
                self.region_dicts,
                ocr_task_type=self.ocr_task_type,
                language=self.language,
                rolmocr_server_url=self.rolmocr_url,
                debug=True,
                preprocess_settings=self.preprocess_settings
            )
            debug_info['typhoon_input'] = typhoon_input
            debug_info['typhoon_output'] = typhoon_output
            self.debug_info = debug_info
            self.tables_extracted.emit((tables, debug_info))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('Extraction error:', tb)
            self.error.emit(tb if tb else str(e) or 'Unknown error occurred during extraction.')
