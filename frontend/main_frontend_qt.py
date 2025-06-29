import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel, QFileDialog,
    QGraphicsView, QGraphicsScene, QSplitter, QInputDialog, QDialog, QTableWidget, QTableWidgetItem, QMessageBox, QToolButton, QStyle, QSpinBox, QListWidgetItem, QMenuBar, QAction, QComboBox, QLineEdit, QDialogButtonBox, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIcon, QDrag, QKeySequence
from PyQt5.QtCore import Qt, QRectF, QPointF, QMimeData, QSize, QThread, pyqtSignal, QObject
from PIL import Image
import numpy as np
import extractor_backend as backend
import ctypes
from region_selector import Region
from table_preview_dialog import show_table_preview_dialog
from table_extraction_worker import TableExtractionWorker
from utils import pil2qimage, load_white_svg_icon, region_to_dict, dict_to_region

# Portable assets directory
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')

class PDFRegionSelectorQt(QMainWindow):
    REGION_COLORS = [
        '#e57373', '#64b5f6', '#81c784', '#ffd54f', '#ba68c8', '#4dd0e1', '#ffb74d', '#a1887f', '#90a4ae', '#f06292',
        '#9575cd', '#4caf50', '#fbc02d', '#0097a7', '#c62828', '#ad1457', '#6a1b9a', '#283593', '#0277bd', '#2e7d32',
        '#f9a825', '#ff8f00', '#6d4c41', '#455a64'
    ]
    def __init__(self):
        super().__init__()
        # Enable dark title bar on Windows 10/11
        self._set_dark_title_bar()
        self.setWindowTitle('OCRaft')
        self.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, 'image.webp')))
        self.resize(1480, 720)
        self.pdf_images = []
        self.pdf_path = None  # Store the opened PDF path
        self.page_num = 0
        self.regions = []
        self.current_region = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self._zoom = 1.0
        self._pan = False
        self._pan_start = None
        self.editing_region_idx = None
        self.editing_handle = None
        self.handle_size = 10
        self.selected_ocr_model = 'Typhoon OCR'
        self.rolmocr_url = 'http://localhost:8000/v1'  # Default URL
        # Default preprocessing settings
        self.preprocess_settings = {
            'denoise_h': 15,
            'binarization': 'Otsu',  # Otsu, Adaptive, None
            'kernel_w': 1,
            'kernel_h': 2,
            'border_cleanup': True,
            'border_thickness': 10
        }
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_settings_menu()
        # Show OCR device in status bar
        try:
            import extractor_backend as backend
            device = backend.get_ocr_device()
            self.statusBar().showMessage(f"Typhoon OCR device: {device.upper()}")
        except Exception:
            self.statusBar().showMessage("Typhoon OCR device: Unknown")

    def _set_dark_title_bar(self):
        # Windows 10/11 dark mode title bar
        if sys.platform == 'win32':
            try:
                hwnd = int(self.winId())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20  # Windows 10 1809+
                value = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value)
                )
            except Exception:
                pass

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout()  # Top-level vertical layout
        main_layout = QHBoxLayout()  # Horizontal layout for main content
        # Left: Thumbnails
        self.thumb_list = QListWidget()
        self.thumb_list.setFixedWidth(140)
        self.thumb_list.currentRowChanged.connect(self.goto_page)
        # Center: Canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMouseTracking(True)
        self.view.mousePressEvent = self.on_canvas_press
        self.view.mouseMoveEvent = self.on_canvas_move
        self.view.mouseReleaseEvent = self.on_canvas_release
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.viewport().installEventFilter(self)
        # Right: Controls
        right_panel = QWidget()
        right_panel.setObjectName("right_panel")
        right_layout = QVBoxLayout(right_panel)
        btn_open = QPushButton('Open PDF')
        btn_open.clicked.connect(self.open_pdf)
        # OCR Model Dropdown
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.addItems(['Typhoon OCR', 'RolmOCR (Advanced)'])
        self.ocr_model_combo.currentTextChanged.connect(self.on_ocr_model_changed)
        self.ocr_model_combo.setToolTip('Select the OCR engine.\nTyphoon OCR: Fast, accurate, supports language hints.\nRolmOCR: Advanced, requires local server.')
        # Language Dropdown
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            'English', 'German', 'French', 'Spanish', 'Italian', 'Dutch', 'Polish', 'Portuguese', 'Russian', 'Chinese', 'Japanese', 'Korean', 'Other...'
        ])
        self.language_combo.setCurrentText('English')
        self.language_combo.setToolTip('Select document language for OCR.\nImproves accuracy for Typhoon OCR. Not used by RolmOCR.')
        btn_extract = QPushButton('Extract Tables')
        btn_extract.clicked.connect(self.extract_tables_multi)
        btn_extract.setToolTip('Run OCR and extract tables from selected regions.\nImages will be pre-processed automatically.')
        btn_clear = QPushButton('Clear Regions')
        btn_clear.clicked.connect(self.clear_all_regions)
        btn_clear.setToolTip('Remove all region selections.')
        self.region_list = QListWidget()
        self.region_list.setFixedHeight(120)
        self.region_list.itemDoubleClicked.connect(self.rename_region_dialog)
        self.region_list.setDragDropMode(QListWidget.InternalMove)
        self.region_list.setDefaultDropAction(Qt.MoveAction)
        self.region_list.model().rowsMoved.connect(self.on_region_reordered)
        # Minimal horizontal icon button bar with SVG icons
        btn_edit = QToolButton()
        btn_edit.setIcon(load_white_svg_icon(os.path.join(ASSETS_DIR, 'edit-icon.svg'), 32))
        btn_edit.setToolTip('Edit selected region')
        btn_edit.setFixedSize(32, 32)
        btn_edit.setIconSize(QSize(32, 32))
        btn_edit.clicked.connect(self.start_edit_region)
        btn_remove = QToolButton()
        btn_remove.setIcon(load_white_svg_icon(os.path.join(ASSETS_DIR, 'remove-icon.svg'), 32))
        btn_remove.setToolTip('Remove selected region')
        btn_remove.setFixedSize(32, 32)
        btn_remove.setIconSize(QSize(32, 32))
        btn_remove.clicked.connect(self.remove_region)
        btn_preview = QToolButton()
        btn_preview.setIcon(load_white_svg_icon(os.path.join(ASSETS_DIR, 'preview-icon.svg'), 32))
        btn_preview.setToolTip('Preview selected region')
        btn_preview.setFixedSize(32, 32)
        btn_preview.setIconSize(QSize(32, 32))
        btn_preview.clicked.connect(self.preview_region)
        btn_debug = QToolButton()
        debug_icon_path = os.path.join(ASSETS_DIR, 'debug-console-svgrepo-com.svg')
        btn_debug.setIcon(load_white_svg_icon(debug_icon_path, 32))
        btn_debug.setToolTip('Debug selected region OCR')
        btn_debug.setFixedSize(32, 32)
        btn_debug.setIconSize(QSize(32, 32))
        btn_debug.clicked.connect(self.debug_region_ocr)
        btn_preview_anchor = QToolButton()
        btn_preview_anchor.setIcon(load_white_svg_icon(os.path.join(ASSETS_DIR, 'preview-svgrepo-com.svg'), 32))
        btn_preview_anchor.setToolTip('Preview anchor text for selected region')
        btn_preview_anchor.setFixedSize(32, 32)
        btn_preview_anchor.setIconSize(QSize(32, 32))
        btn_preview_anchor.clicked.connect(self.preview_region_anchor_text)
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(8)
        btn_bar.addWidget(btn_edit)
        btn_bar.addWidget(btn_remove)
        btn_bar.addWidget(btn_preview)
        btn_bar.addWidget(btn_debug)
        btn_bar.addWidget(btn_preview_anchor)
        btn_bar_widget = QWidget()
        btn_bar_widget.setLayout(btn_bar)
        # Add widgets to right panel
        right_layout.addWidget(btn_open)
        right_layout.addWidget(self.ocr_model_combo)
        right_layout.addWidget(self.language_combo)
        right_layout.addWidget(btn_extract)
        right_layout.addWidget(btn_clear)
        right_layout.addWidget(QLabel('Regions:'))
        right_layout.addWidget(self.region_list)
        right_layout.addWidget(btn_bar_widget)
        right_layout.addStretch(1)
        # Layout
        main_layout.addWidget(self.thumb_list)
        main_layout.addWidget(self.view, stretch=1)
        main_layout.addWidget(right_panel)
        # Bottom navigation bar
        nav_bar = QHBoxLayout()
        nav_bar.setSpacing(12)
        nav_bar.setContentsMargins(12, 6, 12, 6)
        nav_bar.addStretch(1)  # Add stretch before controls to center
        self.btn_prev = QToolButton()
        self.btn_prev.setText('◀')
        self.btn_prev.setFixedSize(32, 32)
        self.btn_prev.clicked.connect(self.goto_prev_page)
        self.btn_next = QToolButton()
        self.btn_next.setText('▶')
        self.btn_next.setFixedSize(32, 32)
        self.btn_next.clicked.connect(self.goto_next_page)
        self.page_label = QLabel('Page 1 / 1')
        self.page_label.setFixedWidth(90)
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(1)
        self.page_spin.setValue(1)
        self.page_spin.setFixedWidth(60)
        self.page_spin.valueChanged.connect(self.goto_page_spin)
        nav_bar.addWidget(self.btn_prev)
        nav_bar.addWidget(self.btn_next)
        nav_bar.addWidget(self.page_label)
        nav_bar.addWidget(QLabel('Go to:'))
        nav_bar.addWidget(self.page_spin)
        nav_bar.addStretch(1)  # Add stretch after controls to center
        # Add layouts to vbox
        vbox.addLayout(main_layout)
        vbox.addLayout(nav_bar)
        main_widget.setLayout(vbox)

    def _setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        save_action = QAction('Save Regions', self)
        save_action.triggered.connect(self.save_regions_dialog)
        load_action = QAction('Load Regions', self)
        load_action.triggered.connect(self.load_regions_dialog)
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)

    def _setup_settings_menu(self):
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Settings')
        rolmocr_url_action = QAction('Set RolmOCR Server URL', self)
        rolmocr_url_action.triggered.connect(self.set_rolmocr_url_dialog)
        settings_menu.addAction(rolmocr_url_action)
        # Add Preprocessing Settings
        preprocess_action = QAction('Preprocessing Settings', self)
        preprocess_action.triggered.connect(self.show_preprocessing_settings_dialog)
        settings_menu.addAction(preprocess_action)

    def set_rolmocr_url_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Set RolmOCR Server URL')
        layout = QVBoxLayout(dlg)
        label = QLabel('RolmOCR Server URL:')
        url_edit = QLineEdit(self.rolmocr_url)
        layout.addWidget(label)
        layout.addWidget(url_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        def accept():
            self.rolmocr_url = url_edit.text().strip()
            dlg.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dlg.reject)
        dlg.setLayout(layout)
        dlg.exec_()

    def show_preprocessing_settings_dialog(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QCheckBox, QSpinBox, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle('Preprocessing Settings')
        layout = QVBoxLayout(dlg)
        # Denoising
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel('Denoising strength (h):'))
        denoise_slider = QSlider(Qt.Horizontal)
        denoise_slider.setMinimum(0)
        denoise_slider.setMaximum(50)
        denoise_slider.setValue(self.preprocess_settings['denoise_h'])
        hbox1.addWidget(denoise_slider)
        layout.addLayout(hbox1)
        # Binarization
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel('Binarization:'))
        bin_combo = QComboBox()
        bin_combo.addItems(['Otsu', 'Adaptive', 'None'])
        bin_combo.setCurrentText(self.preprocess_settings['binarization'])
        hbox2.addWidget(bin_combo)
        layout.addLayout(hbox2)
        # Kernel size
        hbox3 = QHBoxLayout()
        hbox3.addWidget(QLabel('Morph. kernel width:'))
        kernel_w_spin = QSpinBox()
        kernel_w_spin.setMinimum(1)
        kernel_w_spin.setMaximum(10)
        kernel_w_spin.setValue(self.preprocess_settings['kernel_w'])
        hbox3.addWidget(kernel_w_spin)
        hbox3.addWidget(QLabel('height:'))
        kernel_h_spin = QSpinBox()
        kernel_h_spin.setMinimum(1)
        kernel_h_spin.setMaximum(10)
        kernel_h_spin.setValue(self.preprocess_settings['kernel_h'])
        hbox3.addWidget(kernel_h_spin)
        layout.addLayout(hbox3)
        # Border cleanup
        border_check = QCheckBox('Remove border artifacts')
        border_check.setChecked(self.preprocess_settings['border_cleanup'])
        layout.addWidget(border_check)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(QLabel('Border thickness:'))
        border_spin = QSpinBox()
        border_spin.setMinimum(0)
        border_spin.setMaximum(50)
        border_spin.setValue(self.preprocess_settings['border_thickness'])
        hbox4.addWidget(border_spin)
        layout.addLayout(hbox4)
        # Buttons
        btn_box = QHBoxLayout()
        ok_btn = QPushButton('OK')
        cancel_btn = QPushButton('Cancel')
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        def accept():
            self.preprocess_settings['denoise_h'] = denoise_slider.value()
            self.preprocess_settings['binarization'] = bin_combo.currentText()
            self.preprocess_settings['kernel_w'] = kernel_w_spin.value()
            self.preprocess_settings['kernel_h'] = kernel_h_spin.value()
            self.preprocess_settings['border_cleanup'] = border_check.isChecked()
            self.preprocess_settings['border_thickness'] = border_spin.value()
            dlg.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.setLayout(layout)
        dlg.exec_()

    def update_nav_bar(self):
        total = len(self.pdf_images) if self.pdf_images else 1
        self.page_label.setText(f'Page {self.page_num+1} / {total}')
        self.page_spin.setMaximum(total)
        self.page_spin.setValue(self.page_num+1)
        self.btn_prev.setEnabled(self.page_num > 0)
        self.btn_next.setEnabled(self.page_num < total-1)

    def goto_prev_page(self):
        if self.page_num > 0:
            self.page_num -= 1
            self.show_page()
            self.thumb_list.setCurrentRow(self.page_num)
            self.update_nav_bar()

    def goto_next_page(self):
        if self.pdf_images and self.page_num < len(self.pdf_images)-1:
            self.page_num += 1
            self.show_page()
            self.thumb_list.setCurrentRow(self.page_num)
            self.update_nav_bar()

    def goto_page_spin(self, val):
        idx = val - 1
        if 0 <= idx < len(self.pdf_images):
            self.page_num = idx
            self.show_page()
            self.thumb_list.setCurrentRow(self.page_num)
            self.update_nav_bar()

    def select_page_dialog(self):
        if not self.pdf_images:
            return
        total = len(self.pdf_images)
        num, ok = QInputDialog.getInt(self, 'Select Page', f'Enter page number (1-{total}):', value=self.page_num+1, min=1, max=total)
        if ok:
            self.page_num = num - 1
            self.show_page()
            self.thumb_list.setCurrentRow(self.page_num)
            self.update_nav_bar()

    def show_page(self):
        self.scene.clear()
        if not self.pdf_images:
            self.update_nav_bar()
            return
        img = self.pdf_images[self.page_num]
        qimg = pil2qimage(img)
        pix = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self.view.setSceneRect(QRectF(pix.rect()))
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0
        self.draw_regions()
        self.update_nav_bar()

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open PDF', '', 'PDF Files (*.pdf)')
        if not path:
            return
        self.pdf_images = backend.load_pdf_images(path)
        self.pdf_path = path  # Store for anchor text preview
        self.page_num = 0
        self.render_thumbnails()
        self.show_page()
        self.clear_all_regions()
        self.update_nav_bar()

    def render_thumbnails(self):
        self.thumb_list.clear()
        for i, img in enumerate(self.pdf_images):
            qimg = pil2qimage(img.resize((60, 80)))
            pix = QPixmap.fromImage(qimg)
            self.thumb_list.addItem(f'Page {i+1}')
            self.thumb_list.setIconSize(pix.size())
            self.thumb_list.item(i).setIcon(QIcon(pix))
        self.thumb_list.setCurrentRow(self.page_num)
        self.update_nav_bar()

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            if event.type() == event.Wheel:
                self.handle_wheel_event(event)
                return True
            elif event.type() == event.MouseButtonPress:
                if event.button() == Qt.MiddleButton:
                    self._pan = True
                    self._pan_start = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
                    return True
            elif event.type() == event.MouseMove:
                if self._pan and self._pan_start is not None:
                    delta = event.pos() - self._pan_start
                    self._pan_start = event.pos()
                    self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() - delta.x())
                    self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() - delta.y())
                    return True
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.MiddleButton:
                    self._pan = False
                    self.setCursor(Qt.ArrowCursor)
                    return True
        return super().eventFilter(source, event)

    def handle_wheel_event(self, event):
        # Zoom in/out
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self._zoom *= zoom_factor
        self.view.scale(zoom_factor, zoom_factor)

    def show_page(self):
        self.scene.clear()
        if not self.pdf_images:
            return
        img = self.pdf_images[self.page_num]
        qimg = pil2qimage(img)
        pix = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self.view.setSceneRect(QRectF(pix.rect()))
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0
        self.draw_regions()
        self.update_nav_bar()

    def goto_page(self, idx):
        if 0 <= idx < len(self.pdf_images):
            self.page_num = idx
            self.show_page()
            self.update_nav_bar()

    def draw_regions(self):
        img = self.pdf_images[self.page_num]
        w, h = img.width, img.height
        for i, reg in enumerate(self.regions):
            if reg.page_range[0] <= self.page_num+1 <= reg.page_range[1]:
                x1, y1, x2, y2 = reg.norm_coords
                pen = QPen(QColor(reg.color), 2)
                self.scene.addRect(x1*w, y1*h, (x2-x1)*w, (y2-y1)*h, pen=pen)
                # Draw handles if editing
                if i == self.editing_region_idx:
                    self.draw_handles(x1*w, y1*h, x2*w, y2*h)

    def draw_handles(self, x1, y1, x2, y2):
        s = self.handle_size
        for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            self.scene.addRect(cx-s//2, cy-s//2, s, s, QPen(Qt.black), QColor('#fff59d'))

    def draw_temp_rect(self):
        if self.start_point and self.end_point:
            img = self.pdf_images[self.page_num]
            w, h = img.width, img.height
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            pen = QPen(QColor('#1976d2'), 2, Qt.DashLine)
            rect = QRectF(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
            self.scene.addRect(rect, pen)

    def on_canvas_press(self, event):
        if not self.pdf_images:
            return
        pos = self.view.mapToScene(event.pos())
        if self.editing_region_idx is not None:
            # Check if a handle is pressed
            reg = self.regions[self.editing_region_idx]
            img = self.pdf_images[self.page_num]
            w, h = img.width, img.height
            x1, y1, x2, y2 = reg.norm_coords
            handles = [(x1*w, y1*h), (x2*w, y1*h), (x2*w, y2*h), (x1*w, y2*h)]
            for idx, (hx, hy) in enumerate(handles):
                rect = QRectF(hx-self.handle_size, hy-self.handle_size, 2*self.handle_size, 2*self.handle_size)
                if rect.contains(pos):
                    self.editing_handle = idx
                    return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = pos
            self.end_point = self.start_point

    def on_canvas_move(self, event):
        pos = self.view.mapToScene(event.pos())
        if self.drawing:
            self.end_point = pos
            self.scene.clear()
            img = self.pdf_images[self.page_num]
            qimg = pil2qimage(img)
            pix = QPixmap.fromImage(qimg)
            self.scene.addPixmap(pix)
            self.draw_regions()
            self.draw_temp_rect()
        elif self.editing_region_idx is not None and self.editing_handle is not None:
            reg = self.regions[self.editing_region_idx]
            img = self.pdf_images[self.page_num]
            w, h = img.width, img.height
            x1, y1, x2, y2 = reg.norm_coords
            px, py = pos.x()/w, pos.y()/h
            coords = [x1, y1, x2, y2]
            if self.editing_handle == 0:
                coords[0], coords[1] = px, py
            elif self.editing_handle == 1:
                coords[2], coords[1] = px, py
            elif self.editing_handle == 2:
                coords[2], coords[3] = px, py
            elif self.editing_handle == 3:
                coords[0], coords[3] = px, py
            # Ensure valid rectangle
            nx1, ny1, nx2, ny2 = min(coords[0], coords[2]), min(coords[1], coords[3]), max(coords[0], coords[2]), max(coords[1], coords[3])
            reg.norm_coords = (nx1, ny1, nx2, ny2)
            self.scene.clear()
            qimg = pil2qimage(img)
            pix = QPixmap.fromImage(qimg)
            self.scene.addPixmap(pix)
            self.draw_regions()

    def on_canvas_release(self, event):
        if self.drawing:
            self.drawing = False
            self.end_point = self.view.mapToScene(event.pos())
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            img = self.pdf_images[self.page_num]
            w, h = img.width, img.height
            # Normalize coordinates
            nx1, ny1 = min(x1, x2) / w, min(y1, y2) / h
            nx2, ny2 = max(x1, x2) / w, max(y1, y2) / h
            pr, ok = QInputDialog.getText(self, 'Page Range', 'Enter page range (e.g. 2-6):', text=f'{self.page_num+1}-{self.page_num+1}')
            if ok and pr:
                try:
                    start, end = map(int, pr.split('-'))
                    color = self.REGION_COLORS[len(self.regions) % len(self.REGION_COLORS)]
                    label = f'Region {len(self.regions)+1}'
                    region = Region((nx1, ny1, nx2, ny2), (start, end), label, color)
                    self.regions.append(region)
                    self.region_list.addItem(f'{label} (p{start}-{end})')
                except Exception:
                    QMessageBox.warning(self, 'Invalid Range', 'Please enter a valid page range like 2-6.')
            self.show_page()
        elif self.editing_region_idx is not None and self.editing_handle is not None:
            self.editing_handle = None
            self.show_page()

    def clear_all_regions(self):
        self.regions.clear()
        self.region_list.clear()
        self.show_page()

    def rename_region_dialog(self, item):
        idx = self.region_list.row(item)
        reg = self.regions[idx]
        new_label, ok = QInputDialog.getText(self, 'Rename Region', 'Enter new name:', text=reg.label)
        if ok and new_label.strip():
            reg.label = new_label.strip()
            self.region_list.item(idx).setText(f'{reg.label} (p{reg.page_range[0]}-{reg.page_range[1]})')

    def prompt_api_key_gui(self):
        key, ok = QInputDialog.getText(self, 'Typhoon OCR API Key', 'Enter your Typhoon OCR API key:', echo=QInputDialog.Normal)
        if ok and key.strip():
            # Save to file for future use
            keyfile = os.path.join(os.path.expanduser("~"), ".typhoon_ocr_api_key")
            with open(keyfile, "w") as f:
                f.write(key.strip())
            os.environ["TYPHOON_OCR_API_KEY"] = key.strip()
            return key.strip()
        return None

    def get_api_key_gui(self):
        key = os.environ.get("TYPHOON_OCR_API_KEY")
        keyfile = os.path.join(os.path.expanduser("~"), ".typhoon_ocr_api_key")
        if not key and os.path.exists(keyfile):
            with open(keyfile, "r") as f:
                key = f.read().strip()
        if not key:
            key = self.prompt_api_key_gui()
        os.environ["TYPHOON_OCR_API_KEY"] = key or ""
        return key

    def extract_tables_multi(self):
        if not self.pdf_images or not self.regions:
            QMessageBox.information(self, 'Info', 'Open a PDF and select at least one region.')
            return
        key = self.get_api_key_gui()
        if not key:
            QMessageBox.warning(self, 'API Key Required', 'You must enter a Typhoon OCR API key to extract tables.')
            return
        region_dicts = [dict(coords=r.norm_coords, page_range=r.page_range, label=r.label, color=r.color) for r in self.regions]
        ocr_model = self.selected_ocr_model
        selected_language = self.language_combo.currentText()
        extra_kwargs = {}
        if ocr_model == 'RolmOCR':
            extra_kwargs['rolmocr_url'] = self.rolmocr_url
        extra_kwargs['preprocess_settings'] = self.preprocess_settings
        # Progress dialog
        self.progress_dialog = QProgressDialog('Extracting tables...', 'Cancel', 0, len(region_dicts), self)
        self.progress_dialog.setWindowTitle('Extraction Progress')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        self.region_status = [None] * len(region_dicts)  # Track per-region status
        self.extract_thread = TableExtractionWorker(self.pdf_images, region_dicts, ocr_task_type='structure', parent=None, **extra_kwargs)
        self.extract_thread.ocr_model = ocr_model
        self.extract_thread.rolmocr_url = self.rolmocr_url if ocr_model == 'RolmOCR' else None
        self.extract_thread.language = selected_language
        self.extract_thread.tables_extracted.connect(self._on_tables_extracted_with_progress)
        self.extract_thread.error.connect(self._on_extraction_error_with_progress)
        self.extract_thread.start()
        self.extract_thread_finished = self.extract_thread

    def _on_tables_extracted_with_progress(self, tables_and_debug):
        # Mark all as success for now (can be improved for per-region)
        if isinstance(tables_and_debug, tuple) and len(tables_and_debug) == 2:
            tables, debug_info = tables_and_debug
        else:
            tables = tables_and_debug
            debug_info = None
        for i in range(len(tables)):
            self.region_status[i] = 'Success'
            self.progress_dialog.setValue(i+1)
        self.progress_dialog.close()
        show_table_preview_dialog(self, self.regions, tables, debug_info, region_status=self.region_status)

    def _on_extraction_error_with_progress(self, error_msg):
        # Mark all as failed
        for i in range(len(self.region_status)):
            self.region_status[i] = 'Error'
        self.progress_dialog.close()
        self.show_extraction_error(error_msg)

    def show_extraction_error(self, error_msg):
        dlg = QDialog(self)
        dlg.setWindowTitle('Extraction Error')
        dlg.resize(700, 400)
        layout = QVBoxLayout(dlg)
        label = QLabel('An error occurred during table extraction:')
        layout.addWidget(label)
        from PyQt5.QtWidgets import QPlainTextEdit, QPushButton
        text_edit = QPlainTextEdit()
        text_edit.setPlainText(error_msg)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(text_edit)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.setLayout(layout)
        dlg.exec_()

    def show_preview_window_tables(self, tables_and_debug):
        if isinstance(tables_and_debug, tuple) and len(tables_and_debug) == 2:
            tables, debug_info = tables_and_debug
        else:
            tables = tables_and_debug
            debug_info = None
        show_table_preview_dialog(self, self.regions, tables, debug_info)

    def start_edit_region(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            QMessageBox.information(self, 'Info', 'Select a region to edit.')
            return
        self.editing_region_idx = idx
        self.editing_handle = None
        self.show_page()

    def stop_edit_region(self):
        self.editing_region_idx = None
        self.editing_handle = None
        self.show_page()

    def remove_region(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            QMessageBox.information(self, 'Info', 'Select a region to remove.')
            return
        self.regions.pop(idx)
        self.region_list.takeItem(idx)
        self.show_page()

    def preview_region(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            QMessageBox.information(self, 'Info', 'Select a region to preview.')
            return
        reg = self.regions[idx]
        img = self.pdf_images[self.page_num]
        w, h = img.width, img.height
        x1, y1, x2, y2 = reg.norm_coords
        # Clamp coordinates to image bounds
        left = max(0, min(int(x1 * w), w-1))
        top = max(0, min(int(y1 * h), h-1))
        right = max(left+1, min(int(x2 * w), w))
        bottom = max(top+1, min(int(y2 * h), h))
        crop_box = (left, top, right, bottom)
        cropped = img.crop(crop_box)
        qimg = pil2qimage(cropped)
        pix = QPixmap.fromImage(qimg)
        dlg = QDialog(self)
        dlg.setWindowTitle(f'Preview: {reg.label}')
        dlg.setMinimumSize(320, 240)
        dlg.resize(min(pix.width()+40, 1200), min(pix.height()+120, 1200))
        vbox = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setScaledContents(True)
        vbox.addWidget(lbl)
        # Export button
        def export_png():
            path, _ = QFileDialog.getSaveFileName(dlg, 'Export Region as PNG', '', 'PNG Files (*.png)')
            if not path:
                return
            # Save the PIL image as PNG (hi-res)
            cropped.save(path, format='PNG')
            QMessageBox.information(dlg, 'Export', 'Region image exported as PNG.')
        btn_export = QPushButton('Export as PNG')
        btn_export.clicked.connect(export_png)
        vbox.addWidget(btn_export)
        dlg.setSizeGripEnabled(True)
        dlg.exec_()

    def preview_region_anchor_text(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            QMessageBox.information(self, 'Info', 'Select a region to preview anchor text.')
            return
        reg = self.regions[idx]
        page_num = reg.page_range[0] - 1
        pdf_path = getattr(self, 'pdf_path', None)
        if not pdf_path:
            QMessageBox.warning(self, 'No PDF', 'PDF path not available. Please open a PDF file.')
            return
        try:
            from extractor_backend import get_region_anchor_text
            anchor_text = get_region_anchor_text(pdf_path, page_num, reg.norm_coords)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to extract anchor text:\n{e}')
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f'Anchor Text: {reg.label}')
        dlg.setMinimumSize(700, 500)
        dlg.resize(900, 600)
        vbox = QVBoxLayout(dlg)
        from PyQt5.QtWidgets import QPlainTextEdit
        text_edit = QPlainTextEdit()
        text_edit.setPlainText(anchor_text)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        vbox.addWidget(text_edit)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(dlg.accept)
        vbox.addWidget(btn_close)
        dlg.setLayout(vbox)
        dlg.setSizeGripEnabled(True)
        dlg.exec_()

    def on_region_reordered(self, parent, start, end, dest, dest_row):
        # Reorder self.regions to match QListWidget order
        new_order = []
        for i in range(self.region_list.count()):
            text = self.region_list.item(i).text()
            # Find region by label (ignoring page info)
            label = text.split(' (p')[0]
            for reg in self.regions:
                if reg.label == label:
                    new_order.append(reg)
                    break
        self.regions = new_order

    def save_regions_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Regions', '', 'JSON Files (*.json)')
        if not path:
            return
        # Convert Region objects to dicts for saving
        region_dicts = [region_to_dict(r) for r in self.regions]
        try:
            backend.save_regions(region_dicts, path)
            QMessageBox.information(self, 'Success', 'Regions saved successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save regions:\n{e}')

    def load_regions_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load Regions', '', 'JSON Files (*.json)')
        if not path:
            return
        try:
            region_dicts = backend.load_regions(path)
            self.regions = [dict_to_region(d) for d in region_dicts]
            self.update_region_list()
            self.draw_regions()
            QMessageBox.information(self, 'Success', 'Regions loaded successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load regions:\n{e}')

    def update_region_list(self):
        self.region_list.clear()
        for reg in self.regions:
            item = QListWidgetItem(reg.label)
            self.region_list.addItem(item)
            
    def show_rolmocr_instructions(self):
        from rolmocr_instructions_dialog import RolmOCRInstructionsDialog
        dlg = RolmOCRInstructionsDialog(self)
        dlg.exec_()

    def on_ocr_model_changed(self, text):
        # Accept both label and internal name
        if text.startswith('RolmOCR'):
            self.selected_ocr_model = 'RolmOCR'
            self.show_rolmocr_instructions()
        else:
            self.selected_ocr_model = text

    def debug_region_ocr(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            QMessageBox.information(self, 'Info', 'Select a region to debug.')
            return
        reg = self.regions[idx]
        img = self.pdf_images[self.page_num]
        try:
            orig_img, pre_img, raw_text = backend.debug_region_ocr(
                img,
                reg.norm_coords,
                ocr_model=self.selected_ocr_model,
                language=self.language_combo.currentText(),
                rolmocr_server_url=self.rolmocr_url if self.selected_ocr_model == 'RolmOCR' else None,
                preprocess_settings=self.preprocess_settings
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to debug region OCR:\n{e}')
            return
        # Show dialog with both images and raw OCR text
        dlg = QDialog(self)
        dlg.setWindowTitle(f'Debug OCR: {reg.label}')
        dlg.setMinimumSize(600, 400)
        dlg.resize(1200, 700)
        dlg.setSizeGripEnabled(True)
        main_layout = QVBoxLayout(dlg)
        img_hbox = QHBoxLayout()
        # Original image
        orig_label = QLabel()
        orig_label.setAlignment(Qt.AlignCenter)
        orig_qimg = pil2qimage(orig_img)
        orig_pix = QPixmap.fromImage(orig_qimg)
        orig_label.setPixmap(orig_pix)
        orig_label.setScaledContents(True)
        orig_label.setMinimumSize(200, 150)
        img_hbox.addWidget(self._make_img_box('Original Crop', orig_label))
        # Preprocessed image
        pre_label = QLabel()
        pre_label.setAlignment(Qt.AlignCenter)
        pre_qimg = pil2qimage(pre_img)
        pre_pix = QPixmap.fromImage(pre_qimg)
        pre_label.setPixmap(pre_pix)
        pre_label.setScaledContents(True)
        pre_label.setMinimumSize(200, 150)
        img_hbox.addWidget(self._make_img_box('Preprocessed', pre_label))
        main_layout.addLayout(img_hbox)
        # Raw OCR text
        from PyQt5.QtWidgets import QPlainTextEdit
        main_layout.addWidget(QLabel('Raw OCR Output:'))
        text_edit = QPlainTextEdit()
        text_edit.setPlainText(raw_text)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        main_layout.addWidget(text_edit, stretch=1)
        dlg.setLayout(main_layout)
        dlg.exec_()

    def _make_img_box(self, title, label_widget):
        box = QVBoxLayout()
        box.addWidget(QLabel(f'<b>{title}</b>'), alignment=Qt.AlignCenter)
        box.addWidget(label_widget)
        w = QWidget()
        w.setLayout(box)
        return w

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Modern dark (night) theme stylesheet
    dark_stylesheet = """
    QWidget {
        background-color: #212226;
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 14px;
    }
    QMainWindow, QDialog {
        background-color: #232629;
    }
    /* Layered look for panels */
    QWidget#right_panel, QWidget#main_widget {
        background-color: #282a2e;
        border-left: 1.5px solid #35363a;
    }
    QHBoxLayout, QVBoxLayout {
        background: transparent;
    }
    /* Toolbar and navigation bar */
    QToolBar, QToolButton, QFrame[frameShape="4"] {
        background-color: #292b2f;
        border: 1px solid #35363a;
    }
    QPushButton, QToolButton {
        background-color: #31363b;
        color: #e0e0e0;
        border: 0px solid #444;
        border-radius: 6px;
        padding: 6px 12px;
    }
    QToolButton {
        padding: 0px;
        border-radius: 5px;
        background-color: #292b2f;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #3a3f44;
        border: 1px solid #5e81ac;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #232629;
        border: 1px solid #81a1c1;
    }
    QListWidget, QTableWidget, QSpinBox, QInputDialog, QLineEdit, QLabel {
        background-color: #232629;
        color: #e0e0e0;
        border: 0px solid #444;
    }
    QListWidget {
        background-color: #23262b;
        border: 1.5px solid #35363a;
    }
    QTableWidget {
        background-color: #23262b;
        border: 1.5px solid #35363a;
    }
    QListWidget::item:selected, QTableWidget::item:selected {
        background: #5e81ac;
        color: #fff;
    }
    QScrollBar:vertical, QScrollBar:horizontal {
        background: transparent;
        width: 7px;
        height: 7px;
        margin: 0px;
    }
    QScrollBar:vertical:hover, QScrollBar:horizontal:hover,
    QScrollBar:vertical:active, QScrollBar:horizontal:active,
    QScrollBar:vertical:focus, QScrollBar:horizontal:focus {
        background: #232629;
    }
    QMenu, QMenuBar {
        background-color: #232629;
        color: #e0e0e0;
    }
    QMenuBar::item {
        background: transparent;
        color: #e0e0e0;
    }
    QMenuBar::item:selected {
        background: #5e81ac;
        color: #fff;
        /* More opaque highlight */
        border-radius: 4px;
        opacity: 1.0;
    }
    QMenuBar::item:pressed {
        background: #3b5c8c;
        color: #fff;
        /* Even more opaque on click */
        border-radius: 4px;
        opacity: 1.0;
    }
    QMenu {
        background-color: #232629;
        color: #e0e0e0;
    }
    QMenu::item:selected {
        background-color: #5e81ac;
        color: #fff;
        opacity: 1.0;
    }
    QDialog {
        background-color: #232629;
    }
    QHeaderView::section {
        background-color: #31363b;
        color: #e0e0e0;
        border: 1px solid #444;
        padding: 4px;
    }
    QSpinBox, QLineEdit {
        background-color: #31363b;
        color: #e0e0e0;
        border: 0px solid #444;
        border-radius: 4px;
        padding: 2px 6px;
    }
    QMessageBox {
        background-color: #232629;
        color: #e0e0e0;
    }
    """
    app.setStyleSheet(dark_stylesheet)
    window = PDFRegionSelectorQt()
    window.setWindowTitle('OCRaft')
    window.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, 'image.webp')))
    window.show()
    sys.exit(app.exec_())