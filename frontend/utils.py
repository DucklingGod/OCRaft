import os
import numpy as np
from PyQt5.QtGui import QImage, QIcon, QPixmap, QPainter
from PyQt5.QtCore import Qt

# Utility: Convert PIL Image to QImage
def pil2qimage(img):
    if img.mode == 'RGB':
        r, g, b = img.split()
        arr = np.dstack([np.array(r), np.array(g), np.array(b)])
        h, w, ch = arr.shape
        return QImage(arr.data, w, h, 3*w, QImage.Format_RGB888)
    elif img.mode == 'L':
        arr = np.array(img)
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        return QImage()

# Utility: Load and colorize SVG icon as white
def load_white_svg_icon(path, size=32):
    from PyQt5.QtSvg import QSvgRenderer
    renderer = QSvgRenderer(path)
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pix.rect(), Qt.white)
    painter.end()
    return QIcon(pix)

# Utility: Region <-> dict conversion (for save/load)
def region_to_dict(region):
    return region.to_dict()

def dict_to_region(d):
    from region_selector import Region
    return Region.from_dict(d)
