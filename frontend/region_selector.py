from PyQt5.QtGui import QColor

class Region:
    def __init__(self, norm_coords, page_range, label, color):
        self.norm_coords = norm_coords  # (x1, y1, x2, y2) normalized [0,1]
        self.page_range = page_range  # (start, end)
        self.label = label
        self.color = color

    @staticmethod
    def from_dict(d):
        return Region(tuple(d['coords']), tuple(d['page_range']), d.get('label', ''), d.get('color', '#e57373'))

    def to_dict(self):
        return {
            'coords': self.norm_coords,
            'page_range': self.page_range,
            'label': self.label,
            'color': self.color
        }
