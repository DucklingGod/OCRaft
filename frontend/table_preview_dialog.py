from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QWidget, QLabel
)
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtCore import Qt
import csv
import pandas as pd
import tempfile
import base64
import io
import json
import os
from PIL import Image

def show_table_preview_dialog(parent, regions, tables, debug_info=None, region_status=None):
    dlg = QDialog(parent)
    dlg.setWindowTitle('Preview & Arrange Extracted Tables')
    dlg.resize(1200, 700)
    layout = QHBoxLayout(dlg)
    # Left: Region data boxes (draggable)
    region_list_widget = QListWidget()
    region_list_widget.setFixedWidth(220)
    for i, reg in enumerate(regions):
        status = region_status[i] if region_status and i < len(region_status) else 'Unknown'
        item = QListWidgetItem(f"{reg.label} [{status}]")
        item.setBackground(QColor(reg.color))
        item.setData(Qt.UserRole, reg)
        region_list_widget.addItem(item)
    region_list_widget.setDragEnabled(True)
    # Middle: Table widget (drop target)
    table_widget = QTableWidget()
    table_widget.setAcceptDrops(True)
    table_widget.setDragDropMode(QTableWidget.DropOnly)
    table_widget.setSelectionMode(QTableWidget.SingleSelection)
    table_widget.setSelectionBehavior(QTableWidget.SelectItems)
    # Fill with first table if available
    if tables and tables[0]['table']:
        table = tables[0]['table']
        maxlen = max(len(row) for row in table)
        table_widget.setRowCount(len(table))
        table_widget.setColumnCount(maxlen)
        for r, row in enumerate(table):
            for c, val in enumerate(row):
                table_widget.setItem(r, c, QTableWidgetItem(str(val)))
    else:
        table_widget.setRowCount(30)
        table_widget.setColumnCount(30)
    # Show error message if any region failed
    error_label = QLabel()
    error_label.setStyleSheet('color: #ff5555; font-weight: bold;')
    if region_status and 'Error' in region_status:
        error_label.setText('Some regions failed to extract. Please review and retry.')
    # Enable copy-paste
    def keyPressEvent(event):
        if event.matches(QKeySequence.Copy):
            selected = table_widget.selectedRanges()
            if selected:
                s = selected[0]
                data = ''
                for row in range(s.topRow(), s.bottomRow()+1):
                    row_data = []
                    for col in range(s.leftColumn(), s.rightColumn()+1):
                        item = table_widget.item(row, col)
                        row_data.append(item.text() if item else '')
                    data += '\t'.join(row_data) + '\n'
                parent.clipboard().setText(data)
        elif event.matches(QKeySequence.Paste):
            selected = table_widget.selectedRanges()
            if selected:
                s = selected[0]
                paste_data = parent.clipboard().text().splitlines()
                for i, row in enumerate(range(s.topRow(), s.bottomRow()+1)):
                    if i >= len(paste_data): break
                    cols = paste_data[i].split('\t')
                    for j, col in enumerate(range(s.leftColumn(), s.rightColumn()+1)):
                        if j >= len(cols): break
                        table_widget.setItem(row, col, QTableWidgetItem(cols[j]))
        else:
            QTableWidget.keyPressEvent(table_widget, event)
    table_widget.keyPressEvent = keyPressEvent
    # Undo support (simple stack)
    undo_stack = []
    def save_table_state():
        state = [[table_widget.item(r, c).text() if table_widget.item(r, c) else ''
                  for c in range(table_widget.columnCount())]
                 for r in range(table_widget.rowCount())]
        undo_stack.append(state)
    def restore_table_state():
        if not undo_stack:
            return
        state = undo_stack.pop()
        table_widget.setRowCount(len(state))
        table_widget.setColumnCount(len(state[0]) if state else 0)
        for r, row in enumerate(state):
            for c, val in enumerate(row):
                table_widget.setItem(r, c, QTableWidgetItem(val))
    table_widget.itemChanged.connect(save_table_state)
    # Right panel with buttons
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    btn_clear = QPushButton('Clear Table')
    btn_clear.clicked.connect(lambda: table_widget.clearContents())
    btn_undo = QPushButton('Undo')
    btn_undo.clicked.connect(restore_table_state)
    btn_fit = QPushButton('Fit to Data')
    def fit_to_data():
        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
    btn_fit.clicked.connect(fit_to_data)
    btn_export_csv = QPushButton('Export to CSV')
    def export_csv():
        path, _ = QFileDialog.getSaveFileName(dlg, 'Export CSV', '', 'CSV Files (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for r in range(table_widget.rowCount()):
                row = [table_widget.item(r, c).text() if table_widget.item(r, c) else ''
                       for c in range(table_widget.columnCount())]
                writer.writerow(row)
        QMessageBox.information(dlg, 'Export', 'CSV exported successfully.')
    btn_export_csv.clicked.connect(export_csv)
    btn_export_excel = QPushButton('Export to Excel')
    def export_excel():
        path, _ = QFileDialog.getSaveFileName(dlg, 'Export Excel', '', 'Excel Files (*.xlsx)')
        if not path:
            return
        df = pd.DataFrame([[table_widget.item(r, c).text() if table_widget.item(r, c) else ''
                            for c in range(table_widget.columnCount())]
                           for r in range(table_widget.rowCount())])
        df.to_excel(path, index=False, header=False)
        QMessageBox.information(dlg, 'Export', 'Excel exported successfully.')
    btn_export_excel.clicked.connect(export_excel)
    btn_export_multi = QPushButton('Export All Regions (Sheets/Files)')
    def export_multi():
        path, _ = QFileDialog.getSaveFileName(dlg, 'Export All Regions', '', 'Excel Files (*.xlsx);;CSV Files (*.csv)')
        if not path:
            return
        if path.endswith('.xlsx'):
            with pd.ExcelWriter(path) as writer:
                for i, t in enumerate(tables):
                    df = pd.DataFrame(t['table'])
                    sheet_name = t['label'][:31] if t['label'] else f'Region{i+1}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            QMessageBox.information(dlg, 'Export', 'All regions exported to Excel (multiple sheets).')
        else:
            base, ext = os.path.splitext(path)
            for i, t in enumerate(tables):
                fname = f"{base}_region{i+1}.csv"
                with open(fname, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for row in t['table']:
                        writer.writerow(row)
            QMessageBox.information(dlg, 'Export', 'All regions exported as separate CSV files.')
    btn_export_multi.clicked.connect(export_multi)
    # Debug export button
    btn_export_debug = QPushButton('Export Debug Info (Markdown)')
    def export_debug_md():
        if not debug_info:
            QMessageBox.warning(dlg, 'Debug Info', 'No debug info available.')
            return
        md = []
        md.append('# Typhoon OCR Debug Export\n')
        # Preprocessed images (show as base64 PNG)
        md.append('## Preprocessed Images\n')
        for i, img in enumerate(debug_info.get('preprocessed_images', [])):
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            md.append(f'### Page {i+1}\n')
            md.append(f'![](data:image/png;base64,{b64})\n')
        # Input payload
        md.append('## Typhoon OCR Input\n')
        md.append('```json')
        def sanitize(obj):
            if isinstance(obj, Image.Image):
                return '<PIL.Image>'
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(x) for x in obj]
            return obj
        sanitized_input = sanitize(debug_info.get('typhoon_input', {}))
        md.append(json.dumps(sanitized_input, indent=2))
        md.append('```\n')
        # Output
        md.append('## Typhoon OCR Output\n')
        md.append('```json')
        md.append(json.dumps(debug_info.get('typhoon_output', {}), indent=2))
        md.append('```\n')
        # Extracted tables
        md.append('## Extracted Tables\n')
        for t in tables:
            label = t.get('label', '')
            md.append(f'### {label}\n')
            table = t.get('table', [])
            if table:
                header = '| ' + ' | '.join(str(x) for x in table[0]) + ' |\n'
                sep = '| ' + ' | '.join(['---']*len(table[0])) + ' |\n'
                md.append(header)
                md.append(sep)
                for row in table[1:]:
                    md.append('| ' + ' | '.join(str(x) for x in row) + ' |\n')
            else:
                md.append('_No table extracted._\n')
        temp_path = os.path.join(tempfile.gettempdir(), 'ocraft_debug_export.md')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.writelines(md)
        QMessageBox.information(dlg, 'Debug Export', f'Debug info exported to:\n{temp_path}')
    btn_export_debug.clicked.connect(export_debug_md)
    btn_clean = QPushButton('Clean Table')
    def clean_table():
        # Get current table data
        data = [[table_widget.item(r, c).text() if table_widget.item(r, c) else ''
                 for c in range(table_widget.columnCount())]
                for r in range(table_widget.rowCount())]
        # Remove empty rows
        data = [row for row in data if any(cell.strip() for cell in row)]
        if not data:
            QMessageBox.information(dlg, 'Clean Table', 'Table is empty.')
            return
        # Normalize columns
        max_cols = max(len(row) for row in data)
        data = [row + [''] * (max_cols - len(row)) for row in data]
        # Remove empty columns
        non_empty_cols = [i for i in range(max_cols) if any(row[i].strip() for row in data)]
        data = [[row[i] for i in non_empty_cols] for row in data]
        # Refill table widget
        table_widget.setRowCount(len(data))
        table_widget.setColumnCount(len(data[0]) if data else 0)
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                table_widget.setItem(r, c, QTableWidgetItem(val))
        QMessageBox.information(dlg, 'Clean Table', 'Table cleaned and normalized.')
    btn_clean.clicked.connect(clean_table)
    # --- Advanced Table Editing Features ---
    btn_merge = QPushButton('Merge Selected Cells')
    def merge_selected_cells():
        selected = table_widget.selectedRanges()
        if not selected:
            QMessageBox.information(dlg, 'Merge', 'No cells selected.')
            return
        s = selected[0]
        merged_text = []
        for row in range(s.topRow(), s.bottomRow()+1):
            for col in range(s.leftColumn(), s.rightColumn()+1):
                item = table_widget.item(row, col)
                if item and item.text():
                    merged_text.append(item.text())
        merged = ' '.join(merged_text)
        for row in range(s.topRow(), s.bottomRow()+1):
            for col in range(s.leftColumn(), s.rightColumn()+1):
                table_widget.setItem(row, col, QTableWidgetItem(''))
        table_widget.setItem(s.topRow(), s.leftColumn(), QTableWidgetItem(merged))
    btn_merge.clicked.connect(merge_selected_cells)

    btn_split = QPushButton('Split Cell (by space)')
    def split_cell():
        items = table_widget.selectedItems()
        if not items:
            QMessageBox.information(dlg, 'Split', 'No cell selected.')
            return
        item = items[0]
        row, col = item.row(), item.column()
        parts = item.text().split()
        for i, part in enumerate(parts):
            if col + i < table_widget.columnCount():
                table_widget.setItem(row, col + i, QTableWidgetItem(part))
    btn_split.clicked.connect(split_cell)

    btn_row_up = QPushButton('Move Row Up')
    def move_row_up():
        row = table_widget.currentRow()
        if row <= 0:
            return
        for col in range(table_widget.columnCount()):
            above = table_widget.item(row-1, col)
            current = table_widget.item(row, col)
            above_text = above.text() if above else ''
            current_text = current.text() if current else ''
            table_widget.setItem(row-1, col, QTableWidgetItem(current_text))
            table_widget.setItem(row, col, QTableWidgetItem(above_text))
        table_widget.selectRow(row-1)
    btn_row_up.clicked.connect(move_row_up)

    btn_row_down = QPushButton('Move Row Down')
    def move_row_down():
        row = table_widget.currentRow()
        if row < 0 or row >= table_widget.rowCount()-1:
            return
        for col in range(table_widget.columnCount()):
            below = table_widget.item(row+1, col)
            current = table_widget.item(row, col)
            below_text = below.text() if below else ''
            current_text = current.text() if current else ''
            table_widget.setItem(row+1, col, QTableWidgetItem(current_text))
            table_widget.setItem(row, col, QTableWidgetItem(below_text))
        table_widget.selectRow(row+1)
    btn_row_down.clicked.connect(move_row_down)

    btn_insert_row = QPushButton('Insert Row')
    def insert_row():
        row = table_widget.currentRow()
        if row < 0:
            row = 0
        table_widget.insertRow(row)
    btn_insert_row.clicked.connect(insert_row)

    btn_delete_row = QPushButton('Delete Row')
    def delete_row():
        row = table_widget.currentRow()
        if row >= 0:
            table_widget.removeRow(row)
    btn_delete_row.clicked.connect(delete_row)

    btn_insert_col = QPushButton('Insert Column')
    def insert_col():
        col = table_widget.currentColumn()
        if col < 0:
            col = 0
        table_widget.insertColumn(col)
    btn_insert_col.clicked.connect(insert_col)

    btn_delete_col = QPushButton('Delete Column')
    def delete_col():
        col = table_widget.currentColumn()
        if col >= 0:
            table_widget.removeColumn(col)
    btn_delete_col.clicked.connect(delete_col)

    # Add advanced editing buttons to the right panel
    right_layout.addWidget(btn_merge)
    right_layout.addWidget(btn_split)
    right_layout.addWidget(btn_row_up)
    right_layout.addWidget(btn_row_down)
    right_layout.addWidget(btn_insert_row)
    right_layout.addWidget(btn_delete_row)
    right_layout.addWidget(btn_insert_col)
    right_layout.addWidget(btn_delete_col)
    right_layout.addWidget(btn_clear)
    right_layout.addWidget(btn_undo)
    right_layout.addWidget(btn_fit)
    right_layout.addWidget(btn_export_csv)
    right_layout.addWidget(btn_export_excel)
    right_layout.addWidget(btn_export_multi)
    right_layout.addWidget(btn_export_debug)
    right_layout.addWidget(btn_clean)
    right_layout.addStretch(1)
    # Add error label and region list to layout
    left_panel = QVBoxLayout()
    left_panel.addWidget(region_list_widget)
    left_panel.addWidget(error_label)
    left_panel.addStretch(1)
    left_panel_widget = QWidget()
    left_panel_widget.setLayout(left_panel)
    layout.addWidget(left_panel_widget)
    layout.addWidget(table_widget, stretch=1)
    layout.addWidget(right_panel)
    dlg.exec_()
