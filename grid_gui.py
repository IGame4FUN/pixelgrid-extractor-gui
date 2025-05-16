# -----------------------------------------------------------------------------
#  PixelGrid Extractor - Grid Detection and Extraction for Pixel Art
#
#  MIT License
#  Copyright (c) 2025 IG
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# -----------------------------------------------------------------------------

import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QFileDialog, QCheckBox, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from library_infer_pixels import process_image_grid

def np2qpixmap(arr):
    if arr is None:
        return QPixmap()
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w, ch = arr.shape
    try:
        return QPixmap(QImage(arr.data, w, h, ch * w, QImage.Format_RGB888))
    except Exception as e:
        print("QImage conversion error:", e)
        return QPixmap()


def draw_guides_overlay(img, v_guides, h_guides, color=(0, 255, 0)):
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for x in v_guides:
        cv2.line(vis, (x, 0), (x, vis.shape[0]), color, 1)
    for y in h_guides:
        cv2.line(vis, (0, y), (vis.shape[1], y), color, 1)
    return vis


EDGE_THRESHOLD = 65
MERGE_DISTANCE = 7
MIN_EDGES_FOR_GUIDE_CREATION = 20
MAX_EDGES_PER_LINE = 15
FAST_MODE = True
REFINEMENT_PASSES = 0
REFINEMENT_RANGE = 1
REFINEMENT_PENALTY_WEIGHT = 10.25
SIZE_PENALTY_WEIGHT = 16.0


def make_labeled_slider(label_text, minv, maxv, init, int_mode=True, step=1):
    label = QLabel(f"{label_text}: {init}")
    s = QSlider(Qt.Horizontal)
    s.setMinimum(minv)
    s.setMaximum(maxv)
    s.setValue(init)
    if int_mode:
        s.setSingleStep(step)
    else:
        s.setSingleStep(1)
    s.setTickInterval(1)

    def update_label(val):
        val_disp = val if int_mode else val / 100.0
        label.setText(f"{label_text}: {val_disp}")

    s.valueChanged.connect(update_label)
    update_label(init)
    box = QGroupBox()
    lay = QVBoxLayout()
    lay.addWidget(label)
    lay.addWidget(s)
    box.setLayout(lay)
    return box, s, label


class GridGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setWindowTitle("Grid Detection GUI")
        self.img = None
        self.result = None

        # --- Controls (with value labels) ---
        self.thresh_box, self.thresh_slider, self.thresh_label = make_labeled_slider("Edge Thresh", 30, 150,
                                                                                     EDGE_THRESHOLD)
        self.merge_box, self.merge_slider, self.merge_label = make_labeled_slider("Merge Dist", 1, 20, MERGE_DISTANCE)
        self.min_edge_box, self.min_edge_slider, self.min_edge_label = make_labeled_slider("Min Edges", 5, 40,
                                                                                           MIN_EDGES_FOR_GUIDE_CREATION)
        self.refine_passes_box, self.refine_passes_slider, self.refine_passes_label = make_labeled_slider(
            "Refine Passes", 0, 3, REFINEMENT_PASSES)
        self.refine_box = QCheckBox("Enable Refinement")
        self.refine_box.setChecked(False)

        # -- Refine parameters (with decimal sliders) --
        self.refine_range_box, self.refine_range_slider, self.refine_range_label = make_labeled_slider("Refine Range",
                                                                                                       0, 5,
                                                                                                       REFINEMENT_RANGE)
        self.refine_penalty_box, self.refine_penalty_slider, self.refine_penalty_label = make_labeled_slider(
            "Penalty Weight", 0, 3000, int(REFINEMENT_PENALTY_WEIGHT * 100), int_mode=False)
        self.size_penalty_box, self.size_penalty_slider, self.size_penalty_label = make_labeled_slider("Size Penalty",
                                                                                                       0, 3000,
                                                                                                       int(SIZE_PENALTY_WEIGHT * 100),
                                                                                                       int_mode=False)

        # --- Image previews ---
        self.grid_label = QLabel("Load an image...")
        self.pixel_label = QLabel("")
        self.grid_label.setFixedSize(360, 360)
        self.pixel_label.setFixedSize(180, 180)

        # --- Layout ---
        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(self.thresh_box)
        sliders_layout.addWidget(self.merge_box)
        sliders_layout.addWidget(self.min_edge_box)
        sliders_layout.addWidget(self.refine_passes_box)
        sliders_layout.addWidget(self.refine_box)
        sliders_layout.addWidget(self.refine_range_box)
        sliders_layout.addWidget(self.refine_penalty_box)
        sliders_layout.addWidget(self.size_penalty_box)
        sliders_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.grid_label)
        left_layout.addWidget(self.pixel_label)
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.save_btn = QPushButton("Save Outputs")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        left_layout.addLayout(btn_row)

        main = QHBoxLayout()
        main.addLayout(left_layout)
        main.addLayout(sliders_layout)
        self.setLayout(main)

        # --- Signals ---
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_outputs)
        self.thresh_slider.valueChanged.connect(self.update_previews)
        self.merge_slider.valueChanged.connect(self.update_previews)
        self.min_edge_slider.valueChanged.connect(self.update_previews)
        self.refine_box.stateChanged.connect(self.update_previews)
        self.refine_passes_slider.valueChanged.connect(self.update_previews)
        self.refine_range_slider.valueChanged.connect(self.update_previews)
        self.refine_penalty_slider.valueChanged.connect(self.update_previews)
        self.size_penalty_slider.valueChanged.connect(self.update_previews)

        self.setMinimumWidth(900)
        self.setMinimumHeight(500)

    def debug_img_load(self, path, img):
        print(f"Trying to load: {path}")
        if img is None:
            print("Image load failed.")
        else:
            print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            print("File dialog canceled or no file selected.")
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.debug_img_load(path, img)
        if img is None:
            self.grid_label.setText("Load failed")
            return
        self.img = img
        self.setWindowTitle(f"Grid Detection GUI - {path}")
        self.update_previews()

    def update_previews(self):
        if self.img is None:
            return
        params = dict(
            edge_threshold=self.thresh_slider.value(),
            merge_distance=self.merge_slider.value(),
            max_edges_per_line=MAX_EDGES_PER_LINE,
            min_edges_for_guide=self.min_edge_slider.value(),
            fast_mode=FAST_MODE,
            refinement_passes=self.refine_passes_slider.value() if self.refine_box.isChecked() else 0,
            refinement_range=self.refine_range_slider.value(),
            refinement_penalty_weight=self.refine_penalty_slider.value() / 100.0,
            size_penalty_weight=self.size_penalty_slider.value() / 100.0
        )
        try:
            self.result = process_image_grid(self.img, **params)
        except Exception as e:
            print("Grid processing error:", e)
            self.grid_label.setText("Processing error")
            return
        vis = draw_guides_overlay(
            self.img, self.result['v_guides'], self.result['h_guides'])
        vis_small = cv2.resize(vis, (360, 360), interpolation=cv2.INTER_AREA)
        self.grid_label.setPixmap(np2qpixmap(vis_small))
        px = self.result['pixel_perfect']
        if px is not None and px.size > 0:
            px_vis = cv2.resize(px, (180, 180), interpolation=cv2.INTER_NEAREST)
            self.pixel_label.setPixmap(np2qpixmap(px_vis))
        else:
            self.pixel_label.clear()

    def save_outputs(self):
        if self.result is None:
            print("Nothing to save.")
            return
        grid_path, _ = QFileDialog.getSaveFileName(self, "Save Grid Overlay", "", "PNG Files (*.png)")
        if not grid_path:
            print("Save dialog canceled.")
            return
        vis = draw_guides_overlay(self.img, self.result['v_guides'], self.result['h_guides'])
        try:
            cv2.imwrite(grid_path, vis)
            print(f"Saved grid overlay to {grid_path}")
        except Exception as e:
            print("Failed to save grid overlay:", e)
        px = self.result['pixel_perfect']
        if px is not None and px.size > 0:
            px_path = grid_path.rsplit('.', 1)[0] + "_pixel.png"
            try:
                cv2.imwrite(px_path, px)
                print(f"Saved pixel perfect to {px_path}")
            except Exception as e:
                print("Failed to save pixel perfect:", e)

    # --- Drag and Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = url.toLocalFile()
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                self.debug_img_load(img_path, img)
                if img is not None:
                    self.img = img
                    self.setWindowTitle(f"Grid Detection GUI - {img_path}")
                    self.update_previews()
                else:
                    self.grid_label.setText("Load failed")
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = GridGui()
    w.show()
    sys.exit(app.exec_())
