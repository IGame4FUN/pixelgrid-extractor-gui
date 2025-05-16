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

import numpy as np
import cv2
import random

# ---- DEFAULT CONFIG ----
EDGE_THRESHOLD = 65
MERGE_DISTANCE = 7
MAX_EDGES_PER_LINE = 15
MIN_EDGES_FOR_GUIDE_CREATION = 20
FAST_MODE = True
REFINEMENT_PASSES = 0
REFINEMENT_RANGE = 1
REFINEMENT_PENALTY_WEIGHT = 10.25
SIZE_PENALTY_WEIGHT = 16.0


def _prepare_image_data(img):
    if img is None:
        raise ValueError("Input image data is None.")

    img = img.copy()
    if img.ndim == 3 and img.shape[2] == 4:  # BGRA
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        blended = bgr.astype(np.float32) * alpha + 255 * (1.0 - alpha)
        img_gray = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        img_gray = img
    elif img.ndim == 3 and img.shape[2] == 2:
        gray_c = img[:, :, 0]
        alpha_c = img[:, :, 1:2].astype(np.float32) / 255.0
        blended = gray_c.astype(np.float32) * alpha_c.squeeze(-1) + 255 * (1.0 - alpha_c.squeeze(-1))
        img_gray = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img_gray


def _compute_gradients(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def _detect_and_merge_guides(gradient, horizontal,
                             edge_threshold, merge_distance,
                             max_edges, min_edges, fast=True):
    guides = []
    length = gradient.shape[0] if horizontal else gradient.shape[1]
    abs_gradient = np.abs(gradient.astype(np.float64))

    all_edge_positions = []
    for line in range(length):
        vals = abs_gradient[line] if horizontal else abs_gradient[:, line]
        indices = np.where(vals > edge_threshold)[0]
        if len(indices) > max_edges:
            indices = random.sample(list(indices), max_edges)
        all_edge_positions.extend(list(indices))

    if not all_edge_positions:
        return []

    all_edge_positions.sort()
    merged = []
    cur_sum = float(all_edge_positions[0])
    cur_cnt = 1
    for i in range(1, len(all_edge_positions)):
        p = float(all_edge_positions[i])
        mean = cur_sum / cur_cnt
        if abs(p - mean) <= merge_distance:
            cur_sum += p
            cur_cnt += 1
        else:
            if cur_cnt >= min_edges:
                merged.append(int(round(cur_sum / cur_cnt)))
            cur_sum = p
            cur_cnt = 1
    if cur_cnt >= min_edges:
        merged.append(int(round(cur_sum / cur_cnt)))
    return sorted(set(merged))


def _refine_guides(v_guides, h_guides, gray, passes, refinement_range, penalty_weight, size_weight):
    if not v_guides or not h_guides or passes == 0:
        return v_guides, h_guides
    orig_v, orig_h = v_guides.copy(), h_guides.copy()
    refined_v = [float(x) for x in v_guides]
    refined_h = [float(y) for y in h_guides]
    h_img, w_img = gray.shape
    for _ in range(passes):
        ideal_w = np.mean([refined_v[i + 1] - refined_v[i] for i in range(len(refined_v) - 1)]) if len(
            refined_v) > 1 else float(w_img)
        ideal_h = np.mean([refined_h[i + 1] - refined_h[i] for i in range(len(refined_h) - 1)]) if len(
            refined_h) > 1 else float(h_img)
        if np.isnan(ideal_w) or ideal_w == 0: ideal_w = float(w_img) / max(1, len(refined_v))
        if np.isnan(ideal_h) or ideal_h == 0: ideal_h = float(h_img) / max(1, len(refined_h))
        # V
        for i in range(1, len(refined_v) - 1):
            left, right, cur = refined_v[i - 1], refined_v[i + 1], refined_v[i]
            best_x, best_score = cur, float('inf')
            for d in range(-refinement_range, refinement_range + 1):
                x_cand = cur + d
                if not (left < x_cand < right and 0 <= x_cand < w_img): continue
                scores = []
                for j in range(len(refined_h) - 1):
                    y0, y1 = int(round(refined_h[j])), int(round(refined_h[j + 1]))
                    if y1 <= y0: continue
                    region_l = gray[y0:y1, int(round(left)):int(round(x_cand))]
                    region_r = gray[y0:y1, int(round(x_cand)):int(round(right))]
                    if region_l.size > 0: scores.append(region_l.var())
                    if region_r.size > 0: scores.append(region_r.var())
                mean_var = np.mean(scores) if scores else 0.0
                size_pen = abs(x_cand - left - ideal_w) + abs(right - x_cand - ideal_w)
                pos_pen = abs(x_cand - orig_v[i])
                score = mean_var + size_weight * size_pen + penalty_weight * pos_pen
                if score < best_score:
                    best_score, best_x = score, x_cand
            refined_v[i] = best_x
        # H
        for j in range(1, len(refined_h) - 1):
            top, bottom, cur = refined_h[j - 1], refined_h[j + 1], refined_h[j]
            best_y, best_score = cur, float('inf')
            for d in range(-refinement_range, refinement_range + 1):
                y_cand = cur + d
                if not (top < y_cand < bottom and 0 <= y_cand < h_img): continue
                scores = []
                for i in range(len(refined_v) - 1):
                    x0, x1 = int(round(refined_v[i])), int(round(refined_v[i + 1]))
                    if x1 <= x0: continue
                    region_t = gray[int(round(top)):int(round(y_cand)), x0:x1]
                    region_b = gray[int(round(y_cand)):int(round(bottom)), x0:x1]
                    if region_t.size > 0: scores.append(region_t.var())
                    if region_b.size > 0: scores.append(region_b.var())
                mean_var = np.mean(scores) if scores else 0.0
                size_pen = abs(y_cand - top - ideal_h) + abs(bottom - y_cand - ideal_h)
                pos_pen = abs(y_cand - orig_h[j])
                score = mean_var + size_weight * size_pen + penalty_weight * pos_pen
                if score < best_score:
                    best_score, best_y = score, y_cand
            refined_h[j] = best_y
    return [int(round(x)) for x in refined_v], [int(round(y)) for y in refined_h]


def _create_cell_images(img, v_guides, h_guides):
    color_img = img.copy()
    if color_img.ndim == 2:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
    elif color_img.ndim == 3 and color_img.shape[2] == 4:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
    elif color_img.ndim == 3 and color_img.shape[2] == 1:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
    h_img, w_img, num_ch = color_img.shape
    sampled = np.zeros_like(color_img)
    averaged = np.zeros_like(color_img)
    x_coords = sorted(set(map(int, v_guides)))
    y_coords = sorted(set(map(int, h_guides)))
    x_edges = sorted(set([0] + x_coords + [w_img]))
    y_edges = sorted(set([0] + y_coords + [h_img]))
    x_edges = [x for x in x_edges if 0 <= x <= w_img]
    y_edges = [y for y in y_edges if 0 <= y <= h_img]
    num_rows = len(y_edges) - 1 if len(y_edges) > 1 else 0
    num_cols = len(x_edges) - 1 if len(x_edges) > 1 else 0
    if num_rows <= 0 or num_cols <= 0:
        return None, None, None
    pixel_perfect = np.zeros((num_rows, num_cols, num_ch), dtype=np.uint8)
    for ci in range(num_cols):
        x0, x1 = x_edges[ci], x_edges[ci + 1]
        if x1 <= x0: continue
        for ri in range(num_rows):
            y0, y1 = y_edges[ri], y_edges[ri + 1]
            if y1 <= y0: continue
            cx = min(max((x0 + x1) // 2, 0), w_img - 1)
            cy = min(max((y0 + y1) // 2, 0), h_img - 1)
            centre = color_img[cy, cx]
            roi = color_img[y0:y1, x0:x1]
            mean = centre if roi.size == 0 else roi.reshape(-1, num_ch).mean(axis=0).astype(np.uint8)
            sampled[y0:y1, x0:x1] = centre
            averaged[y0:y1, x0:x1] = mean
            pixel_perfect[ri, ci] = centre
    return sampled, averaged, pixel_perfect


def process_image_grid(
        img,
        edge_threshold=EDGE_THRESHOLD,
        merge_distance=MERGE_DISTANCE,
        max_edges_per_line=MAX_EDGES_PER_LINE,
        min_edges_for_guide=MIN_EDGES_FOR_GUIDE_CREATION,
        fast_mode=FAST_MODE,
        refinement_passes=REFINEMENT_PASSES,
        refinement_range=REFINEMENT_RANGE,
        refinement_penalty_weight=REFINEMENT_PENALTY_WEIGHT,
        size_penalty_weight=SIZE_PENALTY_WEIGHT,
):
    """
    Process a grid image. Returns:
      dict with:
        - 'v_guides', 'h_guides': guide positions (after refinement)
        - 'v_guides_initial', 'h_guides_initial': before refinement
        - 'sampled', 'averaged', 'pixel_perfect': cell images (np.ndarray)
    Accepts:
        img: np.ndarray (OpenCV image), or str (image path)
    """
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Image could not be loaded.")
    gray = _prepare_image_data(img)
    gx, gy = _compute_gradients(gray)
    v_guides_initial = _detect_and_merge_guides(
        gx, True, edge_threshold, merge_distance, max_edges_per_line, min_edges_for_guide, fast_mode)
    h_guides_initial = _detect_and_merge_guides(
        gy, False, edge_threshold, merge_distance, max_edges_per_line, min_edges_for_guide, fast_mode)
    v_guides, h_guides = _refine_guides(
        v_guides_initial, h_guides_initial, gray, refinement_passes, refinement_range, refinement_penalty_weight,
        size_penalty_weight)
    sampled, averaged, pixel_perfect = _create_cell_images(img, v_guides, h_guides)
    return {
        "v_guides": v_guides,
        "h_guides": h_guides,
        "v_guides_initial": v_guides_initial,
        "h_guides_initial": h_guides_initial,
        "sampled": sampled,
        "averaged": averaged,
        "pixel_perfect": pixel_perfect
    }
