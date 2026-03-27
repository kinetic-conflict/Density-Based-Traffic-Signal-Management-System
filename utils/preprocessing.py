from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ResizeResult:
    image: np.ndarray
    scale: float  # new = old * scale


def resize_max_side(bgr: np.ndarray, max_side: int) -> ResizeResult:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return ResizeResult(bgr, 1.0)
    scale = max_side / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return ResizeResult(resized, scale)


def brightness_score(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return float(np.mean(v))


def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return bgr
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


def adaptive_enhance(
    bgr: np.ndarray,
    brightness_low: float,
    brightness_high: float,
    gamma_dark: float,
    gamma_bright: float,
    clahe_clip_limit: float,
    clahe_tile_grid_size: int,
) -> np.ndarray:
    score = brightness_score(bgr)
    out = bgr
    if score < brightness_low:
        out = apply_gamma(out, gamma_dark)
    elif score > brightness_high:
        out = apply_gamma(out, gamma_bright)
    out = apply_clahe(out, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
    return out


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def upscale(gray: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def threshold_variants(gray: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    g = gray
    out.append(g)
    out.append(cv2.GaussianBlur(g, (3, 3), 0))

    _, t1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(t1)
    _, t2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(t2)

    t3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    out.append(t3)
    t4 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    out.append(t4)

    return out


def ocr_preprocess_attempts(bgr_plate: np.ndarray, upscale_factor: int) -> list[np.ndarray]:
    gray = to_gray(bgr_plate)
    gray = upscale(gray, upscale_factor)
    variants = threshold_variants(gray)
    # light denoise variants
    variants.extend([cv2.medianBlur(v, 3) for v in variants[:3]])
    # ensure uint8 single channel
    cleaned: list[np.ndarray] = []
    for v in variants:
        if v.ndim == 3:
            v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        cleaned.append(v.astype(np.uint8))
    return cleaned


def clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0

    # Use half-open slicing semantics [x1:x2], [y1:y2]:
    # x1 in [0, w-1], x2 in [1, w], and similarly for y.
    x1c = max(0, min(x1, w - 1))
    y1c = max(0, min(y1, h - 1))
    x2c = max(1, min(x2, w))
    y2c = max(1, min(y2, h))
    if x2c <= x1c:
        x2c = min(w, x1c + 1)
    if y2c <= y1c:
        y2c = min(h, y1c + 1)
    return x1c, y1c, x2c, y2c


def crop_with_padding(bgr: np.ndarray, xyxy: tuple[int, int, int, int], pad_ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad = int(round(max(bw, bh) * pad_ratio))
    x1p, y1p, x2p, y2p = clamp_xyxy(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)
    return bgr[y1p:y2p, x1p:x2p].copy(), (x1p, y1p, x2p, y2p)


def refine_plate_crop_dynamic(
    bgr_crop: np.ndarray,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
) -> np.ndarray:
    """
    Dynamically refine a coarse plate crop to a tighter text-rich plate region.
    If no reliable candidate is found, returns the original crop unchanged.
    """
    if bgr_crop.size == 0:
        return bgr_crop

    h, w = bgr_crop.shape[:2]
    if h < 20 or w < 40:
        return bgr_crop

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)

    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    gmin, gmax = float(grad_x.min()), float(grad_x.max())
    if gmax > gmin:
        grad_x = ((grad_x - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
    else:
        grad_x = np.zeros_like(gray, dtype=np.uint8)

    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_k, iterations=1)
    _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.erode(bw, None, iterations=1)
    bw = cv2.dilate(bw, None, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr_crop

    crop_area = float(h * w)
    best_rect: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for c in contours:
        x, y, bwc, bhc = cv2.boundingRect(c)
        area = float(bwc * bhc)
        area_ratio = area / crop_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        ar = bwc / float(max(1, bhc))
        if not (1.6 <= ar <= 8.5):
            continue

        patch = grad_x[y : y + bhc, x : x + bwc]
        grad_score = float(np.mean(patch) / 255.0)
        center_y = (y + y + bhc) / 2.0
        vertical_prior = 1.0 - abs((center_y / max(1.0, h)) - 0.55)
        score = 0.7 * grad_score + 0.3 * max(0.0, vertical_prior)

        if score > best_score:
            best_score = score
            best_rect = (x, y, x + bwc, y + bhc)

    if best_rect is None:
        return bgr_crop

    x1, y1, x2, y2 = best_rect
    pad_x = int(round((x2 - x1) * 0.05))
    pad_y = int(round((y2 - y1) * 0.12))
    x1, y1, x2, y2 = clamp_xyxy(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)
    return bgr_crop[y1:y2, x1:x2].copy()


def normalize_text(s: str) -> str:
    s2 = "".join(ch for ch in s.upper() if ch.isalnum())
    return s2


def is_valid_plate_text(s: str) -> bool:
    s2 = normalize_text(s)
    if not (5 <= len(s2) <= 10):
        return False
    if not any(ch.isdigit() for ch in s2):
        return False
    return True


def best_text_candidate(candidates: Iterable[tuple[str, float, str]]) -> tuple[str, float, str]:
    """
    candidates: (text, confidence_0_1, source)
    Select highest confidence valid text; if none valid, select highest confidence overall.
    """
    cand = list(candidates)
    valid = [(t, c, src) for (t, c, src) in cand if is_valid_plate_text(t)]
    pool = valid if valid else cand
    if not pool:
        return "", 0.0, "none"
    pool.sort(key=lambda x: x[1], reverse=True)
    t, c, src = pool[0]
    return normalize_text(t), float(c), src

