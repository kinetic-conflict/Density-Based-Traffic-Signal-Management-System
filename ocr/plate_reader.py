from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import pytesseract

from easyocr import Reader as EasyOCRReader

from utils.preprocessing import best_text_candidate


@dataclass(frozen=True)
class OCRResult:
    text: str
    conf: float  # 0..1
    source: str  # "easyocr" | "tesseract" | "none"


class PlateReader:
    def __init__(self, langs: tuple[str, ...], tesseract_cmd: str = "") -> None:
        self.easy_reader = EasyOCRReader(list(langs), gpu=False)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def read_plate(self, gray_or_bin_attempts: list[np.ndarray]) -> OCRResult:
        """
        Try multiple preprocessed attempts; best candidate wins.
        EasyOCR first; fallback to Tesseract when confidence is weak or invalid.
        """
        easy_cands: list[tuple[str, float, str]] = []
        for img in gray_or_bin_attempts[:6]:
            t, c = self._easyocr_once(img)
            if t:
                easy_cands.append((t, c, "easyocr"))
        easy_text, easy_conf, _ = best_text_candidate(easy_cands)

        if easy_text and easy_conf >= 0.35:
            return OCRResult(easy_text, easy_conf, "easyocr")

        tess_cands: list[tuple[str, float, str]] = []
        for img in gray_or_bin_attempts[:8]:
            t, c = self._tesseract_once(img)
            if t:
                tess_cands.append((t, c, "tesseract"))
        tess_text, tess_conf, src = best_text_candidate(tess_cands)

        if tess_text and tess_conf >= max(0.20, easy_conf):
            return OCRResult(tess_text, tess_conf, src)

        if easy_text:
            return OCRResult(easy_text, easy_conf, "easyocr")
        if tess_text:
            return OCRResult(tess_text, tess_conf, src)
        return OCRResult("", 0.0, "none")

    def _easyocr_once(self, img: np.ndarray) -> tuple[str, float]:
        try:
            res = self.easy_reader.readtext(img, detail=1, paragraph=False)
        except Exception:
            return "", 0.0
        if not res:
            return "", 0.0
        best = max(res, key=lambda r: float(r[2]) if len(r) >= 3 else 0.0)
        text = best[1] if len(best) >= 2 else ""
        conf = float(best[2]) if len(best) >= 3 else 0.0
        return str(text), float(np.clip(conf, 0.0, 1.0))

    def _tesseract_once(self, img: np.ndarray) -> tuple[str, float]:
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        except Exception:
            return "", 0.0
        texts = data.get("text", []) or []
        confs = data.get("conf", []) or []
        pairs: list[tuple[str, float]] = []
        for t, c in zip(texts, confs):
            t2 = (t or "").strip()
            if not t2:
                continue
            try:
                c2 = float(c)
            except Exception:
                continue
            if c2 < 0:
                continue
            pairs.append((t2, c2))
        if not pairs:
            return "", 0.0
        merged = "".join([p[0] for p in pairs])
        mean_conf = float(np.mean([p[1] for p in pairs])) / 100.0
        return merged, float(np.clip(mean_conf, 0.0, 1.0))

