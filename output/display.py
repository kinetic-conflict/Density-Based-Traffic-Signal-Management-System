from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DrawConfig:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    thickness: int = 2


class Visualizer:
    def __init__(self, cfg: DrawConfig | None = None) -> None:
        self.cfg = cfg or DrawConfig()

    @staticmethod
    def _clamp_text(s: str, max_len: int = 16) -> str:
        s2 = s.strip()
        return s2 if len(s2) <= max_len else s2[: max_len - 1] + "…"

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 25),
            self.cfg.font,
            self.cfg.font_scale,
            (0, 255, 0),
            self.cfg.thickness,
            cv2.LINE_AA,
        )

    def draw_plate_annotation(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        det_conf: float,
        plate_text: str,
        ocr_conf: float,
        ocr_source: str,
    ) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        # Color encodes detection confidence (roughly).
        conf = float(det_conf)
        if conf >= 0.6:
            color = (0, 255, 0)
        elif conf >= 0.35:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        plate_text = self._clamp_text(plate_text, max_len=18) if plate_text else "?"
        label = f"{plate_text} | det:{conf:.2f} | ocr:{ocr_conf:.2f} {ocr_source}"

        # Place label above the bbox; if not enough space, place inside.
        (tw, th), _ = cv2.getTextSize(label, self.cfg.font, self.cfg.font_scale, self.cfg.thickness)
        label_x = x1
        label_y = y1 - 8
        if label_y - th < 0:
            label_y = y1 + th + 6

        # Background rectangle for readability.
        y_top = max(0, label_y - th - 4)
        y_bottom = min(frame.shape[0] - 1, label_y + 2)
        x_right = min(frame.shape[1] - 1, label_x + tw + 6)
        cv2.rectangle(frame, (label_x, y_top), (x_right, y_bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (label_x + 3, label_y),
            self.cfg.font,
            self.cfg.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

