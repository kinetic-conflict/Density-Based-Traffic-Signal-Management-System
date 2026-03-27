from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    conf: float
    cls_name: str
    source: str  # "plate-yolo" | "general-yolo+heuristic"


COCO_VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
}


class PlateDetector:
    """
    Two-stage detector:
    - Preferred: plate-specific YOLO weights (single-stage plates).
    - Fallback: general YOLOv8n for vehicles, then classical plate-candidate search within vehicle ROIs.
    """

    def __init__(
        self,
        plate_model_path: str,
        general_model_name: str,
        device: str,
        imgsz: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.device = device
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.plate_model_path = plate_model_path
        self.general_model_name = general_model_name

        self.plate_model: Optional[YOLO] = None
        self.general_model: Optional[YOLO] = None

        self._init_models()

    def _init_models(self) -> None:
        try:
            self.plate_model = YOLO(self.plate_model_path)
            _ = self.plate_model.names
        except Exception:
            self.plate_model = None

        try:
            self.general_model = YOLO(self.general_model_name)
            _ = self.general_model.names
        except Exception:
            self.general_model = None

        if self.plate_model is None and self.general_model is None:
            raise RuntimeError(
                "Could not load any YOLO models. Ensure `ultralytics` is installed and "
                "either a plate model path exists or `yolov8n.pt` can be downloaded."
            )

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        if self.plate_model is not None:
            dets = self._detect_with_plate_yolo(bgr)
            if dets:
                return dets
        return self._detect_with_general_fallback(bgr)

    def _detect_with_plate_yolo(self, bgr: np.ndarray) -> list[Detection]:
        assert self.plate_model is not None
        results = self.plate_model.predict(
            source=bgr,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            verbose=False,
        )
        out: list[Detection] = []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return out
        names = r0.names
        for b in r0.boxes:
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            cls_i = int(b.cls[0].item()) if b.cls is not None else -1
            cls_name = names.get(cls_i, "plate")
            out.append(Detection((x1, y1, x2, y2), conf, cls_name, "plate-yolo"))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out

    def _detect_with_general_fallback(self, bgr: np.ndarray) -> list[Detection]:
        if self.general_model is None:
            return []
        results = self.general_model.predict(
            source=bgr,
            device=self.device,
            imgsz=self.imgsz,
            conf=max(0.15, self.conf_thres - 0.1),
            iou=self.iou_thres,
            max_det=50,
            verbose=False,
        )
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        names = r0.names
        h, w = bgr.shape[:2]

        vehicle_boxes: list[tuple[int, int, int, int, float, str]] = []
        for b in r0.boxes:
            cls_i = int(b.cls[0].item()) if b.cls is not None else -1
            cls_name = names.get(cls_i, "")
            if cls_name not in COCO_VEHICLE_CLASSES:
                continue
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue
            vehicle_boxes.append((x1, y1, x2, y2, conf, cls_name))

        vehicle_boxes.sort(key=lambda x: x[4], reverse=True)
        vehicle_boxes = vehicle_boxes[:5]

        plates: list[Detection] = []
        for (x1, y1, x2, y2, vconf, vname) in vehicle_boxes:
            roi = bgr[y1:y2, x1:x2]
            candidates = self._plate_candidates_in_roi(roi)
            for (cx1, cy1, cx2, cy2, score) in candidates[:2]:
                px1, py1, px2, py2 = x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2
                conf = float(min(0.99, 0.35 + 0.65 * score) * max(0.3, vconf))
                plates.append(Detection((px1, py1, px2, py2), conf, f"plate@{vname}", "general-yolo+heuristic"))

        plates.sort(key=lambda d: d.conf, reverse=True)
        return plates[: self.max_det]

    def _plate_candidates_in_roi(self, bgr_roi: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        Classical candidates: strong rectangular, high edge density, plate-like aspect ratio.
        Returns candidates as (x1,y1,x2,y2,score_0_1) in ROI coords.
        """
        if bgr_roi.size == 0:
            return []
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 60, 60)
        edges = cv2.Canny(gray, 80, 200)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        roi_area = float(h * w)
        out: list[tuple[int, int, int, int, float]] = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 40 or bh < 12:
                continue
            area = bw * bh
            if area / roi_area < 0.01:
                continue
            ar = bw / float(bh)
            if not (1.8 <= ar <= 6.5):
                continue

            # edge density score inside the candidate
            patch = edges[y : y + bh, x : x + bw]
            density = float(np.mean(patch > 0))
            # prefer mid-lower region of vehicle (plates often there)
            y_center = (y + y + bh) / 2.0
            vertical_prior = 1.0 - abs((y_center / max(1.0, h)) - 0.7)
            score = 0.7 * density + 0.3 * max(0.0, vertical_prior)
            out.append((x, y, x + bw, y + bh, float(np.clip(score, 0.0, 1.0))))

        out.sort(key=lambda t: t[4], reverse=True)
        return out

