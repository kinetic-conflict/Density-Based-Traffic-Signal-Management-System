from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v.strip() == "" else v.strip()


@dataclass(frozen=True)
class AppConfig:
    # Dataset roots (can be overridden via environment variables)
    dataset_dir_1: Path = Path(
        _env(
            "PLATE_DATASET_1",
            r"D:\VIT_ACADEMICS\MDP_SW\Indian_Number_Plates\Sample_Images",
        )
    )
    dataset_dir_2: Path = Path(
        _env(
            "PLATE_DATASET_2",
            r"D:\VIT_ACADEMICS\MDP_SW\number_plate_images_ocr\number_plate_images_ocr",
        )
    )

    # Detection models
    # Preferred: a plate-specific YOLOv8 weight (.pt). If missing/unloadable, fallback mode triggers.
    plate_model_path: str = _env("PLATE_MODEL_PATH", "license_plate_detector.pt")
    general_model_name: str = _env("GENERAL_YOLO_MODEL", "yolov8n.pt")

    # Runtime
    device: str = _env("PLATE_DEVICE", "cpu")  # must run on CPU for this phase
    imgsz: int = int(_env("PLATE_IMGSZ", "640"))
    conf_thres: float = float(_env("PLATE_CONF", "0.25"))
    iou_thres: float = float(_env("PLATE_IOU", "0.45"))
    max_det: int = int(_env("PLATE_MAX_DET", "10"))

    # Preprocessing
    max_side: int = int(_env("PLATE_MAX_SIDE", "1280"))  # resize for speed
    clahe_clip_limit: float = float(_env("PLATE_CLAHE_CLIP", "2.0"))
    clahe_tile_grid_size: int = int(_env("PLATE_CLAHE_TILE", "8"))
    brightness_low: float = float(_env("PLATE_BRIGHT_LOW", "0.35"))
    brightness_high: float = float(_env("PLATE_BRIGHT_HIGH", "0.70"))
    gamma_dark: float = float(_env("PLATE_GAMMA_DARK", "1.6"))
    gamma_bright: float = float(_env("PLATE_GAMMA_BRIGHT", "0.85"))

    # Cropping
    crop_padding_ratio: float = float(_env("PLATE_CROP_PAD", "0.08"))
    min_plate_area: int = int(_env("PLATE_MIN_AREA", "400"))  # reject tiny crops

    # OCR
    easyocr_langs: tuple[str, ...] = ("en",)
    tesseract_cmd: str = _env("TESSERACT_CMD", "")  # optional explicit path to tesseract.exe
    ocr_upscale: int = int(_env("PLATE_OCR_UPSCALE", "2"))

    # Output / debug
    debug: bool = _env("PLATE_DEBUG", "0") == "1"
    save_crops: bool = _env("PLATE_SAVE_CROPS", "0") == "1"
    crops_dir: Path = Path(_env("PLATE_CROPS_DIR", str(Path("output") / "crops")))


def load_config() -> AppConfig:
    return AppConfig()

