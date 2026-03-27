from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Ensure local imports work even if run from other directories.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from detection.plate_detector import PlateDetector
from input.image_loader import ImageLoader
from ocr.plate_reader import PlateReader
from output.display import Visualizer
from utils.config import load_config
from utils.preprocessing import (
    adaptive_enhance,
    crop_with_padding,
    ocr_preprocess_attempts,
    resize_max_side,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="License plate detection + OCR (YOLOv8 + EasyOCR/Tesseract).")
    p.add_argument("--debug", action="store_true", help="Enable debug logs.")
    p.add_argument("--no-debug", action="store_true", help="Disable debug logs.")
    p.add_argument("--save-crops", action="store_true", help="Save plate crops to disk.")
    p.add_argument("--no-save-crops", action="store_true", help="Do not save plate crops.")
    p.add_argument("--window-name", type=str, default="License Plate Detection (OCR)")
    p.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    debug = cfg.debug
    if args.debug:
        debug = True
    if args.no_debug:
        debug = False

    save_crops = cfg.save_crops
    if args.save_crops:
        save_crops = True
    if args.no_save_crops:
        save_crops = False

    image_roots = [cfg.dataset_dir_1, cfg.dataset_dir_2]
    loader = ImageLoader(image_roots)
    items = loader.iter_images()
    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]

    if not items:
        raise RuntimeError(
            "No images found. Check dataset directories:\n"
            f"- {cfg.dataset_dir_1}\n"
            f"- {cfg.dataset_dir_2}"
        )

    detector = PlateDetector(
        plate_model_path=cfg.plate_model_path,
        general_model_name=cfg.general_model_name,
        device=cfg.device,
        imgsz=cfg.imgsz,
        conf_thres=cfg.conf_thres,
        iou_thres=cfg.iou_thres,
        max_det=cfg.max_det,
        debug=debug,
    )
    reader = PlateReader(langs=cfg.easyocr_langs, tesseract_cmd=cfg.tesseract_cmd)
    viz = Visualizer()

    cfg.crops_dir.mkdir(parents=True, exist_ok=True)
    output_dir = THIS_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"plate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    detected_lines: list[str] = []
    undetected_images: list[str] = []
    processed_count = 0

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    fps_ema = 0.0
    alpha = 0.2

    for idx, item in enumerate(items):
        img_bgr = cv2.imread(str(item.path))
        if img_bgr is None:
            if debug:
                print(f"[WARN] Could not read image: {item.path}")
            undetected_images.append(f"{item.path.name} | reason=read_failed")
            continue

        processed_count += 1
        t0 = time.perf_counter()

        resized = resize_max_side(img_bgr, max_side=cfg.max_side)
        enhanced = adaptive_enhance(
            resized.image,
            brightness_low=cfg.brightness_low,
            brightness_high=cfg.brightness_high,
            gamma_dark=cfg.gamma_dark,
            gamma_bright=cfg.gamma_bright,
            clahe_clip_limit=cfg.clahe_clip_limit,
            clahe_tile_grid_size=cfg.clahe_tile_grid_size,
        )

        detections = detector.detect(enhanced)
        if debug:
            print(f"\n[{idx+1}/{len(items)}] Image: {item.path.name} ({item.source})")
            print(f"[DEBUG] Plates detected: {len(detections)}")

        ocr_outputs = []
        image_detected_texts: list[str] = []
        for det_i, det in enumerate(detections):
            crop, crop_xyxy = crop_with_padding(
                enhanced,
                det.xyxy,
                pad_ratio=cfg.crop_padding_ratio,
            )

            # OCR on crop: enhance first, then generate multiple thresholding attempts.
            crop_enh = adaptive_enhance(
                crop,
                brightness_low=cfg.brightness_low,
                brightness_high=cfg.brightness_high,
                gamma_dark=cfg.gamma_dark,
                gamma_bright=cfg.gamma_bright,
                clahe_clip_limit=cfg.clahe_clip_limit,
                clahe_tile_grid_size=cfg.clahe_tile_grid_size,
            )
            attempts = ocr_preprocess_attempts(crop_enh, upscale_factor=cfg.ocr_upscale)

            ocr_result = reader.read_plate(attempts)
            plate_text = ocr_result.text
            ocr_conf = ocr_result.conf
            ocr_source = ocr_result.source

            if save_crops:
                # Save raw crop and enhanced crop (same region; helps debug OCR).
                crop_name = f"{item.path.stem}_plate{det_i}_{ocr_source}_{plate_text or 'NA'}.png"
                out_path = cfg.crops_dir / crop_name
                cv2.imwrite(str(out_path), crop)

            if debug:
                print(
                    f"[DEBUG] det#{det_i} det_conf={det.conf:.3f} "
                    f"bbox={det.xyxy} text='{plate_text}' ocr_conf={ocr_conf:.3f} src={ocr_source}"
                )

            ocr_outputs.append((det, plate_text, ocr_conf, ocr_source))
            if plate_text:
                image_detected_texts.append(plate_text)
                detected_lines.append(
                    (
                        f"{item.path.name} | plate={plate_text} | det_conf={det.conf:.3f} "
                        f"| ocr_conf={ocr_conf:.3f} | ocr_source={ocr_source}"
                    )
                )

        for det, plate_text, ocr_conf, ocr_source in ocr_outputs:
            viz.draw_plate_annotation(
                enhanced,
                bbox_xyxy=det.xyxy,
                det_conf=det.conf,
                plate_text=plate_text,
                ocr_conf=ocr_conf,
                ocr_source=ocr_source,
            )

        t1 = time.perf_counter()
        loop_fps = 1.0 / max(1e-6, (t1 - t0))
        fps_ema = loop_fps if fps_ema == 0.0 else (alpha * loop_fps + (1 - alpha) * fps_ema)
        viz.draw_fps(enhanced, fps_ema)

        cv2.imshow(args.window_name, enhanced)
        key = cv2.waitKey(1) & 0xFF

        if not image_detected_texts:
            reason = "no_plate_detected" if len(detections) == 0 else "plate_detected_but_ocr_empty"
            undetected_images.append(f"{item.path.name} | reason={reason}")

        # Keyboard controls:
        # - q / Esc: quit
        # - d: toggle debug logs
        # - s: toggle crop saving
        if key in (27, ord("q")):
            break
        if key == ord("d"):
            debug = not debug
            print(f"[INFO] Debug toggled -> {debug}")
        if key == ord("s"):
            save_crops = not save_crops
            print(f"[INFO] Save crops toggled -> {save_crops}")

    cv2.destroyAllWindows()

    with report_path.open("w", encoding="utf-8") as f:
        f.write("License Plate Detection + OCR Results\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Processed images: {processed_count}\n")
        f.write(f"Source folders: {cfg.dataset_dir_1} | {cfg.dataset_dir_2}\n")
        f.write("\n=== DETECTED NUMBER PLATES ===\n")
        if detected_lines:
            for line in detected_lines:
                f.write(f"{line}\n")
        else:
            f.write("None\n")

        f.write("\n=== UNDETECTED IMAGES (listed at end) ===\n")
        if undetected_images:
            for line in undetected_images:
                f.write(f"{line}\n")
        else:
            f.write("None\n")

    print(f"[INFO] Results file written: {report_path}")


if __name__ == "__main__":
    main()

