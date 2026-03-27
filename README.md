# License Plate Detection + OCR (Phase 1)

This project detects license plates with **YOLOv8** and extracts plate text with **EasyOCR** (primary) and **Tesseract** (fallback). Results are visualized using an **OpenCV GUI**.

## Folder Structure

- `input/image_loader.py` - loads images from both datasets
- `detection/plate_detector.py` - YOLOv8 plate detection (with fallback)
- `ocr/plate_reader.py` - dual OCR pipeline (EasyOCR -> Tesseract)
- `utils/preprocessing.py` - resize + CLAHE + gamma + OCR preprocessing + plate cropping helpers
- `output/display.py` - OpenCV visualization (FPS + boxes + confidence + text)
- `main.py` - orchestration / GUI loop

## Install (Windows / CPU)

1. Install Python 3.9+ (make sure `python` and `pip` work in PowerShell).
2. Install dependencies:

```powershell
cd D:\VIT_ACADEMICS\MDP_SW\plate-system
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Install Tesseract on Windows

1. Download Tesseract OCR installer:
   - Recommended: from [UB Mannheim (official builds)](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install it (default path is commonly):
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Add to `PATH` (so `pytesseract` can find it):
   - Open **Environment Variables**
   - Edit **Path** -> **New**:
     - `C:\Program Files\Tesseract-OCR`
4. Restart PowerShell and verify:

```powershell
where tesseract
```

If `where tesseract` finds `tesseract.exe`, you are good.

Optional: If your `tesseract.exe` is elsewhere, set an env var before running:

```powershell
$env:TESSERACT_CMD="C:\path\to\tesseract.exe"
```

## Run

```powershell
cd D:\VIT_ACADEMICS\MDP_SW\plate-system
python main.py
```

Useful flags:

- `--debug` / `--no-debug`
- `--save-crops` / `--no-save-crops`
- `--max-images N`

Keyboard during the GUI:

- `d` toggle debug logs
- `s` toggle crop saving
- `q` or `Esc` quit

## Model Notes

- `PLATE_MODEL_PATH` (optional) points to a plate-specific YOLOv8 `.pt` weight.
- If that model cannot be loaded, the system falls back to `yolov8n.pt` + heuristic plate-candidate search inside detected vehicle regions.

