---
title: Persian Doc & Image Upscaler
emoji: 🇮🇷
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: 3.12.12
app_file: app.py
---

# Persian Doc & Image Upscaler + OCR

A Persian-first, open-source image enhancement and OCR platform focused on preserving Persian character geometry while improving readability and recognition quality.

## Current P0 scope

- Persian OCR using PaddleOCR 3.x with PP-OCRv5 multilingual recognition (`lang="fa"`).
- Non-generative image enhancement to reduce the risk of changing Persian dots, teeth, and character shapes.
- Separate visual-restoration and OCR-preprocessing pipelines.
- File-based input for PNG, JPG/JPEG, WebP, BMP, TIFF/TIF.
- OCR text output with average confidence.
- Safe per-request temporary output paths.
- Gradio UI, local Windows bootstrap, CI, Docker fallback, and Hugging Face-ready metadata.

> P0 intentionally does not use Real-ESRGAN in the canonical text path. Generative/super-resolution models will only return as an optional profile after Persian CER/WER and glyph-preservation benchmarks prove they improve the product.

## Architecture

```text
Image file
   |
   +--> Decode + EXIF normalization + resource validation
   |
   +--> Non-generative visual restoration --------> enhanced image
   |
   +--> Dedicated OCR preprocessing
              |
              +--> PaddleOCR / PP-OCRv5 / fa
                        |
                        +--> Persian text + confidence
```

## Supported image types

`PNG` · `JPG` · `JPEG` · `WebP` · `BMP` · `TIFF` · `TIF`

## Windows / PowerShell

No manual file editing is required. From PowerShell:

```powershell
irm https://raw.githubusercontent.com/AmirMotefaker/Persian-Doc-Image-Upscaler/feat/p0-platform-rebuild/scripts/bootstrap.ps1 -OutFile "$HOME\Downloads\persian-upscaler-bootstrap.ps1"
pwsh -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\persian-upscaler-bootstrap.ps1"
```

Then start the app:

```powershell
& "C:\Project\Persian-Doc-Image-Upscaler\.venv\Scripts\python.exe" "C:\Project\Persian-Doc-Image-Upscaler\app.py"
```

## Development gates

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py src
```

## Roadmap

- P0: stable image enhancement + Persian OCR foundation
- P1: real Persian benchmark corpus and CER/WER baselines
- P2: optional SR engines benchmarked against non-generative baseline
- P3: batch processing, downloadable ZIP/JSON, OCR overlays
- P4: PDF/document workflows after the image pipeline is proven
- P5: Hugging Face ZeroGPU acceleration where it measurably improves quality or latency

## Quality policy

Claims such as “preserves Persian dots” are treated as hypotheses until measured against a versioned Persian benchmark dataset. The project will publish benchmark inputs, ground-truth text, CER/WER results, and regression thresholds.

## License

Apache-2.0 planned for the P0 foundation.
