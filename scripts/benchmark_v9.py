from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sharpness(path: str) -> float:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _parse_confidence(summary: str) -> float:
    for line in summary.splitlines():
        if "%" not in line:
            continue
        digits = "".join(char for char in line if char.isdigit() or char == ".")
        if digits:
            try:
                return float(digits)
            except ValueError:
                pass
    return 0.0


def main() -> None:
    from app import SAMPLE_SCAN
    from persian_upscaler.service import process_image

    parser = argparse.ArgumentParser(description="Benchmark V9 enhancement engines")
    parser.add_argument("image", nargs="?", default=SAMPLE_SCAN)
    parser.add_argument("--profile", default="سند")
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    source = str(Path(args.image).resolve())
    if not Path(source).is_file():
        raise FileNotFoundError(source)

    print(f"V9 benchmark source: {source}")
    rows: list[tuple[str, float, float, float]] = []

    for engine in ("Standard", "Text-Safe Pro"):
        started = perf_counter()
        enhanced, _ocr_preview, summary, _text_file = process_image(
            source,
            profile=args.profile,
            scale=args.scale,
            language="fa",
            output_format="PNG",
            engine=engine,
        )
        elapsed = perf_counter() - started
        sharpness = _sharpness(enhanced)
        confidence = _parse_confidence(summary)
        rows.append((engine, elapsed, sharpness, confidence))
        print(
            f"{engine}: elapsed={elapsed:.2f}s "
            f"sharpness={sharpness:.2f} ocr_confidence={confidence:.2f}%"
        )

    standard = rows[0]
    pro = rows[1]
    print("\n=== V9 DELTA ===")
    print(f"Sharpness delta: {pro[2] - standard[2]:+.2f}")
    print(f"OCR confidence delta: {pro[3] - standard[3]:+.2f} pp")
    print(f"Latency delta: {pro[1] - standard[1]:+.2f}s")


if __name__ == "__main__":
    main()
