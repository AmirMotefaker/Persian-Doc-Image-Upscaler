from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from persian_upscaler.ocr import recognize


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    canvas = np.full((420, 1600, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Persian OCR smoke fixture",
        (60, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # This smoke test validates model initialization, inference and result parsing.
    # Accuracy for Persian glyphs is benchmarked separately with real Persian fixtures.
    result = recognize(canvas, device="cpu")

    print("OCR runtime smoke: PASS")
    print(f"Recognized lines: {len(result.lines)}")
    print(f"Average confidence: {result.average_confidence:.4f}")
    if result.text:
        print("Text:")
        print(result.text)


if __name__ == "__main__":
    main()
