from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from persian_upscaler.restoration.registry import AVAILABLE_ENGINES

FIXTURE = Path(r"C:\Market\گواهی سپرده مس کاتد.png")


def read_image_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def main() -> None:
    report: dict[str, object] = {
        "phase": "p2-foundation",
        "fixture": str(FIXTURE),
        "engines": [
            {
                "key": engine.key,
                "status": engine.status,
                "default_allowed": engine.can_be_default,
            }
            for engine in AVAILABLE_ENGINES
        ],
    }

    if not FIXTURE.exists():
        report["status"] = "fixture-missing"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    image = read_image_unicode(FIXTURE)
    report.update(
        {
            "status": "foundation-ready",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "source_sharpness": round(sharpness(image), 4),
            "source_contrast": round(contrast(image), 4),
            "next_step": "implement isolated candidate runners and compare OCR fidelity",
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
