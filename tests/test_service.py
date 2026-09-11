from pathlib import Path

import cv2
import numpy as np

from persian_upscaler import service
from persian_upscaler.ocr import OCRResult


def test_process_image_grayscale_file_end_to_end(tmp_path, monkeypatch):
    source = tmp_path / "grayscale.png"
    assert cv2.imwrite(str(source), np.full((48, 96), 235, dtype=np.uint8))

    monkeypatch.setattr(
        service,
        "recognize_best",
        lambda candidates: OCRResult(
            text="ستون دوم\nستون اول",
            layout_text="ستون دوم\tستون اول",
            average_confidence=0.98,
            lines=(("ستون دوم", 0.98), ("ستون اول", 0.98)),
            pass_name="adaptive",
        ),
    )
    monkeypatch.setattr(
        service,
        "super_resolve_visual",
        lambda image, scale=2.0, profile="سند": np.repeat(
            np.repeat(image, 2, axis=0),
            2,
            axis=1,
        ),
    )

    enhanced, ocr_preview, display_text, text_file = service.process_image(
        str(source),
        profile="سند",
        scale=2.0,
        language="fa",
    )

    assert Path(enhanced).is_file()
    assert Path(ocr_preview).is_file()
    assert display_text == "ستون دوم\tستون اول"
    assert Path(text_file).read_text(encoding="utf-8") == display_text
