from pathlib import Path

import cv2
import numpy as np

from persian_upscaler import service
from persian_upscaler.ocr import OCRResult


def test_process_image_grayscale_file_end_to_end(tmp_path, monkeypatch):
    source = tmp_path / "grayscale.png"
    assert cv2.imwrite(str(source), np.full((48, 96), 235, dtype=np.uint8))

    seen_shape = None

    def _fake_vl(image, profile="سند"):
        nonlocal seen_shape
        seen_shape = image.shape
        return OCRResult(
            text="ستون دوم\nستون اول",
            layout_text="ستون دوم\tستون اول",
            average_confidence=0.99,
            lines=(("ستون دوم", 0.99), ("ستون اول", 0.99)),
            pass_name="paddleocr-vl-1.6:test",
        )

    monkeypatch.setattr(service, "recognize_document_vl", _fake_vl)
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
    assert seen_shape == (48, 96, 3)


def test_text_profile_refuses_inaccurate_fallback(tmp_path, monkeypatch):
    source = tmp_path / "doc.png"
    assert cv2.imwrite(str(source), np.full((32, 64, 3), 240, dtype=np.uint8))

    def _fail_vl(image, profile="سند"):
        raise RuntimeError("VL offline")

    monkeypatch.setattr(service, "recognize_document_vl", _fail_vl)
    monkeypatch.setattr(
        service,
        "super_resolve_visual",
        lambda image, scale=2.0, profile="سند": np.repeat(
            np.repeat(image, 2, axis=0),
            2,
            axis=1,
        ),
    )

    try:
        service.process_image(str(source), profile="سند", language="fa")
    except RuntimeError as exc:
        assert "fallback ضعیف غیرفعال" in str(exc)
    else:
        raise AssertionError("Persian text profiles must not silently fall back")
