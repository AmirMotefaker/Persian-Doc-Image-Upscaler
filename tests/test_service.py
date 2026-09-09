from pathlib import Path

import numpy as np
from PIL import Image

from persian_upscaler import service
from persian_upscaler.ocr import OCRResult


def test_process_image_grayscale_file_end_to_end(tmp_path, monkeypatch):
    source = tmp_path / "grayscale.png"
    Image.fromarray(np.full((48, 96), 235, dtype=np.uint8), mode="L").save(source)

    monkeypatch.setattr(
        service,
        "recognize_best",
        lambda candidates: OCRResult(
            text="متن آزمایشی فارسی",
            average_confidence=0.98,
            lines=(("متن آزمایشی فارسی", 0.98),),
            pass_name="adaptive",
        ),
    )

    enhanced, ocr_preview, summary, text_file = service.process_image(
        str(source),
        profile="سند",
        scale=2.0,
        language="fa",
    )

    assert Path(enhanced).is_file()
    assert Path(ocr_preview).is_file()
    assert Path(text_file).read_text(encoding="utf-8") == "متن آزمایشی فارسی"
    assert "98.00%" in summary
    assert "adaptive" in summary
    assert "متن آزمایشی فارسی" in summary
