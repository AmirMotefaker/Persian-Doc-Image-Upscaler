from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .enhancement import prepare_for_ocr, restore_visual
from .io import load_image, save_png
from .ocr import recognize


def _stage(name: str, fn):
    try:
        return fn()
    except Exception as exc:
        raise RuntimeError(f"مرحله «{name}» ناموفق بود: {exc}") from exc


def process_image(
    file_path: str,
    profile: str,
    scale: float = 2.0,
    language: str = "fa",
) -> tuple[str, str, str, str]:
    image = _stage("بارگذاری تصویر", lambda: load_image(file_path))
    restored = _stage(
        "بهبود کیفیت",
        lambda: restore_visual(image, profile=profile, scale=scale),
    )
    ocr_input = _stage(
        "آماده‌سازی OCR",
        lambda: prepare_for_ocr(restored, profile=profile),
    )
    result = _stage("تشخیص متن فارسی", lambda: recognize(ocr_input))

    workdir = Path(tempfile.mkdtemp(prefix="persian-upscaler-"))
    restored_path = _stage(
        "ذخیره تصویر بهبودیافته",
        lambda: save_png(workdir / "enhanced.png", restored),
    )
    ocr_preview_path = _stage(
        "ذخیره نمای OCR",
        lambda: save_png(workdir / "ocr-preprocessed.png", ocr_input),
    )
    text_path = workdir / "ocr.txt"
    json_path = workdir / "ocr.json"

    text_path.write_text(result.text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "text": result.text,
                "average_confidence": result.average_confidence,
                "lines": [{"text": text, "confidence": score} for text, score in result.lines],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if language == "en":
        meta = (
            f"Average OCR confidence: {result.average_confidence * 100:.2f}%\n"
            f"Recognized lines: {len(result.lines)}"
        )
    else:
        meta = (
            f"میانگین اطمینان OCR: {result.average_confidence * 100:.2f}%\n"
            f"تعداد خطوط شناسایی‌شده: {len(result.lines)}"
        )

    summary = f"{result.text}\n\n{meta}"
    return restored_path, ocr_preview_path, summary, str(text_path)
