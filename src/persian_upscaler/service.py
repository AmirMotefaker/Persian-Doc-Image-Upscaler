from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from .enhancement import build_ocr_candidates, prepare_for_ocr, restore_visual
from .io import load_image, save_image, save_png
from .ocr import recognize_best


def _stage(name_fa: str, name_en: str, language: str, fn):
    try:
        return fn()
    except Exception as exc:
        if language == "en":
            raise RuntimeError(f"{name_en} failed: {exc}") from exc
        raise RuntimeError(f"مرحله «{name_fa}» ناموفق بود: {exc}") from exc


def process_image(
    file_path: str,
    profile: str,
    scale: float = 2.0,
    language: str = "fa",
    output_format: str = "PNG",
    engine: str = "Text-Safe Pro",
) -> tuple[str, str, str, str]:
    image = _stage("بارگذاری تصویر", "Image loading", language, lambda: load_image(file_path))
    restored = _stage(
        "بهبود کیفیت",
        "Image enhancement",
        language,
        lambda: restore_visual(
            image,
            profile=profile,
            scale=scale,
            engine=engine,
        ),
    )
    ocr_input = _stage(
        "آماده‌سازی OCR",
        "OCR preprocessing",
        language,
        lambda: prepare_for_ocr(restored, profile=profile),
    )
    candidates = _stage(
        "ساخت نماهای OCR",
        "OCR candidate generation",
        language,
        lambda: build_ocr_candidates(restored, profile=profile),
    )
    result = _stage(
        "تشخیص متن فارسی",
        "Persian text recognition",
        language,
        lambda: recognize_best(candidates),
    )

    workdir = Path(tempfile.mkdtemp(prefix="persian-upscaler-"))
    restored_path = _stage(
        "ذخیره تصویر بهبودیافته",
        "Saving enhanced image",
        language,
        lambda: save_image(workdir, "enhanced", restored, output_format),
    )
    ocr_preview_path = _stage(
        "ذخیره نمای OCR",
        "Saving OCR preview",
        language,
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
                "selected_pass": result.pass_name,
                "engine": engine,
                "lines": [
                    {"text": text, "confidence": score}
                    for text, score in result.lines
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if language == "en":
        meta = (
            f"OCR confidence: {result.average_confidence * 100:.2f}%\n"
            f"Recognized lines: {len(result.lines)}\n"
            f"Selected OCR pass: {result.pass_name}\n"
            f"Enhancement engine: {engine}"
        )
    else:
        meta = (
            f"اطمینان OCR: {result.average_confidence * 100:.2f}%\n"
            f"تعداد خطوط: {len(result.lines)}\n"
            f"بهترین مسیر OCR: {result.pass_name}\n"
            f"موتور بهبود: {engine}"
        )

    summary = f"{result.text}\n\n{meta}"
    return restored_path, ocr_preview_path, summary, str(text_path)


def process_batch(
    file_paths: list[str],
    profile: str,
    scale: float = 2.0,
    language: str = "fa",
    output_format: str = "PNG",
    engine: str = "Text-Safe Pro",
) -> tuple[str, str]:
    if not file_paths:
        raise ValueError("هیچ فایلی برای پردازش گروهی انتخاب نشده است.")
    if len(file_paths) > 20:
        raise ValueError("حداکثر ۲۰ تصویر در هر پردازش گروهی پشتیبانی می‌شود.")

    batch_dir = Path(tempfile.mkdtemp(prefix="persian-upscaler-batch-"))
    zip_path = batch_dir / "daqiqkhan-batch.zip"
    report_lines: list[str] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, file_path in enumerate(file_paths, start=1):
            enhanced, _, summary, text_file = process_image(
                file_path,
                profile=profile,
                scale=scale,
                language=language,
                output_format=output_format,
                engine=engine,
            )
            source_name = Path(file_path).stem
            archive.write(
                enhanced,
                arcname=f"{index:02d}-{source_name}{Path(enhanced).suffix}",
            )
            archive.write(text_file, arcname=f"{index:02d}-{source_name}.txt")
            report_lines.append(f"{index}. {Path(file_path).name}\n{summary}")

    return str(zip_path), "\n\n".join(report_lines)
