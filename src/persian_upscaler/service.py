from __future__ import annotations

import json
import tempfile
import traceback
import zipfile
from pathlib import Path

from .enhancement import build_ocr_candidates, prepare_for_ocr, restore_visual
from .io import load_image, save_image, save_png
from .ocr import recognize_best as recognize_best_legacy
from .ocr_bina import recognize_best as recognize_best_bina
from .ocr_vl import recognize_document as recognize_document_vl
from .super_resolution import super_resolve_visual

TEXT_PROFILES = {"سند", "اسکرین‌شات", "اسکن ضعیف"}


def _stage(name_fa: str, name_en: str, language: str, fn):
    try:
        return fn()
    except Exception as exc:
        if language == "en":
            raise RuntimeError(f"{name_en} failed: {exc}") from exc
        raise RuntimeError(f"مرحله «{name_fa}» ناموفق بود: {exc}") from exc


def _recognize_with_fallback(candidates):
    try:
        result = recognize_best_bina(candidates)
        if result.layout_text.strip() or result.text.strip():
            return result
        raise RuntimeError("Bina OCR returned no usable text")
    except Exception as exc:
        print(f"[OCR] Bina failed; fallback=legacy reason={exc}", flush=True)
        traceback.print_exc()
        result = recognize_best_legacy(candidates)
        print(f"[OCR] fallback selected engine={result.pass_name}", flush=True)
        return result


def recognize_best(candidates):
    """Stable service seam kept for tests and alternate OCR backends."""
    return _recognize_with_fallback(candidates)


def _recognize_document(image, candidates, profile):
    try:
        result = recognize_document_vl(image, profile=profile)
        if result.layout_text.strip() or result.text.strip():
            print(
                f"[OCR] primary selected engine={result.pass_name}",
                flush=True,
            )
            return result
        raise RuntimeError("PaddleOCR-VL-1.6 returned no usable text")
    except Exception as exc:
        print(f"[OCR] PaddleOCR-VL-1.6 failed reason={exc}", flush=True)
        traceback.print_exc()
        if profile in TEXT_PROFILES:
            raise RuntimeError(
                "OCR تخصصی فارسی آماده نیست؛ برای جلوگیری از نمایش متن اشتباه، "
                "fallback ضعیف غیرفعال شده است. سرویس PaddleOCR-VL را بررسی کنید."
            ) from exc
        return recognize_best(candidates)


def process_image(
    file_path: str,
    profile: str,
    scale: float = 2.0,
    language: str = "fa",
    output_format: str = "PNG",
    engine: str = "Super-Resolution Pro",
) -> tuple[str, str, str, str]:
    image = _stage(
        "بارگذاری تصویر",
        "Image loading",
        language,
        lambda: load_image(file_path),
    )

    ocr_restored = _stage(
        "بهبود امن متن برای OCR",
        "Text-safe OCR enhancement",
        language,
        lambda: restore_visual(
            image,
            profile=profile,
            scale=2.0,
            engine="Text-Safe Pro",
        ),
    )

    if engine == "Super-Resolution Pro":
        visual = _stage(
            "بهبود تصویر",
            "Visual enhancement",
            language,
            lambda: super_resolve_visual(
                image,
                scale=scale,
                profile=profile,
            ),
        )
    else:
        visual = _stage(
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
        lambda: prepare_for_ocr(ocr_restored, profile=profile),
    )
    candidates = _stage(
        "ساخت نماهای OCR",
        "OCR candidate generation",
        language,
        lambda: build_ocr_candidates(ocr_restored, profile=profile),
    )
    result = _stage(
        "تشخیص متن فارسی",
        "Persian text recognition",
        language,
        # VL gets the untouched source. Classical fallback candidates remain separate.
        lambda: _recognize_document(image, candidates, profile),
    )

    canonical_text = result.layout_text.strip() or result.text.strip()

    workdir = Path(tempfile.mkdtemp(prefix="persian-upscaler-"))
    visual_path = _stage(
        "ذخیره تصویر بهبودیافته",
        "Saving enhanced image",
        language,
        lambda: save_image(workdir, "enhanced", visual, output_format),
    )
    ocr_preview_path = _stage(
        "ذخیره نمای OCR",
        "Saving OCR preview",
        language,
        lambda: save_png(workdir / "ocr-preprocessed.png", ocr_input),
    )
    text_path = workdir / "ocr.txt"
    json_path = workdir / "ocr.json"

    text_path.write_text(canonical_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "text": canonical_text,
                "plain_text": result.text,
                "average_confidence": result.average_confidence,
                "selected_pass": result.pass_name,
                "visual_engine": engine,
                "profile": profile,
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

    return visual_path, ocr_preview_path, canonical_text, str(text_path)


def process_batch(
    file_paths: list[str],
    profile: str,
    scale: float = 2.0,
    language: str = "fa",
    output_format: str = "PNG",
    engine: str = "Super-Resolution Pro",
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
            enhanced, _, text, text_file = process_image(
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
            report_lines.append(f"{index}. {Path(file_path).name}\n{text}")

    return str(zip_path), "\n\n".join(report_lines)
