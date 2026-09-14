from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR, TextRecognition

MAX_OCR_PIXELS = 1_500_000
MAX_OCR_SIDE = 1600
REFINE_BELOW = 0.96
MAX_REFINE_CROPS = 96
MAX_PADDLE_PASSES = 2
TESSERACT_TIMEOUT_SECONDS = 25
TESSDATA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main"
TESSDATA_LANGS = ("fas", "eng")


@dataclass(frozen=True)
class OCRResult:
    text: str
    average_confidence: float
    lines: tuple[tuple[str, float], ...]
    pass_name: str = "default"
    elapsed_seconds: float = 0.0
    layout_text: str = ""


def _find_payload(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if "rec_texts" in node and "rec_scores" in node:
            return node
        for value in node.values():
            found = _find_payload(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _find_payload(value)
            if found is not None:
                return found
    return None


def _find_single_rec_payload(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if "rec_text" in node and "rec_score" in node:
            return node
        for value in node.values():
            found = _find_single_rec_payload(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _find_single_rec_payload(value)
            if found is not None:
                return found
    return None


def _normalize_inference_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر OCR معتبر نیست.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3:
        raise ValueError("ساختار تصویر OCR پشتیبانی نمی‌شود.")
    else:
        channels = image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif channels != 3:
            raise ValueError(f"تعداد کانال‌های OCR پشتیبانی نمی‌شود: {channels}")

    height, width = image.shape[:2]
    pixels = height * width
    side_scale = min(1.0, MAX_OCR_SIDE / max(height, width))
    pixel_scale = min(1.0, (MAX_OCR_PIXELS / max(1, pixels)) ** 0.5)
    scale = min(side_scale, pixel_scale)

    if scale < 0.999:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    return image


def _boxes(payload: dict[str, Any], count: int) -> np.ndarray | None:
    raw = payload.get("rec_boxes")
    if raw is None:
        return None
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 4:
        return None
    return array[:, :4]


def _group_rows(array: np.ndarray) -> list[list[int]]:
    heights = np.maximum(1.0, array[:, 3] - array[:, 1])
    median_height = float(np.median(heights)) if len(array) else 1.0
    tolerance = max(6.0, median_height * 0.58)

    items = []
    for index, box in enumerate(array):
        center_y = float((box[1] + box[3]) / 2.0)
        items.append((center_y, index))
    items.sort(key=lambda item: item[0])

    rows: list[list[int]] = []
    centers: list[float] = []
    for center_y, index in items:
        best_row = None
        best_distance = float("inf")
        for row_index, row_center in enumerate(centers):
            distance = abs(center_y - row_center)
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_row = row_index
        if best_row is None:
            rows.append([index])
            centers.append(center_y)
        else:
            rows[best_row].append(index)
            members = rows[best_row]
            centers[best_row] = float(
                np.mean([(array[i, 1] + array[i, 3]) / 2.0 for i in members])
            )

    ordered_rows = sorted(zip(centers, rows, strict=True), key=lambda item: item[0])
    return [row for _, row in ordered_rows]


def _layout_text(payload: dict[str, Any], texts: list[str]) -> str:
    array = _boxes(payload, len(texts))
    if array is None or not texts:
        return "\n".join(text.strip() for text in texts if str(text).strip())

    rendered: list[str] = []
    for row_indices in _group_rows(array):
        row_indices.sort(key=lambda index: -float(max(array[index, 0], array[index, 2])))
        cells = [str(texts[index]).strip() for index in row_indices]
        cells = [cell for cell in cells if cell]
        if cells:
            rendered.append("\t".join(cells))
    return "\n".join(rendered)


def _prepare_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box[:4]]
    pad_x = max(3, int((x2 - x1) * 0.07))
    pad_y = max(3, int((y2 - y1) * 0.22))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_height, crop_width = crop.shape[:2]
    target_height = 96
    scale = max(1.0, target_height / max(1, crop_height))
    if scale > 1.01:
        crop = cv2.resize(
            crop,
            (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 50.0:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(
            lab[:, :, 0]
        )
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return cv2.copyMakeBorder(
        crop,
        8,
        8,
        12,
        12,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


@lru_cache(maxsize=2)
def get_ocr(device: str = "cpu") -> PaddleOCR:
    return PaddleOCR(
        device=device,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


@lru_cache(maxsize=2)
def get_crop_recognizer(device: str = "cpu") -> TextRecognition:
    return TextRecognition(
        model_name="arabic_PP-OCRv5_mobile_rec",
        device=device,
    )


def warmup(device: str = "cpu") -> None:
    started = perf_counter()
    get_ocr(device)
    print(f"[OCR] Paddle model ready elapsed={perf_counter() - started:.2f}s", flush=True)


def _refine_texts(
    image: np.ndarray,
    payload: dict[str, Any],
    texts: list[str],
    scores: list[float],
    device: str,
) -> tuple[list[str], list[float]]:
    array = _boxes(payload, len(texts))
    if array is None or not texts:
        return texts, scores

    selected: list[int] = []
    crops: list[np.ndarray] = []
    for index, score in enumerate(scores):
        if len(selected) >= MAX_REFINE_CROPS:
            break
        text = texts[index].strip()
        if float(score) >= REFINE_BELOW and len(text) > 3:
            continue
        crop = _prepare_crop(image, array[index])
        if crop is None:
            continue
        selected.append(index)
        crops.append(crop)

    if not crops:
        return texts, scores

    started = perf_counter()
    print(f"[OCR] crop-refine start count={len(crops)}", flush=True)
    try:
        refined_results = list(
            get_crop_recognizer(device).predict(
                input=crops,
                batch_size=min(16, len(crops)),
            )
        )
    except Exception as exc:
        print(f"[OCR] crop-refine skipped reason={exc}", flush=True)
        return texts, scores

    refined_count = 0
    for index, result in zip(selected, refined_results, strict=False):
        payload_single = _find_single_rec_payload(result.json)
        if not payload_single:
            continue
        candidate = str(payload_single.get("rec_text") or "").strip()
        candidate_score = float(payload_single.get("rec_score") or 0.0)
        if not candidate:
            continue
        original = texts[index].strip()
        original_score = float(scores[index]) if index < len(scores) else 0.0

        should_replace = (
            candidate_score >= original_score + 0.01
            or (
                original_score < 0.78
                and candidate_score >= original_score - 0.02
                and len(candidate) >= max(1, len(original) - 1)
            )
        )
        if should_replace:
            texts[index] = candidate
            scores[index] = candidate_score
            refined_count += 1

    print(
        f"[OCR] crop-refine done replaced={refined_count} "
        f"elapsed={perf_counter() - started:.2f}s",
        flush=True,
    )
    return texts, scores


def recognize(
    image: np.ndarray,
    device: str = "cpu",
    pass_name: str = "default",
) -> OCRResult:
    started = perf_counter()
    inference_image = _normalize_inference_image(image)
    height, width = inference_image.shape[:2]
    print(f"[OCR] Paddle start pass={pass_name} size={width}x{height}", flush=True)

    results = get_ocr(device).predict(inference_image)
    lines: list[tuple[str, float]] = []
    layout_parts: list[str] = []

    for result in results:
        payload = _find_payload(result.json)
        if not payload:
            continue
        texts = [str(value) for value in (payload.get("rec_texts") or [])]
        raw_scores = payload.get("rec_scores") or []
        scores = [
            float(raw_scores[index]) if index < len(raw_scores) else 0.0
            for index in range(len(texts))
        ]
        texts, scores = _refine_texts(
            inference_image,
            payload,
            texts,
            scores,
            device,
        )

        array = _boxes(payload, len(texts))
        if array is None:
            for index, text in enumerate(texts):
                cleaned = text.strip()
                if cleaned:
                    lines.append((cleaned, scores[index]))
        else:
            for row_indices in _group_rows(array):
                row_indices.sort(
                    key=lambda index: -float(max(array[index, 0], array[index, 2]))
                )
                row_cells = [texts[index].strip() for index in row_indices]
                row_cells = [cell for cell in row_cells if cell]
                if row_cells:
                    row_text = "\t".join(row_cells)
                    row_scores = [scores[index] for index in row_indices]
                    lines.append((row_text, float(np.mean(row_scores))))

        spatial = _layout_text(payload, texts)
        if spatial:
            layout_parts.append(spatial)

    average = sum(score for _, score in lines) / len(lines) if lines else 0.0
    elapsed = perf_counter() - started
    plain_text = "\n".join(text for text, _ in lines)
    layout_text = "\n".join(layout_parts).strip() or plain_text
    print(
        f"[OCR] Paddle done pass={pass_name} lines={len(lines)} "
        f"confidence={average:.4f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return OCRResult(
        text=plain_text,
        average_confidence=average,
        lines=tuple(lines),
        pass_name=pass_name,
        elapsed_seconds=elapsed,
        layout_text=layout_text,
    )


def _tesseract_executable() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _tessdata_root() -> Path:
    root = Path.home() / ".cache" / "daqiqkhan" / "tessdata"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_tessdata() -> Path | None:
    root = _tessdata_root()
    try:
        for lang in TESSDATA_LANGS:
            target = root / f"{lang}.traineddata"
            if target.is_file() and target.stat().st_size > 100_000:
                continue
            tmp = target.with_suffix(".download")
            tmp.unlink(missing_ok=True)
            print(f"[OCR] downloading Tesseract language={lang}", flush=True)
            urllib.request.urlretrieve(
                f"{TESSDATA_URL}/{lang}.traineddata",
                tmp,
            )
            if not tmp.is_file() or tmp.stat().st_size <= 100_000:
                tmp.unlink(missing_ok=True)
                return None
            tmp.replace(target)
    except Exception as exc:
        print(f"[OCR] Tesseract language data unavailable reason={exc}", flush=True)
        return None
    return root


def _tesseract_once(image: np.ndarray, psm: int) -> OCRResult | None:
    executable = _tesseract_executable()
    if not executable:
        return None
    tessdata = _ensure_tessdata()
    if tessdata is None:
        return None

    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="daqiqkhan-tesseract-") as temp_dir:
        image_path = Path(temp_dir) / "input.png"
        if not cv2.imwrite(str(image_path), image):
            return None
        command = [
            executable,
            str(image_path),
            "stdout",
            "--tessdata-dir",
            str(tessdata),
            "-l",
            "fas+eng",
            "--oem",
            "1",
            "--psm",
            str(psm),
            "tsv",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TESSERACT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[OCR] Tesseract skipped reason={exc}", flush=True)
            return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    texts: list[str] = []
    scores: list[float] = []
    boxes: list[list[float]] = []
    reader = csv.DictReader(io.StringIO(completed.stdout), delimiter="\t")
    for row in reader:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence_raw = float(row.get("conf") or -1)
            left = float(row.get("left") or 0)
            top = float(row.get("top") or 0)
            width = float(row.get("width") or 0)
            height = float(row.get("height") or 0)
        except ValueError:
            continue
        if confidence_raw < 0 or width <= 0 or height <= 0:
            continue
        texts.append(text)
        scores.append(max(0.0, min(1.0, confidence_raw / 100.0)))
        boxes.append([left, top, left + width, top + height])

    if not texts:
        return None

    payload = {"rec_boxes": boxes}
    array = np.asarray(boxes, dtype=float)
    lines: list[tuple[str, float]] = []
    for row_indices in _group_rows(array):
        row_indices.sort(key=lambda index: -float(max(array[index, 0], array[index, 2])))
        row_texts = [texts[index] for index in row_indices if texts[index]]
        if not row_texts:
            continue
        row_score = float(np.mean([scores[index] for index in row_indices]))
        lines.append(("\t".join(row_texts), row_score))

    layout_text = _layout_text(payload, texts)
    average = sum(score for _, score in lines) / len(lines) if lines else 0.0
    elapsed = perf_counter() - started
    print(
        f"[OCR] Tesseract psm={psm} lines={len(lines)} "
        f"confidence={average:.4f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return OCRResult(
        text="\n".join(text for text, _ in lines),
        average_confidence=average,
        lines=tuple(lines),
        pass_name=f"tesseract-psm{psm}",
        elapsed_seconds=elapsed,
        layout_text=layout_text,
    )


def _script_signal(text: str) -> float:
    compact = [char for char in text if not char.isspace() and char not in "\t|_-—–.,:;،؛()[]{}"]
    if not compact:
        return 0.0
    useful = 0
    for char in compact:
        code = ord(char)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
            or char.isdigit()
            or ("A" <= char <= "Z")
            or ("a" <= char <= "z")
        ):
            useful += 1
    return useful / len(compact)


def _quality_key(result: OCRResult) -> tuple[float, float, int, int]:
    useful_chars = sum(1 for char in result.text if not char.isspace())
    confident_lines = sum(1 for _, score in result.lines if score >= 0.60)
    script_signal = _script_signal(result.text)
    composite = (
        result.average_confidence * 0.62
        + script_signal * 0.28
        + min(1.0, useful_chars / 80.0) * 0.10
    )
    return (round(composite, 5), round(result.average_confidence, 5), confident_lines, useful_chars)


def _tesseract_candidates(image: np.ndarray) -> list[OCRResult]:
    if not _tesseract_executable():
        print("[OCR] Tesseract not installed; ensemble uses Paddle only", flush=True)
        return []
    results: list[OCRResult] = []
    first = _tesseract_once(image, psm=6)
    if first is not None:
        results.append(first)
    if first is None or len(first.lines) < 4 or first.average_confidence < 0.60:
        sparse = _tesseract_once(image, psm=11)
        if sparse is not None:
            results.append(sparse)
    return results


def recognize_best(
    candidates: list[tuple[str, np.ndarray]],
    device: str = "cpu",
) -> OCRResult:
    if not candidates:
        raise ValueError("هیچ ورودی OCR برای ارزیابی وجود ندارد.")

    total_started = perf_counter()
    results: list[OCRResult] = []

    for pass_name, image in candidates[:MAX_PADDLE_PASSES]:
        results.append(recognize(image, device=device, pass_name=pass_name))

    results.extend(_tesseract_candidates(_normalize_inference_image(candidates[0][1])))

    best = max(results, key=_quality_key)
    print(
        f"[OCR] selected engine={best.pass_name} "
        f"score={_quality_key(best)[0]:.4f} "
        f"total={perf_counter() - total_started:.2f}s",
        flush=True,
    )
    return best
