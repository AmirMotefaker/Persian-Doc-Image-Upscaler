from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR, TextRecognition

MAX_OCR_PIXELS = 1_500_000
MAX_OCR_SIDE = 1600
FALLBACK_CONFIDENCE = 0.72
FALLBACK_MIN_LINES = 4
REFINE_BELOW = 0.94
MAX_REFINE_CROPS = 96


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


def _ordered_indices(payload: dict[str, Any], count: int) -> list[int]:
    array = _boxes(payload, count)
    if array is None:
        return list(range(count))

    heights = np.maximum(1.0, array[:, 3] - array[:, 1])
    median_height = float(np.median(heights)) if count else 1.0
    line_quantum = max(8.0, median_height * 0.65)

    keyed: list[tuple[float, float, int]] = []
    for index, box in enumerate(array):
        center_y = float((box[1] + box[3]) / 2.0)
        right_x = float(max(box[0], box[2]))
        line_bucket = round(center_y / line_quantum)
        keyed.append((line_bucket, -right_x, index))

    return [item[2] for item in sorted(keyed)]


def _group_rows(array: np.ndarray) -> list[list[int]]:
    heights = np.maximum(1.0, array[:, 3] - array[:, 1])
    median_height = float(np.median(heights)) if len(array) else 1.0
    tolerance = max(7.0, median_height * 0.58)

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

    ordered_rows = sorted(
        zip(centers, rows, strict=True),
        key=lambda item: item[0],
    )
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
    pad_x = max(2, int((x2 - x1) * 0.05))
    pad_y = max(2, int((y2 - y1) * 0.18))
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
    target_height = 72
    scale = max(1.0, target_height / max(1, crop_height))
    if scale > 1.01:
        crop = cv2.resize(
            crop,
            (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 45.0:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(4, 4),
        ).apply(lab[:, :, 0])
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return cv2.copyMakeBorder(
        crop,
        6,
        6,
        8,
        8,
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
    print(f"[OCR] model ready elapsed={perf_counter() - started:.2f}s", flush=True)


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
        if float(score) >= REFINE_BELOW and len(texts[index].strip()) > 2:
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
            candidate_score >= original_score + 0.012
            or (
                original_score < 0.72
                and candidate_score >= original_score - 0.015
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
    print(f"[OCR] start pass={pass_name} size={width}x{height}", flush=True)

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

        order = _ordered_indices(payload, len(texts))
        for index in order:
            cleaned = texts[index].strip()
            if not cleaned:
                continue
            lines.append((cleaned, scores[index]))

        spatial = _layout_text(payload, texts)
        if spatial:
            layout_parts.append(spatial)

    average = sum(score for _, score in lines) / len(lines) if lines else 0.0
    elapsed = perf_counter() - started
    print(
        f"[OCR] done pass={pass_name} lines={len(lines)} "
        f"confidence={average:.4f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    plain_text = "\n".join(text for text, _ in lines)
    layout_text = "\n".join(layout_parts).strip() or plain_text
    return OCRResult(
        text=plain_text,
        average_confidence=average,
        lines=tuple(lines),
        pass_name=pass_name,
        elapsed_seconds=elapsed,
        layout_text=layout_text,
    )


def _quality_key(result: OCRResult) -> tuple[float, int, int]:
    useful_chars = sum(1 for char in result.text if not char.isspace())
    confident_lines = sum(1 for _, score in result.lines if score >= 0.65)
    return (round(result.average_confidence, 5), confident_lines, useful_chars)


def _needs_fallback(result: OCRResult) -> bool:
    return (
        result.average_confidence < FALLBACK_CONFIDENCE
        or len(result.lines) < FALLBACK_MIN_LINES
    )


def recognize_best(
    candidates: list[tuple[str, np.ndarray]],
    device: str = "cpu",
) -> OCRResult:
    if not candidates:
        raise ValueError("هیچ ورودی OCR برای ارزیابی وجود ندارد.")

    total_started = perf_counter()
    first_name, first_image = candidates[0]
    results = [recognize(first_image, device=device, pass_name=first_name)]

    if len(candidates) > 1 and _needs_fallback(results[0]):
        second_name, second_image = candidates[1]
        print(
            f"[OCR] fallback enabled confidence={results[0].average_confidence:.4f} "
            f"lines={len(results[0].lines)}",
            flush=True,
        )
        results.append(recognize(second_image, device=device, pass_name=second_name))
    elif len(candidates) > 1:
        print("[OCR] fallback skipped", flush=True)

    best = max(results, key=_quality_key)
    print(
        f"[OCR] selected pass={best.pass_name} total={perf_counter() - total_started:.2f}s",
        flush=True,
    )
    return best
