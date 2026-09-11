from __future__ import annotations

import hashlib
import os
import re
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

from paddleocr import TextDetection, TextRecognition

BINA_REVISION = "2af6ae7eeb38d195d4d26b13467fccf0e27f37e4"
BINA_BASE_URL = (
    "https://huggingface.co/Reza2kn/Bina-0.2-Rizeh/resolve/"
    f"{BINA_REVISION}"
)
BINA_RECOGNIZER_FILES = {
    "inference.json": "b802ea6f182ffb341716b2f649b8f48a2c1f3e4fb6db26f4cfd31771d80cd8b8",
    "inference.pdiparams": "f8e1b5b4e9ef11b46c521fed7d1c9b7e7b2d08c4c879aa01353acd7308a6e9fb",
    "inference.yml": "650bfd52635c3a5680479df99a4c5cb3b94e2875b25a7bc3aea45ec1bad6a841",
}
BINA_DETECTOR_FILES = {
    "inference.json": "0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5",
    "inference.pdiparams": "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960",
    "inference.yml": "7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
}
_LTR_RUN = re.compile(r"[a-zA-Z0-9 :*./%+-]")
TARGET_LONG_SIDE = 1100
MAX_DETECTION_SIDE = 1600
MAX_LINES = 160
LOW_CONFIDENCE = 0.62


@dataclass(frozen=True)
class OCRResult:
    text: str
    average_confidence: float
    lines: tuple[tuple[str, float], ...]
    pass_name: str = "bina-tight-lines"
    elapsed_seconds: float = 0.0
    layout_text: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_root() -> Path:
    root = Path(
        os.environ.get(
            "DAQIQKHAN_BINA_MODEL_DIR",
            Path.home() / ".cache" / "daqiqkhan" / "bina-0.2-rizeh",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_component(component: str, files: dict[str, str]) -> Path:
    root = _runtime_root() / component
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha in files.items():
        target = root / filename
        if target.is_file() and _sha256(target) == expected_sha:
            continue
        tmp = target.with_suffix(target.suffix + ".download")
        tmp.unlink(missing_ok=True)
        print(
            f"[OCR] Bina download start component={component} file={filename}",
            flush=True,
        )
        urllib.request.urlretrieve(
            f"{BINA_BASE_URL}/{component}/{filename}?download=true",
            tmp,
        )
        if not tmp.is_file() or _sha256(tmp) != expected_sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Bina checksum failed: {component}/{filename}")
        tmp.replace(target)
    return root


def _ensure_bina_runtime() -> tuple[Path, Path]:
    recognizer = _ensure_component("inference", BINA_RECOGNIZER_FILES)
    detector = _ensure_component("detector", BINA_DETECTOR_FILES)
    return recognizer, detector


def _pred_reverse(text: str) -> str:
    """Official Bina visual-order to logical-order conversion."""
    segments: list[str] = []
    current_ltr = ""
    for character in text:
        if _LTR_RUN.search(character):
            current_ltr += character
            continue
        if current_ltr:
            segments.append(current_ltr)
            current_ltr = ""
        segments.append(character)
    if current_ltr:
        segments.append(current_ltr)
    return "".join(reversed(segments))


def _json_payload(result: Any) -> dict[str, Any]:
    payload = result.json() if callable(result.json) else result.json
    if isinstance(payload, dict):
        return payload.get("res", payload)
    return {}


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر OCR معتبر نیست.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("ساختار تصویر OCR پشتیبانی نمی‌شود.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError("تعداد کانال‌های تصویر OCR پشتیبانی نمی‌شود.")
    return image


def _scale_for_tiny_text(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= 0:
        return image
    scale = min(3.0, max(1.0, TARGET_LONG_SIDE / long_side))
    if scale <= 1.01:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def _remove_table_lines(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove long wired-table borders before text detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(24, width // 14), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, height // 14)),
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_pixels = int(np.count_nonzero(horizontal))
    vertical_pixels = int(np.count_nonzero(vertical))
    has_grid = horizontal_pixels > width * 1.4 and vertical_pixels > height * 1.4
    if not has_grid:
        return image, False

    grid = cv2.bitwise_or(horizontal, vertical)
    grid = cv2.dilate(
        grid,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    cleaned = cv2.inpaint(image, grid, 2.0, cv2.INPAINT_TELEA)
    print(
        f"[OCR] table-grid removed h={horizontal_pixels} v={vertical_pixels}",
        flush=True,
    )
    return cleaned, True


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        x, y, width, height = cv2.boundingRect(pts.astype(np.int32))
        return np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def _crop_quad(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    quad = _order_quad(points)
    tl, tr, br, bl = quad
    width = max(
        1,
        int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))),
    )
    height = max(
        1,
        int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))),
    )
    if width < 4 or height < 4:
        return None
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] > crop.shape[1] * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _prepare_line_crop(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    if height < 64:
        scale = min(4.0, 64 / max(1, height))
        crop = cv2.resize(
            crop,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 48.0:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=1.4,
            tileGridSize=(4, 4),
        ).apply(lightness)
        crop = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
    return cv2.copyMakeBorder(
        crop,
        5,
        5,
        8,
        8,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _bbox_from_poly(poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return np.array(
        [
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        ]
    )


def _group_rows(boxes: list[np.ndarray]) -> list[list[int]]:
    if not boxes:
        return []
    heights = np.array([max(1.0, box[3] - box[1]) for box in boxes])
    tolerance = max(8.0, float(np.median(heights)) * 0.62)
    items = sorted(
        (((box[1] + box[3]) / 2.0, index) for index, box in enumerate(boxes)),
        key=lambda item: item[0],
    )
    rows: list[list[int]] = []
    centers: list[float] = []
    for center_y, index in items:
        chosen = None
        distance = float("inf")
        for row_index, row_center in enumerate(centers):
            current = abs(center_y - row_center)
            if current <= tolerance and current < distance:
                chosen = row_index
                distance = current
        if chosen is None:
            rows.append([index])
            centers.append(float(center_y))
            continue
        rows[chosen].append(index)
        centers[chosen] = float(
            np.mean([(boxes[i][1] + boxes[i][3]) / 2.0 for i in rows[chosen]])
        )
    return [row for _, row in sorted(zip(centers, rows, strict=True))]


@lru_cache(maxsize=1)
def get_detector(device: str = "cpu") -> TextDetection:
    _recognizer_dir, detector_dir = _ensure_bina_runtime()
    print(
        f"[OCR] detector=PP-OCRv6-medium mode=det-only device={device}",
        flush=True,
    )
    return TextDetection(
        model_dir=str(detector_dir),
        device=device,
        enable_mkldnn=False,
    )


@lru_cache(maxsize=1)
def get_recognizer(device: str = "cpu") -> TextRecognition:
    recognizer_dir, _detector_dir = _ensure_bina_runtime()
    print(
        f"[OCR] recognizer=Bina-0.2-Rizeh mode=line-crops device={device}",
        flush=True,
    )
    return TextRecognition(
        model_dir=str(recognizer_dir),
        device=device,
    )


def _detect_lines(image: np.ndarray, device: str) -> tuple[list[np.ndarray], list[float]]:
    detected_polys: list[np.ndarray] = []
    detected_scores: list[float] = []
    for result in get_detector(device).predict(
        image,
        batch_size=1,
        limit_side_len=MAX_DETECTION_SIDE,
        thresh=0.20,
        box_thresh=0.35,
        unclip_ratio=1.35,
    ):
        payload = _json_payload(result)
        raw_polys = payload.get("dt_polys")
        raw_scores = payload.get("dt_scores")
        polys = [] if raw_polys is None else list(raw_polys)
        scores = [] if raw_scores is None else list(raw_scores)
        for index, poly in enumerate(polys):
            score = float(scores[index]) if index < len(scores) else 0.0
            box = _bbox_from_poly(poly)
            if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                continue
            detected_polys.append(np.asarray(poly, dtype=np.float32))
            detected_scores.append(score)
            if len(detected_polys) >= MAX_LINES:
                break
    return detected_polys, detected_scores


def _recognize_crops(
    crops: list[np.ndarray],
    device: str,
) -> tuple[list[str], list[float]]:
    if not crops:
        return [], []
    texts: list[str] = []
    scores: list[float] = []
    results = get_recognizer(device).predict(
        input=crops,
        batch_size=min(16, len(crops)),
    )
    for result in results:
        payload = _json_payload(result)
        visual = str(payload.get("rec_text") or "")
        texts.append(_pred_reverse(visual).strip())
        scores.append(float(payload.get("rec_score") or 0.0))
    return texts, scores


def _refine_low_confidence(
    crops: list[np.ndarray],
    texts: list[str],
    scores: list[float],
    device: str,
) -> tuple[list[str], list[float]]:
    indexes = [index for index, score in enumerate(scores) if score < LOW_CONFIDENCE]
    if not indexes:
        return texts, scores
    retry_crops: list[np.ndarray] = []
    for index in indexes:
        crop = crops[index]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=1.7,
            tileGridSize=(4, 4),
        ).apply(gray)
        retry_crops.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
    retry_texts, retry_scores = _recognize_crops(retry_crops, device)
    for position, index in enumerate(indexes):
        if position >= len(retry_scores):
            break
        if retry_scores[position] > scores[index] + 0.015:
            texts[index] = retry_texts[position]
            scores[index] = retry_scores[position]
    return texts, scores


def recognize(
    image: np.ndarray,
    device: str = "cpu",
    pass_name: str = "tight-lines",
) -> OCRResult:
    started = perf_counter()
    source = _scale_for_tiny_text(_ensure_bgr(image))
    detection_image, grid_removed = _remove_table_lines(source)
    print(
        f"[OCR] Bina tight-line start size={source.shape[1]}x{source.shape[0]} "
        f"grid={grid_removed}",
        flush=True,
    )

    polys, detection_scores = _detect_lines(detection_image, device)
    crops: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    kept_detection_scores: list[float] = []
    for poly, detection_score in zip(polys, detection_scores, strict=True):
        crop = _crop_quad(detection_image, poly)
        if crop is None:
            continue
        crops.append(_prepare_line_crop(crop))
        boxes.append(_bbox_from_poly(poly))
        kept_detection_scores.append(detection_score)

    texts, scores = _recognize_crops(crops, device)
    texts, scores = _refine_low_confidence(crops, texts, scores, device)

    count = min(len(texts), len(boxes), len(scores))
    texts = texts[:count]
    boxes = boxes[:count]
    scores = scores[:count]
    kept_detection_scores = kept_detection_scores[:count]

    rows: list[str] = []
    line_items: list[tuple[str, float]] = []
    for row_indices in _group_rows(boxes):
        row_indices.sort(
            key=lambda index: -float((boxes[index][0] + boxes[index][2]) / 2.0)
        )
        cells: list[str] = []
        cell_scores: list[float] = []
        for index in row_indices:
            text = texts[index].strip()
            if not text:
                continue
            cells.append(text)
            combined_score = scores[index] * 0.88 + kept_detection_scores[index] * 0.12
            cell_scores.append(combined_score)
        if not cells:
            continue
        row_text = "\t".join(cells)
        row_score = float(np.mean(cell_scores)) if cell_scores else 0.0
        rows.append(row_text)
        line_items.append((row_text, row_score))

    layout_text = "\n".join(rows).strip()
    average = (
        float(np.mean([score for _, score in line_items]))
        if line_items
        else 0.0
    )
    elapsed = perf_counter() - started
    for index, (row_text, row_score) in enumerate(line_items[:30], start=1):
        print(
            f"[OCR] row={index:02d} conf={row_score:.3f} text={row_text[:180]}",
            flush=True,
        )
    print(
        f"[OCR] Bina tight-line done boxes={len(boxes)} rows={len(rows)} "
        f"confidence={average:.4f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return OCRResult(
        text=layout_text,
        average_confidence=average,
        lines=tuple(line_items),
        pass_name=(
            "bina-tight-lines-grid" if grid_removed else "bina-tight-lines"
        ),
        elapsed_seconds=elapsed,
        layout_text=layout_text,
    )


def recognize_best(
    candidates: list[tuple[str, np.ndarray]],
    device: str = "cpu",
) -> OCRResult:
    if not candidates:
        raise ValueError("هیچ ورودی OCR برای ارزیابی وجود ندارد.")
    started = perf_counter()
    _pass_name, image = candidates[0]
    result = recognize(image, device=device)
    print(
        f"[OCR] selected engine={result.pass_name} total={perf_counter() - started:.2f}s",
        flush=True,
    )
    return result
