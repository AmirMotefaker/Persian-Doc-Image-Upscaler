from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ESPCN_URL = (
    "https://raw.githubusercontent.com/Saafke/ESPCN_Tensorflow/"
    "master/models/ESPCN_x4.pb"
)
ESPCN_NAME = "ESPCN_x4.pb"
DOCUMENT_PROFILES = {"سند", "اسکن ضعیف"}
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
)
REALESRGAN_SHA256 = "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"
REALESRGAN_ARCHIVE = "realesrgan-ncnn-vulkan-20220424-windows.zip"


def _model_root() -> Path:
    root = Path(
        os.environ.get(
            "DAQIQKHAN_MODEL_DIR",
            Path.home() / ".cache" / "daqiqkhan" / "models",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_model() -> Path:
    path = _model_root() / ESPCN_NAME
    if path.is_file() and path.stat().st_size > 10_000:
        return path
    tmp = path.with_suffix(".download")
    tmp.unlink(missing_ok=True)
    print(f"[SR] downloading {ESPCN_NAME}...", flush=True)
    urllib.request.urlretrieve(ESPCN_URL, tmp)
    if not tmp.is_file() or tmp.stat().st_size <= 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("مدل Lightweight Super-Resolution معتبر دانلود نشد.")
    tmp.replace(path)
    print(f"[SR] model ready: {path}", flush=True)
    return path


@lru_cache(maxsize=1)
def _engine():
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError("ماژول OpenCV dnn_superres در دسترس نیست.")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(_ensure_model()))
    sr.setModel("espcn", 4)
    return sr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_realesrgan_bundle(root: Path) -> tuple[Path, Path] | None:
    executables = list(root.rglob("realesrgan-ncnn-vulkan.exe"))
    params = list(root.rglob("realesrgan-x4plus.param"))
    for exe in executables:
        for param in params:
            model_dir = param.parent
            model_bin = model_dir / "realesrgan-x4plus.bin"
            if model_bin.is_file():
                return exe, model_dir
    return None


@lru_cache(maxsize=1)
def _ensure_realesrgan() -> tuple[Path, Path]:
    if os.name != "nt":
        raise RuntimeError("Real-ESRGAN NCNN visual engine is currently configured for Windows.")

    root = _model_root() / "realesrgan-ncnn-v0250"
    existing = _find_realesrgan_bundle(root)
    if existing is not None:
        return existing

    root.mkdir(parents=True, exist_ok=True)
    archive = root / REALESRGAN_ARCHIVE
    if not archive.is_file() or _sha256(archive) != REALESRGAN_SHA256:
        archive.unlink(missing_ok=True)
        tmp = archive.with_suffix(".download")
        tmp.unlink(missing_ok=True)
        print("[SR] downloading official Real-ESRGAN portable bundle...", flush=True)
        urllib.request.urlretrieve(REALESRGAN_URL, tmp)
        if _sha256(tmp) != REALESRGAN_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("هش فایل رسمی Real-ESRGAN معتبر نیست.")
        tmp.replace(archive)

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)

    found = _find_realesrgan_bundle(root)
    if found is None:
        raise RuntimeError("Real-ESRGAN executable/model bundle پس از استخراج کامل نیست.")
    return found


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر ورودی معتبر نیست.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("ساختار تصویر پشتیبانی نمی‌شود.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError("ساختار کانال‌های تصویر پشتیبانی نمی‌شود.")
    return image


def _normalize_document_luma(image: np.ndarray, weak_scan: bool) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    if weak_scan:
        local = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8)).apply(lightness)
        lightness = cv2.addWeighted(lightness, 0.74, local, 0.26, 0)
    else:
        local = cv2.createCLAHE(clipLimit=1.10, tileGridSize=(10, 10)).apply(lightness)
        lightness = cv2.addWeighted(lightness, 0.91, local, 0.09, 0)

    return cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def _edge_limited_unsharp(image: np.ndarray, amount: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = max(18.0, float(np.percentile(magnitude, 82.0)))
    mask = np.clip((magnitude - threshold * 0.35) / threshold, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.65)[..., None]

    blur = cv2.GaussianBlur(image, (0, 0), 0.52)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    mixed = (
        image.astype(np.float32) * (1.0 - mask)
        + sharpened.astype(np.float32) * mask
    )
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _document_restore(image: np.ndarray, scale: int, weak_scan: bool) -> np.ndarray:
    cleaned = _normalize_document_luma(image, weak_scan=weak_scan)
    height, width = cleaned.shape[:2]
    output = cv2.resize(
        cleaned,
        (width * scale, height * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return _edge_limited_unsharp(output, amount=0.20 if weak_scan else 0.16)


def _realesrgan_restore(image: np.ndarray, requested_scale: float) -> np.ndarray:
    exe, model_dir = _ensure_realesrgan()
    with tempfile.TemporaryDirectory(prefix="daqiqkhan-sr-") as workdir:
        input_path = Path(workdir) / "input.png"
        output_path = Path(workdir) / "output.png"
        if not cv2.imwrite(str(input_path), image):
            raise RuntimeError("ذخیره ورودی موقت Real-ESRGAN ناموفق بود.")

        command = [
            str(exe),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-m",
            str(model_dir),
            "-n",
            "realesrgan-x4plus",
            "-s",
            "4",
            "-f",
            "png",
        ]
        completed = subprocess.run(
            command,
            cwd=str(exe.parent),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "unknown failure").strip()
            raise RuntimeError(f"Real-ESRGAN NCNN failed: {detail}")

        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if output is None:
            raise RuntimeError("خروجی Real-ESRGAN قابل خواندن نیست.")

    target_scale = max(1.0, float(requested_scale))
    if abs(target_scale - 4.0) > 0.01:
        height, width = image.shape[:2]
        target = (round(width * target_scale), round(height * target_scale))
        interpolation = cv2.INTER_AREA if target_scale < 4.0 else cv2.INTER_LANCZOS4
        output = cv2.resize(output, target, interpolation=interpolation)
    return output


def _natural_ai_restore(image: np.ndarray) -> np.ndarray:
    try:
        print("[SR] engine=ESPCN x4 profile=natural", flush=True)
        return _engine().upsample(image)
    except Exception as exc:
        print(f"[SR] ESPCN unavailable; fallback=Lanczos reason={exc}", flush=True)
        height, width = image.shape[:2]
        return cv2.resize(
            image,
            (width * 4, height * 4),
            interpolation=cv2.INTER_LANCZOS4,
        )


def super_resolve_visual(
    image: np.ndarray,
    scale: float = 2.0,
    profile: str = "سند",
) -> np.ndarray:
    started = perf_counter()
    image = _ensure_bgr(image)
    requested = max(1.0, float(scale))

    if profile in DOCUMENT_PROFILES:
        try:
            print(
                f"[SR] engine=RealESRGAN-NCNN visual-only scale={requested:g} "
                f"profile={profile}",
                flush=True,
            )
            output = _realesrgan_restore(image, requested)
        except Exception as exc:
            document_scale = max(1, min(2, round(requested)))
            weak_scan = profile == "اسکن ضعیف"
            print(
                f"[SR] RealESRGAN unavailable; fallback=DocumentRestoreSharp "
                f"reason={exc}",
                flush=True,
            )
            output = _document_restore(image, document_scale, weak_scan=weak_scan)
    else:
        output = _natural_ai_restore(image)
        native_scale = 4
        if abs(requested - native_scale) > 0.01:
            height, width = image.shape[:2]
            target = (
                max(1, round(width * requested)),
                max(1, round(height * requested)),
            )
            interpolation = (
                cv2.INTER_LANCZOS4 if requested > native_scale else cv2.INTER_AREA
            )
            output = cv2.resize(output, target, interpolation=interpolation)

    print(
        f"[SR] done size={output.shape[1]}x{output.shape[0]} "
        f"elapsed={perf_counter() - started:.2f}s",
        flush=True,
    )
    return output
