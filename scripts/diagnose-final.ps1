$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root   = Split-Path -Parent $PSScriptRoot
$venv   = Join-Path $root ".venv"
$Py     = Join-Path $venv "Scripts\python.exe"
$outDir = "C:\Project\_DAQIQKHAN_DIAG"
$port   = 8118

Set-Location $root

function Test-VlServer {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/models" -TimeoutSec 3
        return $true
    }
    catch {
        return $false
    }
}

Write-Host "`n=== 1. SELECT EXACT SOURCE IMAGE ===" -ForegroundColor Cyan
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "فایل اصلی تصویر را انتخاب کنید"
$dialog.Filter = "Image files|*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff|All files|*.*"
$dialog.Multiselect = $false
$dialog.CheckFileExists = $true
$dialog.InitialDirectory = [Environment]::GetFolderPath("Desktop")

$result = $dialog.ShowDialog()
if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
    throw "هیچ فایل تصویری انتخاب نشد."
}

$input = $dialog.FileName
Write-Host "Selected: $input" -ForegroundColor Green

Write-Host "`n=== 2. ENSURE VL SERVER ===" -ForegroundColor Cyan
if (-not (Test-VlServer)) {
    Write-Host "VL server is down. Starting full local stack in a separate PowerShell..." -ForegroundColor Yellow
    $launcher = Join-Path $PSScriptRoot "start-daqiqkhan-final.ps1"
    Start-Process `
        -FilePath "pwsh" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$launcher`"") `
        -WorkingDirectory $root | Out-Null

    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 3
        if (Test-VlServer) {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        throw "PaddleOCR-VL server did not become ready within 6 minutes."
    }
}

Write-Host "PaddleOCR-VL server: READY" -ForegroundColor Green

$env:DQ_INPUT  = $input
$env:DQ_OUTDIR = $outDir
$env:PYTHONIOENCODING = "utf-8"
$env:DAQIQKHAN_VL_BACKEND = "llama-cpp-server"
$env:DAQIQKHAN_VL_SERVER_URL = "http://127.0.0.1:$port/v1"

Write-Host "`n=== 3. RUN RELEASE-CANDIDATE DIAGNOSTIC ===" -ForegroundColor Green

@'
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path.cwd()
input_path = Path(os.environ["DQ_INPUT"])
out_dir = Path(os.environ["DQ_OUTDIR"])
out_dir.mkdir(parents=True, exist_ok=True)

for child in list(out_dir.iterdir()):
    if child.is_file():
        child.unlink()
    elif child.is_dir():
        shutil.rmtree(child)

sys.path.insert(0, str(root))
from src.persian_upscaler.service import process_image


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def normalize_text(text: str) -> str:
    table = str.maketrans({
        "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
        "أ": "ا", "إ": "ا", "ٱ": "ا",
    })
    return " ".join(text.translate(table).replace("\t", " ").split())


def compact(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def text_stats(text: str) -> dict:
    return {
        "length": len(text.strip()),
        "persian_chars": sum("\u0600" <= ch <= "\u06ff" for ch in text),
        "latin_chars": sum(("a" <= ch.lower() <= "z") for ch in text),
        "digits": sum(ch.isdigit() for ch in text),
        "non_empty_lines": len([line for line in text.splitlines() if line.strip()]),
        "tabs": text.count("\t"),
    }


def fixture_token_recall(text: str, width: int, height: int):
    if (width, height) != (335, 224):
        return None

    expected = [
        "آخرین قیمت",
        "بالاترین قیمت",
        "پایین ترین قیمت",
        "اولین قیمت",
        "قیمت پایانی",
        "حجم معاملات",
        "ساعت آخرین معامله",
        "آستانه بالا",
        "آستانه پایین",
        "27,000,000",
        "27,768,000",
        "26,670,130",
        "26,779,180",
        "27,135,086",
        "55,918",
        "1,517,559,720",
        "28,905,560",
        "23,650,010",
    ]
    haystack = compact(text)
    found = [token for token in expected if compact(token) in haystack]
    missing = [token for token in expected if compact(token) not in haystack]
    return {
        "expected": len(expected),
        "found": len(found),
        "recall": round(len(found) / len(expected), 4),
        "found_tokens": found,
        "missing_tokens": missing,
    }


source = read_image(input_path)
source_h, source_w = source.shape[:2]

print("\n=== INPUT ===")
print(input_path)
print(f"Source: {source_w}x{source_h}")

print("\n=== PRODUCT PIPELINE ===")
enhanced_path, ocr_preview_path, text, text_path = process_image(
    str(input_path),
    profile="سند",
    scale=4.0,
    language="fa",
    output_format="PNG",
    engine="Super-Resolution Pro",
)

enhanced_path = Path(enhanced_path)
ocr_preview_path = Path(ocr_preview_path)
text_path = Path(text_path)
enhanced = read_image(enhanced_path)

bicubic = cv2.resize(
    source,
    (enhanced.shape[1], enhanced.shape[0]),
    interpolation=cv2.INTER_CUBIC,
)

source_sharp = sharpness(source)
bicubic_sharp = sharpness(bicubic)
output_sharp = sharpness(enhanced)
source_contrast = contrast(source)
output_contrast = contrast(enhanced)

ocr_stats = text_stats(text)
fixture = fixture_token_recall(text, source_w, source_h)

report = {
    "input": str(input_path),
    "source_size": [source_w, source_h],
    "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
    "scale": round(enhanced.shape[1] / source_w, 2),
    "source_sharpness": round(source_sharp, 4),
    "bicubic_same_scale_sharpness": round(bicubic_sharp, 4),
    "output_sharpness": round(output_sharp, 4),
    "sharpness_vs_bicubic": round(output_sharp / max(bicubic_sharp, 1e-6), 4),
    "source_contrast": round(source_contrast, 4),
    "output_contrast": round(output_contrast, 4),
    "contrast_gain": round(output_contrast / max(source_contrast, 1e-6), 4),
    "mad_vs_plain_bicubic": round(mean_abs_diff(enhanced, bicubic), 4),
    "ocr": ocr_stats,
    "fixture_token_recall": fixture,
    "ocr_first_30_lines": text.splitlines()[:30],
}

shutil.copy2(input_path, out_dir / ("01-original" + input_path.suffix.lower()))
shutil.copy2(enhanced_path, out_dir / "02-enhanced.png")
shutil.copy2(ocr_preview_path, out_dir / "03-ocr-input.png")
shutil.copy2(text_path, out_dir / "04-ocr.txt")
(out_dir / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

failures = []
if report["scale"] < 3.9:
    failures.append(f"document output is not true 4x: {report['scale']}x")
if report["sharpness_vs_bicubic"] < 1.50:
    failures.append(
        f"same-scale sharpness gain is weak: {report['sharpness_vs_bicubic']}x vs bicubic"
    )
if report["mad_vs_plain_bicubic"] < 3.0:
    failures.append(
        f"enhanced image is too close to ordinary resize: {report['mad_vs_plain_bicubic']}"
    )
if ocr_stats["persian_chars"] < 20 or ocr_stats["non_empty_lines"] < 5:
    failures.append("Persian OCR/layout is incomplete")
if fixture and fixture["recall"] < 0.85:
    failures.append(
        f"acceptance fixture OCR token recall is too low: {fixture['recall']:.1%}"
    )

print("\n=== FINAL REPORT ===")
print(json.dumps(report, ensure_ascii=False, indent=2))

summary = [
    "DAQIQKHAN RELEASE-CANDIDATE DIAGNOSTIC",
    "",
    f"Input: {report['source_size']}",
    f"Output: {report['output_size']} ({report['scale']}x)",
    f"Sharpness vs same-scale Bicubic: {report['sharpness_vs_bicubic']}x",
    f"MAD vs Bicubic: {report['mad_vs_plain_bicubic']}",
    f"Persian chars: {ocr_stats['persian_chars']}",
    f"Digits: {ocr_stats['digits']}",
    f"Lines: {ocr_stats['non_empty_lines']}",
    f"Table separators: {ocr_stats['tabs']}",
]

if fixture:
    summary += [
        f"Fixture token recall: {fixture['found']}/{fixture['expected']} ({fixture['recall']:.1%})",
        "Missing fixture tokens: " + ", ".join(fixture["missing_tokens"]),
    ]

if failures:
    summary += ["", "RESULT: FAIL"]
    summary += [f"{index}. {failure}" for index, failure in enumerate(failures, 1)]
else:
    summary += ["", "RESULT: PASS"]

(out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n" + "\n".join(summary))

if failures:
    sys.exit(2)
'@ | & $Py -

$code = $LASTEXITCODE
Write-Host "`n=== ARTIFACTS ===" -ForegroundColor Cyan
Write-Host $outDir -ForegroundColor Yellow
if (Test-Path (Join-Path $outDir "summary.txt")) {
    Get-Content (Join-Path $outDir "summary.txt")
}
Start-Process $outDir

if ($code -ne 0) {
    Write-Host "`nDIAGNOSTIC: FAIL" -ForegroundColor Red
    exit $code
}

Write-Host "`nDIAGNOSTIC: PASS" -ForegroundColor Green
