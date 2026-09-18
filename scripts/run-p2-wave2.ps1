$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root   = Split-Path -Parent $PSScriptRoot
$branch = "feat/p2-generative-document-restoration"
$Py     = Join-Path $root ".venv\Scripts\python.exe"
$port   = 8118
$cache  = Join-Path $env:USERPROFILE ".cache\daqiqkhan\realesrgan-ncnn-vulkan"
$outDir = "C:\Project\_DAQIQKHAN_P2_WAVE2"

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

Write-Host "`n=== P2 WAVE 2 — SAFE PREFLIGHT ===" -ForegroundColor Cyan
Write-Host "BENCHMARK ONLY — NO MERGE / NO DEPLOY / PRODUCT DEFAULT UNTOUCHED" -ForegroundColor Green

if ((git branch --show-current).Trim() -ne $branch) {
    throw "STOP: expected branch $branch"
}
if (git status --porcelain) {
    git status --short
    throw "STOP: worktree must be clean."
}
if (-not (Test-Path $Py)) {
    throw "STOP: Python venv not found: $Py"
}
if (-not (Test-Path "C:\Market\گواهی سپرده مس کاتد.png")) {
    throw "STOP: acceptance fixture is missing."
}

Write-Host "`n=== ENSURE OFFICIAL REAL-ESRGAN PORTABLE PACKAGE ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $cache | Out-Null

$headers = @{ "User-Agent" = "DaqiqKhan-P2-Benchmark" }
$portableTag = "v0.2.5.0"
$portableAssetName = "realesrgan-ncnn-vulkan-20220424-windows.zip"
$portableReleaseApi = "https://api.github.com/repos/xinntao/Real-ESRGAN/releases/tags/$portableTag"

function Find-CompletePortablePackage {
    return Get-ChildItem -Path $cache -Recurse -Filter "realesrgan-ncnn-vulkan.exe" -File -ErrorAction SilentlyContinue |
        Where-Object {
            $candidateModelDir = Join-Path $_.Directory.FullName "models"
            (Test-Path (Join-Path $candidateModelDir "realesrgan-x4plus.param")) -and
            (Test-Path (Join-Path $candidateModelDir "realesrgan-x4plus.bin"))
        } |
        Select-Object -First 1
}

$exe = Find-CompletePortablePackage

if ($null -eq $exe) {
    Write-Host "Complete portable package is not cached. Resolving official $portableTag assets..." -ForegroundColor Yellow

    $release = Invoke-RestMethod `
        -Uri $portableReleaseApi `
        -Headers $headers `
        -TimeoutSec 30

    $asset = $release.assets |
        Where-Object { $_.name -eq $portableAssetName } |
        Select-Object -First 1

    if ($null -eq $asset) {
        Write-Host "Official assets returned for ${portableTag}:" -ForegroundColor Yellow
        @($release.assets | ForEach-Object { $_.name }) |
            ForEach-Object { Write-Host " - $_" -ForegroundColor DarkYellow }
        throw "STOP: official model-complete Windows portable asset was not found."
    }

    $zip = Join-Path $cache $asset.name

    if (-not (Test-Path $zip) -or (Get-Item $zip).Length -lt 10MB) {
        Write-Host "Downloading official model-complete asset: $($asset.name)" -ForegroundColor Yellow
        Invoke-WebRequest `
            -Uri $asset.browser_download_url `
            -Headers $headers `
            -OutFile $zip `
            -TimeoutSec 600
    }
    else {
        Write-Host "Reusing cached official archive: $zip" -ForegroundColor Yellow
    }

    $extract = Join-Path $cache "portable-v0.2.5.0"
    if (Test-Path $extract) {
        Remove-Item -Recurse -Force $extract
    }
    Expand-Archive -Path $zip -DestinationPath $extract -Force

    $exe = Get-ChildItem -Path $extract -Recurse -Filter "realesrgan-ncnn-vulkan.exe" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

if ($null -eq $exe) {
    throw "STOP: realesrgan-ncnn-vulkan.exe was not found in the official portable package."
}

$modelDir = Join-Path $exe.Directory.FullName "models"
$paramPath = Join-Path $modelDir "realesrgan-x4plus.param"
$binPath = Join-Path $modelDir "realesrgan-x4plus.bin"

if (-not (Test-Path $paramPath)) {
    throw "STOP: realesrgan-x4plus.param missing after extracting official $portableTag portable package."
}
if (-not (Test-Path $binPath)) {
    throw "STOP: realesrgan-x4plus.bin missing after extracting official $portableTag portable package."
}

Write-Host "Official portable release: $portableTag" -ForegroundColor Green
Write-Host "Model files: $paramPath ; $binPath" -ForegroundColor Green

Write-Host "Real-ESRGAN: $($exe.FullName)" -ForegroundColor Green
Write-Host "Model dir: $modelDir" -ForegroundColor Green

Write-Host "`n=== ENSURE PADDLEOCR-VL SERVER ===" -ForegroundColor Cyan
if (-not (Test-VlServer)) {
    Write-Host "VL server is down. Starting the local stack..." -ForegroundColor Yellow
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
        throw "STOP: PaddleOCR-VL server did not become ready within 6 minutes."
    }
}
Write-Host "PaddleOCR-VL server: READY" -ForegroundColor Green

$env:PYTHONIOENCODING = "utf-8"
$env:DAQIQKHAN_VL_BACKEND = "llama-cpp-server"
$env:DAQIQKHAN_VL_SERVER_URL = "http://127.0.0.1:$port/v1"
$env:DAQIQKHAN_REALESRGAN_BIN = $exe.FullName
$env:DAQIQKHAN_REALESRGAN_MODEL_DIR = $modelDir

Write-Host "`n=== RUN QUALITY GATES ===" -ForegroundColor Cyan
& $Py -m ruff check app.py src tests scripts
if ($LASTEXITCODE -ne 0) { throw "STOP: ruff failed." }

& $Py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP: pytest failed." }

Write-Host "`n=== RUN REAL-ESRGAN FIDELITY BENCHMARK ===" -ForegroundColor Cyan
& $Py ".\scripts\benchmark_p2_restoration.py"
$code = $LASTEXITCODE

Write-Host "`n=== P2 WAVE 2 ARTIFACTS ===" -ForegroundColor Cyan
Write-Host $outDir -ForegroundColor Yellow
if (Test-Path $outDir) {
    Start-Process $outDir
}

if ($code -eq 3) {
    Write-Host "`nRESULT: REAL-ESRGAN REJECTED BY OCR FIDELITY GATE" -ForegroundColor Red
    exit 3
}
if ($code -ne 0) {
    throw "STOP: P2 Wave 2 benchmark failed with exit $code."
}

Write-Host "`nRESULT: OCR FIDELITY DID NOT REGRESS" -ForegroundColor Green
Write-Host "Human visual review is still mandatory. Candidate remains benchmark-only." -ForegroundColor Yellow
Write-Host "Open: $outDir\06-montage.png" -ForegroundColor Yellow
