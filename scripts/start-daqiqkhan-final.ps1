$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$llamaRoot = Join-Path $env:USERPROFILE ".cache\daqiqkhan\llama.cpp"
$modelRoot = Join-Path $env:USERPROFILE ".cache\daqiqkhan\paddleocr-vl-1.6"
$port = 8118

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

function Ensure-Download([string]$Url, [string]$Path, [long]$MinimumBytes) {
    if ((Test-Path $Path) -and ((Get-Item $Path).Length -ge $MinimumBytes)) {
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Write-Host "Downloading $(Split-Path -Leaf $Path) (resumable)..." -ForegroundColor Yellow
    curl.exe -L `
        --retry 50 `
        --retry-delay 3 `
        --retry-all-errors `
        --connect-timeout 30 `
        -C - `
        -o $Path `
        $Url

    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Path"
    }
    if ((Get-Item $Path).Length -lt $MinimumBytes) {
        throw "Downloaded file is incomplete: $Path"
    }
}

Write-Host "`n=== DAQIQKHAN FINAL LOCAL START ===" -ForegroundColor Cyan

if (-not (Test-Path $py)) {
    throw "Python venv is missing: $py"
}

Write-Host "`n=== STOP OLD APP ===" -ForegroundColor Cyan
Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match '^python(w)?(\.exe)?$' -and
        $_.ExecutablePath -and
        $_.ExecutablePath.StartsWith($venv, [System.StringComparison]::OrdinalIgnoreCase)
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host "`n=== QUALITY GATES ===" -ForegroundColor Cyan
& $py -m ruff check app.py src tests scripts
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }

& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

& $py -m compileall -q app.py src scripts
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

if (-not (Test-VlServer)) {
    Write-Host "`n=== PREPARE LLAMA.CPP VULKAN ===" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $llamaRoot | Out-Null

    $llamaServer = Get-ChildItem -Path $llamaRoot -Filter "llama-server.exe" -Recurse `
        -ErrorAction SilentlyContinue | Select-Object -First 1

    if (-not $llamaServer) {
        $releases = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
        $asset = $null
        foreach ($release in $releases) {
            $candidate = $release.assets |
                Where-Object { $_.name -match '(?i)bin-win-vulkan-x64\.zip$' } |
                Select-Object -First 1
            if ($candidate) {
                $asset = $candidate
                break
            }
        }
        if (-not $asset) {
            throw "Windows Vulkan llama.cpp release asset not found."
        }

        $zip = Join-Path $llamaRoot $asset.name
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $llamaRoot -Force
        $llamaServer = Get-ChildItem -Path $llamaRoot -Filter "llama-server.exe" -Recurse |
            Select-Object -First 1
    }

    if (-not $llamaServer) {
        throw "llama-server.exe not found."
    }

    Write-Host "`n=== PREPARE PADDLEOCR-VL 1.6 GGUF ===" -ForegroundColor Cyan
    $model = Join-Path $modelRoot "PaddleOCR-VL-1.6-GGUF.gguf"
    $mmproj = Join-Path $modelRoot "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"

    Ensure-Download `
        "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF.gguf?download=true" `
        $model `
        800MB

    Ensure-Download `
        "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf?download=true" `
        $mmproj `
        700MB

    Write-Host "`n=== START PADDLEOCR-VL SERVER ===" -ForegroundColor Cyan
    Get-Process llama-server -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue

    $stdout = Join-Path $llamaRoot "server.out.log"
    $stderr = Join-Path $llamaRoot "server.err.log"
    Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue

    $server = Start-Process `
        -FilePath $llamaServer.FullName `
        -ArgumentList @(
            "-m", "`"$model`"",
            "--mmproj", "`"$mmproj`"",
            "--host", "127.0.0.1",
            "--port", "$port",
            "--temp", "0",
            "-ngl", "99"
        ) `
        -WorkingDirectory $llamaServer.DirectoryName `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 3
        if ($server.HasExited) {
            if (Test-Path $stderr) { Get-Content $stderr -Tail 100 }
            throw "PaddleOCR-VL server exited."
        }
        if (Test-VlServer) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        if (Test-Path $stderr) { Get-Content $stderr -Tail 100 }
        throw "PaddleOCR-VL server did not become ready."
    }
}

Write-Host "PaddleOCR-VL server: READY" -ForegroundColor Green

$env:DAQIQKHAN_VL_BACKEND = "llama-cpp-server"
$env:DAQIQKHAN_VL_SERVER_URL = "http://127.0.0.1:$port/v1"

Write-Host "`n=== START DAQIQKHAN ===" -ForegroundColor Green
& $py ".\app.py"
