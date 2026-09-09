$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = "https://github.com/AmirMotefaker/Persian-Doc-Image-Upscaler.git"
$Root = "C:\Project\Persian-Doc-Image-Upscaler"
$Branch = "feat/p0-platform-rebuild"
$ExpectedMarker = "BOOTSTRAP_V6"

Write-Host "=== Persian Doc/Image Upscaler bootstrap [$ExpectedMarker] ===" -ForegroundColor Cyan
Write-Host "Script path: $PSCommandPath" -ForegroundColor DarkGray

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required but was not found in PATH."
}

function Get-Python312Command {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $version = & $pyLauncher.Source -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.12") {
                return @($pyLauncher.Source, "-3.12")
            }
        } catch {}
    }

    $pythonCandidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "C:\Python312\python.exe"
    )

    foreach ($candidate in $pythonCandidates) {
        if (Test-Path $candidate) {
            try {
                $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.12") {
                    return @($candidate)
                }
            } catch {}
        }
    }

    return $null
}

function Install-Python312 {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python 3.12 is missing and winget is unavailable. Install Microsoft App Installer so winget is available, then rerun the same command."
    }

    Write-Host "Python 3.12 not found. Installing automatically with winget..." -ForegroundColor Yellow
    & $winget.Source install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic Python 3.12 installation failed with exit code $LASTEXITCODE."
    }

    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Stop-RepoVenvProcesses([string]$VenvRoot) {
    $normalized = [System.IO.Path]::GetFullPath($VenvRoot).TrimEnd('\')
    $matches = @()

    try {
        $matches = @(
            Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
                $_.ExecutablePath -and ([System.IO.Path]::GetFullPath($_.ExecutablePath)).StartsWith($normalized, [System.StringComparison]::OrdinalIgnoreCase)
            }
        )
    } catch {
        $matches = @()
        Write-Host "Could not enumerate process executable paths; continuing with delete retry." -ForegroundColor DarkYellow
    }

    foreach ($proc in $matches) {
        Write-Host "Stopping locked venv process PID $($proc.ProcessId): $($proc.Name)" -ForegroundColor DarkYellow
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Host "Failed to stop PID $($proc.ProcessId): $($_.Exception.Message)" -ForegroundColor DarkYellow
        }
    }

    if (@($matches).Count -gt 0) {
        Start-Sleep -Milliseconds 700
    }
}

function Remove-VenvSafely([string]$VenvPath) {
    if (-not (Test-Path $VenvPath)) { return }

    $full = [System.IO.Path]::GetFullPath($VenvPath)
    Stop-RepoVenvProcesses $full
    $lastError = $null

    for ($attempt = 1; $attempt -le 6; $attempt++) {
        try {
            Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction Stop
            Write-Host "Old virtual environment removed." -ForegroundColor Green
            return
        } catch {
            $lastError = $_
            Write-Host "Venv delete attempt $attempt failed; retrying..." -ForegroundColor DarkYellow
            Stop-RepoVenvProcesses $full
            Start-Sleep -Milliseconds (500 * $attempt)
        }
    }

    throw "Unable to remove locked virtual environment at $full. Last error: $($lastError.Exception.Message)"
}

function Clear-PythonTlsOverrides {
    foreach ($name in @("PIP_CERT", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")) {
        if (Test-Path "Env:$name") {
            $value = (Get-Item "Env:$name").Value
            Write-Host "Clearing inherited TLS override $name=$value" -ForegroundColor DarkYellow
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
    }
}

$PythonCommand = Get-Python312Command
if (-not $PythonCommand) {
    Install-Python312
    $PythonCommand = Get-Python312Command
}
if (-not $PythonCommand) {
    throw "Python 3.12 installation finished but this process still cannot discover it. Reopen PowerShell and rerun the same bootstrap command; no manual file edits are required."
}

Write-Host "Python 3.12 detected: $($PythonCommand -join ' ')" -ForegroundColor Green

if (-not (Test-Path $Root)) {
    git clone $Repo $Root
    if ($LASTEXITCODE -ne 0) { throw "git clone failed." }
}

Set-Location $Root
git fetch origin --prune
if ($LASTEXITCODE -ne 0) { throw "git fetch failed." }
git switch $Branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git switch -c $Branch --track "origin/$Branch"
    if ($LASTEXITCODE -ne 0) { throw "Unable to switch to $Branch." }
}
git pull --ff-only origin $Branch
if ($LASTEXITCODE -ne 0) { throw "git pull failed." }

if (Test-Path ".venv") {
    Write-Host "Removing incomplete/old virtual environment safely..." -ForegroundColor DarkYellow
    Remove-VenvSafely ".venv"
}

Clear-PythonTlsOverrides

Write-Host "Creating Python 3.12 virtual environment..." -ForegroundColor Yellow
if ($PythonCommand.Count -eq 2) {
    & $PythonCommand[0] $PythonCommand[1] -m venv .venv
} else {
    & $PythonCommand[0] -m venv .venv
}
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Virtual environment Python was not created at $Py" }
Write-Host "Virtual environment ready: $Py" -ForegroundColor Green

& $Py -m pip --version
if ($LASTEXITCODE -ne 0) { throw "Bundled pip is unavailable." }

$PipCa = & $Py -c "import os, pip._vendor.certifi as c; p=c.where(); print(p); raise SystemExit(0 if os.path.isfile(p) else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Bundled pip CA certificate bundle is missing before dependency installation: $PipCa"
}
Write-Host "pip CA bundle verified: $PipCa" -ForegroundColor Green

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

Write-Host "Installing runtime dependencies..." -ForegroundColor Yellow
& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements installation failed." }

Write-Host "Installing development dependencies..." -ForegroundColor Yellow
& $Py -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "development dependencies installation failed." }

Write-Host "=== Quality gates ===" -ForegroundColor Yellow
& $Py -m ruff check src tests
if ($LASTEXITCODE -ne 0) { throw "ruff failed." }
& $Py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed." }
& $Py -m compileall -q app.py src
if ($LASTEXITCODE -ne 0) { throw "compileall failed." }

Write-Host "=== Environment ready ===" -ForegroundColor Green
Write-Host "Run app with:"
Write-Host "& '$Py' app.py" -ForegroundColor Cyan
