$ErrorActionPreference = "Stop"

$Repo = "https://github.com/AmirMotefaker/Persian-Doc-Image-Upscaler.git"
$Root = "C:\Project\Persian-Doc-Image-Upscaler"
$Branch = "feat/p0-platform-rebuild"
$PythonVersion = "3.12"

Write-Host "=== Persian Doc/Image Upscaler bootstrap ===" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required but was not found in PATH."
}

function Test-Python312 {
    try {
        $version = & py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.12")
    }
    catch {
        return $false
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python launcher not found. Installing Python 3.12 automatically..." -ForegroundColor Yellow
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Neither Python launcher nor winget is available. Install App Installer/winget, then rerun this script."
    }

    winget install --id Python.Python.3.12 --exact --source winget --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic Python 3.12 installation failed with exit code $LASTEXITCODE."
    }
}
elseif (-not (Test-Python312)) {
    Write-Host "Python 3.12 not found. Installing it automatically..." -ForegroundColor Yellow
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Python 3.12 is missing and winget is unavailable."
    }

    winget install --id Python.Python.3.12 --exact --source winget --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic Python 3.12 installation failed with exit code $LASTEXITCODE."
    }
}

# Refresh PATH for the current PowerShell process after winget installation.
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    $launcherCandidates = @(
        "$env:WINDIR\py.exe",
        "$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe"
    )
    $launcher = $launcherCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($launcher) {
        Set-Alias py $launcher -Scope Script
    }
}

if (-not (Test-Python312)) {
    throw "Python 3.12 installation completed but the runtime is still not discoverable. Close and reopen PowerShell, then rerun the same bootstrap command."
}

Write-Host "Python 3.12 detected." -ForegroundColor Green

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
    Remove-Item -Recurse -Force ".venv"
}

Write-Host "Creating Python 3.12 virtual environment..." -ForegroundColor Yellow
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    throw "Virtual environment creation failed."
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Virtual environment Python was not created at $Py"
}

& $Py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }

& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements installation failed." }

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
