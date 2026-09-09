$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = "https://github.com/AmirMotefaker/Persian-Doc-Image-Upscaler.git"
$Root = "C:\Project\Persian-Doc-Image-Upscaler"
$Branch = "feat/p0-platform-rebuild"
$PythonVersion = "3.12"
$ExpectedMarker = "BOOTSTRAP_V3"

Write-Host "=== Persian Doc/Image Upscaler bootstrap [$ExpectedMarker] ===" -ForegroundColor Cyan
Write-Host "Script path: $PSCommandPath" -ForegroundColor DarkGray

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required but was not found in PATH."
}

function Invoke-External([scriptblock]$Command, [string]$ErrorMessage) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$ErrorMessage (exit code $LASTEXITCODE)"
    }
}

function Get-Python312Command {
    $candidates = @()

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

$PythonCommand = Get-Python312Command
if (-not $PythonCommand) {
    Install-Python312
    $PythonCommand = Get-Python312Command
}

if (-not $PythonCommand) {
    throw "Python 3.12 installation finished but this process still cannot discover it. The installer may require a new shell. Reopen PowerShell and rerun the same bootstrap command; no manual file edits are required."
}

Write-Host "Python 3.12 detected: $($PythonCommand -join ' ')" -ForegroundColor Green

if (-not (Test-Path $Root)) {
    Write-Host "Cloning repository..." -ForegroundColor Yellow
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
    Write-Host "Removing incomplete/old virtual environment..." -ForegroundColor DarkYellow
    Remove-Item -Recurse -Force ".venv"
}

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
