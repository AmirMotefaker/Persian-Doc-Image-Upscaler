$ErrorActionPreference = "Stop"

$Repo = "https://github.com/AmirMotefaker/Persian-Doc-Image-Upscaler.git"
$Root = "C:\Project\Persian-Doc-Image-Upscaler"
$Branch = "feat/p0-platform-rebuild"

Write-Host "=== Persian Doc/Image Upscaler bootstrap ===" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is required." }
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' is required." }

if (-not (Test-Path $Root)) {
    git clone $Repo $Root
}

Set-Location $Root
git fetch origin --prune
git switch $Branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git switch -c $Branch --track "origin/$Branch"
}
git pull --ff-only origin $Branch

$Python = "3.12"
if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
py -$Python -m venv .venv
$Py = Join-Path $Root ".venv\Scripts\python.exe"

& $Py -m pip install --upgrade pip setuptools wheel
& $Py -m pip install -r requirements.txt
& $Py -m pip install -e ".[dev]"

Write-Host "=== Quality gates ===" -ForegroundColor Yellow
& $Py -m ruff check src tests
& $Py -m pytest -q
& $Py -m compileall -q app.py src

Write-Host "=== Environment ready ===" -ForegroundColor Green
Write-Host "Run app with:"
Write-Host "& '$Py' app.py" -ForegroundColor Cyan
