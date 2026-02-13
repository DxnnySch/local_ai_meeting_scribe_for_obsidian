param(
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

$python = Resolve-Python

if ($InstallDeps) {
    Write-Host "[Companion EXE] Installing Python dependencies..." -ForegroundColor Cyan
    & $python -m pip install --upgrade pip
    & $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    & $python -m pip install -r (Join-Path $PSScriptRoot "requirements_companion_exe.txt")
}

Write-Host "[Companion EXE] Building LocalMeetingScribeCompanion.exe..." -ForegroundColor Cyan
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "LocalMeetingScribeCompanion" `
    (Join-Path $PSScriptRoot "companion_tray.py")

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

$exePath = Join-Path $PSScriptRoot "dist\LocalMeetingScribeCompanion.exe"
if (-not (Test-Path $exePath)) {
    throw "Build succeeded but executable was not found at $exePath"
}

Write-Host "[Companion EXE] Build completed." -ForegroundColor Green
Write-Host "[Companion EXE] Output: $exePath" -ForegroundColor Green
