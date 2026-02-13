param(
    [string]$ConfigPath = "",
    [switch]$InstallDeps,
    [switch]$KillExisting
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

function Resolve-ConfigPath([string]$UserProvided) {
    if ($UserProvided -and (Test-Path $UserProvided)) {
        return (Resolve-Path $UserProvided).Path
    }

    $localConfig = Join-Path $PSScriptRoot "companion_config.env"
    if (Test-Path $localConfig) {
        return $localConfig
    }

    $exampleConfig = Join-Path $PSScriptRoot "companion_config.example.env"
    if (Test-Path $exampleConfig) {
        return $exampleConfig
    }

    return ""
}

function Import-EnvFile([string]$PathToFile) {
    if (-not $PathToFile) {
        Write-Host "[MeetingScribe] No config file found. Using existing shell environment values." -ForegroundColor Yellow
        return
    }

    Write-Host "[MeetingScribe] Loading environment from $PathToFile" -ForegroundColor Cyan
    Get-Content $PathToFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line) { return }
        if ($line.StartsWith("#")) { return }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) { return }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (-not $name) { return }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Get-PortOwnerPid([int]$Port) {
    $matches = netstat -ano | Select-String "127\.0\.0\.1:$Port\s+.+\s+(\d+)$"
    foreach ($match in $matches) {
        if ($match -match "(\d+)$") {
            return [int]$Matches[1]
        }
    }
    return $null
}

$python = Resolve-Python
$resolvedConfig = Resolve-ConfigPath $ConfigPath

Import-EnvFile $resolvedConfig

$port = 8000
if ($env:MSCRIBE_PORT) {
    $parsed = 0
    if ([int]::TryParse($env:MSCRIBE_PORT, [ref]$parsed)) {
        $port = $parsed
    }
}

$ownerPid = Get-PortOwnerPid $port
if ($ownerPid) {
    if ($KillExisting) {
        Write-Host "[MeetingScribe] Port $port is in use by PID $ownerPid. Killing existing process..." -ForegroundColor Yellow
        taskkill /PID $ownerPid /F | Out-Null
        Start-Sleep -Milliseconds 500
    }
    else {
        throw "Port $port is already in use by PID $ownerPid. Re-run with -KillExisting or stop that process first."
    }
}

if ($InstallDeps) {
    Write-Host "[MeetingScribe] Installing requirements..." -ForegroundColor Cyan
    & $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
}

Write-Host "[MeetingScribe] Starting companion app..." -ForegroundColor Green
& $python (Join-Path $PSScriptRoot "companion_app.py")
