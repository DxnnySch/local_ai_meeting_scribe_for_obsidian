param(
    [string]$VaultPath = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    return $PSScriptRoot
}

function Resolve-PluginId([string]$ProjectRoot) {
    $manifestPath = Join-Path $ProjectRoot "manifest.json"
    if (-not (Test-Path $manifestPath)) {
        throw "manifest.json not found at $manifestPath"
    }
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    if (-not $manifest.id) {
        throw "manifest.json does not contain an 'id' field."
    }
    return [string]$manifest.id
}

function Resolve-VaultPath([string]$UserProvided) {
    if ($UserProvided) {
        if (-not (Test-Path $UserProvided)) {
            throw "Vault path does not exist: $UserProvided"
        }
        return (Resolve-Path $UserProvided).Path
    }

    if ($env:OBSIDIAN_VAULT_PATH -and (Test-Path $env:OBSIDIAN_VAULT_PATH)) {
        return (Resolve-Path $env:OBSIDIAN_VAULT_PATH).Path
    }

    throw "Vault path missing. Use -VaultPath <path> or set OBSIDIAN_VAULT_PATH."
}

function Invoke-NodeBuild([string]$ProjectRoot) {
    Write-Host "[Release] Running npm build..." -ForegroundColor Cyan
    Push-Location $ProjectRoot
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "npm run build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Copy-PluginArtifacts([string]$ProjectRoot, [string]$TargetPluginDir) {
    $requiredFiles = @("manifest.json", "main.js")
    foreach ($file in $requiredFiles) {
        $sourcePath = Join-Path $ProjectRoot $file
        if (-not (Test-Path $sourcePath)) {
            throw "Required build artifact missing: $sourcePath"
        }
    }

    if (-not (Test-Path $TargetPluginDir)) {
        New-Item -ItemType Directory -Path $TargetPluginDir -Force | Out-Null
    }

    Copy-Item (Join-Path $ProjectRoot "manifest.json") -Destination (Join-Path $TargetPluginDir "manifest.json") -Force
    Copy-Item (Join-Path $ProjectRoot "main.js") -Destination (Join-Path $TargetPluginDir "main.js") -Force

    $stylesPath = Join-Path $ProjectRoot "styles.css"
    if (Test-Path $stylesPath) {
        Copy-Item $stylesPath -Destination (Join-Path $TargetPluginDir "styles.css") -Force
    }
}

$projectRoot = Resolve-ProjectRoot
$pluginId = Resolve-PluginId $projectRoot
$resolvedVaultPath = Resolve-VaultPath $VaultPath
$targetPluginDir = Join-Path $resolvedVaultPath ".obsidian\plugins\$pluginId"

if (-not $SkipBuild) {
    Invoke-NodeBuild $projectRoot
}

Copy-PluginArtifacts $projectRoot $targetPluginDir

Write-Host "[Release] Plugin deployed successfully." -ForegroundColor Green
Write-Host "[Release] Target: $targetPluginDir" -ForegroundColor Green
Write-Host "[Release] Restart Obsidian or reload plugins to apply changes." -ForegroundColor Yellow
