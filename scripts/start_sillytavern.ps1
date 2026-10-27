param(
    [int]$Port = 8001
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stRoot = Join-Path $projectRoot 'third_party\SillyTavern'
$configPath = Join-Path $stRoot 'config.yaml'

if (-not (Test-Path (Join-Path $stRoot 'server.js'))) {
    throw '没有找到 SillyTavern 源码。请先按项目说明获取固定版本。'
}

if (-not (Test-Path $configPath)) {
    $configLines = Get-Content (Join-Path $stRoot 'default\config.yaml')
    $insideBrowser = $false
    for ($i = 0; $i -lt $configLines.Count; $i++) {
        if ($configLines[$i] -match '^browserLaunch:') { $insideBrowser = $true; continue }
        if ($insideBrowser -and $configLines[$i] -match '^\S') { $insideBrowser = $false }
        if ($insideBrowser -and $configLines[$i] -match '^  enabled: true$') { $configLines[$i] = '  enabled: false' }
        if ($configLines[$i] -match '^dataRoot:') { $configLines[$i] = 'dataRoot: ../../outputs/sillytavern-data' }
        if ($configLines[$i] -match '^port: 8000$') { $configLines[$i] = "port: $Port" }
    }
    Set-Content -Path $configPath -Value $configLines -Encoding utf8
}

if (-not (Test-Path (Join-Path $stRoot 'node_modules'))) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw '需要安装 Node.js 20 或更新版本（包括 npm），再运行此脚本。'
    }
    Push-Location $stRoot
    try {
        npm install --no-save --no-audit --no-fund --omit=dev --ignore-scripts
        if ($LASTEXITCODE -ne 0) { throw 'SillyTavern 运行依赖安装失败。' }
    }
    finally { Pop-Location }
}

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) { throw '没有找到 Node.js。请安装 Node.js 20 或更新版本。' }
$nodeMajor = [int]((& node --version).TrimStart('v').Split('.')[0])
if ($nodeMajor -lt 20) { throw 'SillyTavern 需要 Node.js 20 或更新版本。' }

Push-Location $stRoot
try {
    $env:NODE_ENV = 'production'
    & node server.js --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'SillyTavern 已退出，请查看上方错误信息。' }
}
finally { Pop-Location }
