[CmdletBinding()]
param(
    [string]$AdbPath = "",
    [string]$DeviceSerial = "",
    [string]$ModelDirectory = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$remoteRoot = "/data/local/tmp/mnn_models"

if (-not $ModelDirectory) {
    $ModelDirectory = Join-Path $projectRoot "outputs\models\qwen3-1.7b-base-mnn-3.6.1-q4b64"
}
if (-not (Test-Path -LiteralPath $ModelDirectory -PathType Container)) {
    throw "项目自导出的 MNN 模型目录不存在：$ModelDirectory"
}
$modelPath = (Resolve-Path -LiteralPath $ModelDirectory).Path
$modelName = Split-Path -Leaf $modelPath
if ($modelName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "模型目录名只能包含英文字母、数字、点、下划线和连字符。"
}
$remoteModelPath = "$remoteRoot/$modelName"

if (-not $AdbPath) {
    $adbCommand = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($adbCommand) {
        $AdbPath = $adbCommand.Source
    } else {
        $bundledAdb = Join-Path $projectRoot "outputs\platform-tools\adb.exe"
        if (Test-Path -LiteralPath $bundledAdb -PathType Leaf) {
            $AdbPath = $bundledAdb
        } else {
            throw "找不到 adb.exe。请安装 Android SDK Platform-Tools，或通过 -AdbPath 指定 adb.exe 的完整路径。"
        }
    }
}

if (-not (Test-Path -LiteralPath $AdbPath -PathType Leaf)) {
    throw "adb.exe 不存在：$AdbPath"
}
$AdbPath = (Resolve-Path -LiteralPath $AdbPath).Path

$deviceLines = @(& $AdbPath devices -l 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "ADB 无法启动设备服务：$($deviceLines -join " ")"
}
$connectedSerials = @(
    foreach ($line in $deviceLines) {
        if ($line -match '^([^\s]+)\s+device(?:\s|$)') { $Matches[1] }
    }
)

if (-not $DeviceSerial) {
    if ($connectedSerials.Count -ne 1) {
        throw "需要恰好一台已授权的 Android 设备；当前识别到 $($connectedSerials.Count) 台。请用 -DeviceSerial 指定。ADB 输出：$($deviceLines -join " | ")"
    }
    $DeviceSerial = $connectedSerials[0]
} elseif ($connectedSerials -notcontains $DeviceSerial) {
    throw "设备 $DeviceSerial 未处于已授权的 device 状态。ADB 输出：$($deviceLines -join " | ")"
}

Write-Host "目标设备：$DeviceSerial"
Write-Host "项目模型：$modelPath"
Write-Host "手机目录：$remoteModelPath"

& $AdbPath -s $DeviceSerial shell mkdir -p $remoteRoot
if ($LASTEXITCODE -ne 0) { throw "无法在手机创建 MNN Chat 本地模型目录。" }

& $AdbPath -s $DeviceSerial push $modelPath $remoteRoot
if ($LASTEXITCODE -ne 0) { throw "模型传输失败。" }

$localFiles = @(Get-ChildItem -LiteralPath $modelPath -File)
if ($localFiles.Count -eq 0) { throw "本地模型目录中没有模型文件。" }

foreach ($file in $localFiles) {
    $expectedHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $remoteFile = "$remoteModelPath/$($file.Name)"
    $hashOutput = @(& $AdbPath -s $DeviceSerial shell sha256sum $remoteFile 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "无法读取手机端文件校验值：$remoteFile；输出：$($hashOutput -join " ")"
    }
    $match = [regex]::Match(($hashOutput -join " "), '(?i)\b[0-9a-f]{64}\b')
    if (-not $match.Success -or $match.Value.ToLowerInvariant() -ne $expectedHash) {
        throw "文件校验不一致：$($file.Name)"
    }
    Write-Host "SHA-256 已核对：$($file.Name)"
}

Write-Host "模型已传输并完成逐文件 SHA-256 校验。请在 MNN Chat 的本地模型列表中选择 $modelName。"
