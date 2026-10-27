$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot 'outputs\mnn-export-env\Scripts\python.exe'
$mnnRuntime = Join-Path $projectRoot 'outputs\mnn-runtime-3.5.0'
$mnnSource = Join-Path $projectRoot 'outputs\mnn-3.5.0-source'
$noTelemetryEntry = Join-Path $projectRoot 'outputs\mnn-export-no-telemetry.py'
$baseModel = Join-Path $projectRoot 'outputs\models\qwen3-1.7b'
$destination = Join-Path $projectRoot 'outputs\models\qwen3-1.7b-base-mnn-3.5.0-q4b64-attempt2'

foreach ($requiredPath in @($python, $mnnRuntime, $mnnSource, $noTelemetryEntry, $baseModel)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required export input is missing: $requiredPath"
    }
}
if (Test-Path -LiteralPath $destination) {
    throw "Destination already exists; choose a new output path instead of overwriting it: $destination"
}

$previousPythonPath = $env:PYTHONPATH
$previousMnnSource = $env:MNN_EXPORT_SOURCE
$previousPythonIoEncoding = $env:PYTHONIOENCODING
try {
    # Select both the 3.5.0 Python native converter and matching upstream exporter.
    $env:PYTHONPATH = $mnnRuntime
    $env:MNN_EXPORT_SOURCE = $mnnSource
    # The upstream spinner prints Unicode symbols that the default Windows GBK stream rejects.
    $env:PYTHONIOENCODING = 'utf-8'

    & $python $noTelemetryEntry `
        --path $baseModel `
        --dst_path $destination `
        --export mnn `
        --quant_bit 4 `
        --quant_block 64 `
        --lm_quant_bit 4 `
        --lm_quant_block 64

    if ($LASTEXITCODE -ne 0) {
        throw "MNN 3.5.0 export failed with exit code $LASTEXITCODE. Original exports were not modified."
    }
}
finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }

    if ($null -eq $previousMnnSource) {
        Remove-Item Env:MNN_EXPORT_SOURCE -ErrorAction SilentlyContinue
    }
    else {
        $env:MNN_EXPORT_SOURCE = $previousMnnSource
    }

    if ($null -eq $previousPythonIoEncoding) {
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONIOENCODING = $previousPythonIoEncoding
    }
}
