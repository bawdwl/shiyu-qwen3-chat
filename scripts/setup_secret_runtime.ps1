$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\Others\Study\Anaconda3\envs\qwen3-im-chat\python.exe"
if (-not (Test-Path $python)) {
    throw "找不到 qwen3-im-chat Python：$python"
}
& $python -m pip install --target (Join-Path $root "outputs/secret-runtime") cryptography
