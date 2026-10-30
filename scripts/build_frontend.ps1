$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "frontend")
try {
    pnpm install --frozen-lockfile
    pnpm build
}
finally {
    Pop-Location
}
