param(
    [string]$Variant = 'standardDebug'
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent $PSScriptRoot
$androidRoot = Join-Path $repoRoot 'android'
$toolchainRoot = Join-Path $repoRoot 'outputs/android-toolchain'

# Reuse the developer's Android debug key so rebuilt APKs can update an already
# installed local copy without uninstalling the app and losing its data.
if (-not $env:SHIYU_DEBUG_KEYSTORE -and $env:USERPROFILE) {
    $userDebugKeystore = Join-Path $env:USERPROFILE '.android/debug.keystore'
    if (Test-Path $userDebugKeystore) { $env:SHIYU_DEBUG_KEYSTORE = $userDebugKeystore }
}

if (-not $env:JAVA_HOME) {
    $localJdk = Join-Path $toolchainRoot 'jdk17-unpacked/jdk-17.0.20.1+1'
    if (Test-Path $localJdk) { $env:JAVA_HOME = $localJdk }
}

if (-not $env:ANDROID_SDK_ROOT) {
    $localSdk = Join-Path $toolchainRoot 'sdk'
    if (Test-Path $localSdk) { $env:ANDROID_SDK_ROOT = $localSdk }
}
if ($env:ANDROID_SDK_ROOT) { $env:ANDROID_HOME = $env:ANDROID_SDK_ROOT }

if (-not $env:ANDROID_NDK_HOME -and $env:ANDROID_SDK_ROOT) {
    $localNdk = Join-Path $env:ANDROID_SDK_ROOT 'ndk/27.2.12479018'
    if (Test-Path $localNdk) { $env:ANDROID_NDK_HOME = $localNdk }
}
if (-not $env:GRADLE_USER_HOME) {
    $localGradleHome = Join-Path $toolchainRoot 'gradle-home'
    if (Test-Path $localGradleHome) { $env:GRADLE_USER_HOME = $localGradleHome }
}

if ($env:GRADLE_BIN) {
    $gradle = $env:GRADLE_BIN
} else {
    $localGradle = Join-Path $toolchainRoot 'gradle-unpacked/gradle-8.9/bin/gradle.bat'
    if (Test-Path $localGradle) {
        $gradle = $localGradle
    } else {
        $gradle = Join-Path $androidRoot 'gradlew.bat'
    }
}

if (-not $env:JAVA_HOME -or -not (Test-Path $env:JAVA_HOME)) {
    throw 'JDK 17 not found. Set JAVA_HOME.'
}
if (-not $env:ANDROID_SDK_ROOT -or -not (Test-Path $env:ANDROID_SDK_ROOT)) {
    throw 'Android SDK not found. Set ANDROID_SDK_ROOT.'
}
if (-not (Test-Path $gradle)) { throw "未找到 Gradle：$gradle" }

Push-Location $androidRoot
try {
    & $gradle "assemble$Variant" --no-daemon --console plain
    if ($LASTEXITCODE -ne 0) { throw "Android 构建失败，Gradle exit code=$LASTEXITCODE" }
} finally {
    Pop-Location
}
