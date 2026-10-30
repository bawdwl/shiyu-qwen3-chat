$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$toolchain = Join-Path $root "outputs/android-toolchain"
$jdk = if ($env:JAVA_HOME) { $env:JAVA_HOME } else { Join-Path $toolchain "jdk17/jdk-17.0.20.1+1" }
$sdk = if ($env:ANDROID_HOME) {
    $env:ANDROID_HOME
} elseif ($env:ANDROID_SDK_ROOT) {
    $env:ANDROID_SDK_ROOT
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "Android/Sdk"
} else {
    Join-Path $toolchain "sdk"
}
$gradleHome = if ($env:GRADLE_USER_HOME) {
    $env:GRADLE_USER_HOME
} elseif (Test-Path (Join-Path $toolchain "gradle-home-current")) {
    Join-Path $toolchain "gradle-home-current"
} else {
    Join-Path $env:USERPROFILE ".gradle"
}

if (-not $jdk -or -not (Test-Path (Join-Path $jdk "bin/java.exe"))) {
    throw "未找到 JDK：请安装 JDK 17，并设置 JAVA_HOME。当前值：$jdk"
}
if (-not (Test-Path (Join-Path $sdk "platforms/android-35/android.jar"))) {
    throw "未找到 Android SDK 35：请设置 ANDROID_HOME，或在 Android Studio SDK Manager 安装 Android 15 / API 35。当前值：$sdk"
}

$env:JAVA_HOME = $jdk
$env:ANDROID_HOME = $sdk
$env:GRADLE_USER_HOME = $gradleHome
$env:Path = "$jdk\bin;$env:Path"

& (Join-Path $root "android/gradlew.bat") -p (Join-Path $root "android") assembleDebug --no-daemon
if ($LASTEXITCODE -ne 0) {
    throw "Android 构建失败"
}

$apk = Join-Path $root "android/app/build/outputs/apk/debug/app-debug.apk"
Get-Item $apk
Get-FileHash $apk -Algorithm SHA256
