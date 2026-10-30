# 拾语 Android

当前版本：`0.9.4 / versionCode 14`

技术栈：Kotlin、Jetpack Compose、Room schema 10、WorkManager、OkHttp、Android Keystore、sherpa-onnx。

## 功能

- OpenCode Go `deepseek-v4.1-flash` 与 DeepSeek `deepseek-flash` 两个独立 Provider。
- 每个 Provider 的 API Key 单独加密保存。
- 自定义角色、角色卡草稿、世界书、知识边界和固定事实。
- 每个角色一个持续对话，创建或重置时插入主动剧情开场。
- 原生流式回答、停止生成与图片输入。
- 自动结构化长期记忆、WorkManager 持久任务、事实有效期和删除墓碑。
- FTS4 中文单字、二元组和三元组检索。
- 内部心情、熟悉、信任、温暖、关系阶段和用户边界。
- sherpa-onnx 本地 ASR、Kokoro 本地 TTS 与多音色播放。
- MediaPipe SD1.5 本地头像与千问 Qwen-Image 云端头像。
- 中文外貌描述，以及首次本地生图时的视觉提示缓存。
- 删除、重置和编辑角色；删除时清理关联消息、记忆、图片和状态。

## 构建

前置条件：

- JDK 17
- Android SDK 35
- Git LFS

从仓库根目录：

```powershell
git lfs pull
$env:JAVA_HOME = "C:\path\to\jdk-17"
$env:ANDROID_HOME = "C:\Users\<you>\AppData\Local\Android\Sdk"
.\scripts\build_android.ps1
```

或者直接使用 Android Studio 打开本目录。

输出：

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

## 本地模型

APK 中包含本地语音所需的 AAR、ASR 和 Kokoro 模型。MediaPipe SD1.5 约 1.78 GiB，不包含在 APK 中：

```powershell
python scripts/fetch_mediapipe_sd15.py
```

然后在拾语“应用设置”中导入生成出的模型 ZIP。

## 数据与安全

- API Key 使用 Android Keystore/EncryptedSharedPreferences。
- 角色、消息、记忆、状态和头像路径保存在 Room/应用私有存储。
- 没有账号系统，不与 PC 自动同步。
- 清除 API Key 不会删除历史聊天。
- 重置或删除角色会删除该角色的关联数据。

## 已知限制

- Android 尚未内置 BGE，只使用 FTS4 + 词法排序和可选语义接口。
- 摘要编辑和版本恢复尚未与 PC 完全对齐。
- 当前是录音后识别，不是实时双向语音。
- 反思、人物成长、长期目标、Live2D 和剧情 World Clock 尚未实现。
- ColorOS 15 在强制结束进程后可能暂缓 WorkManager 恢复；重新打开应用后会继续待处理任务。
- APK 约 441 MiB，主要来自本地语音运行库和模型。

