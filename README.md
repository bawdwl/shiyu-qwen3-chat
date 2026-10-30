# 拾语 SHIYU

拾语是一套面向中文虚拟角色陪伴场景的 PC 与 Android 双端聊天应用。它把云端大模型、角色卡、世界书、长期记忆、情绪关系状态、语音输入和角色语音组织成一条可运行的聊天链路。

当前主线使用 **OpenCode Go `deepseek-v4.1-flash`**，同时支持独立 DeepSeek Provider。API Key 由使用者在本机填写，不会写入源码、数据库快照或 Git 仓库。

![拾语 PC 首页](docs/images/shiyu-home-deepseek-2026-10-28.png)

## 2026-10-30 更新

本次更新把仓库从早期的本地 Qwen 单页原型升级为当前双端产品代码：

- PC 迁移到 React + TypeScript + Vite + FastAPI + SQLite。
- Android 加入 Kotlin + Jetpack Compose + Room 客户端。
- 两端支持 OpenCode Go 与 DeepSeek，可分别保存 Key 并切换。
- 新增角色卡、世界书、知识边界、固定事实和唯一持续对话。
- 新增自动结构化长期记忆、任务重试、事实有效期和删除墓碑。
- PC 加入 FTS5 + 本地 BGE 混合检索；Android 使用 FTS4 与词法排序回退。
- 双端加入受限更新且不默认展示数值的内部状态。
- 双端恢复单条自然回复，动作写在中文括号中；创建或重置角色时生成主动剧情开场。
- PC 加入 Whisper、GPT-SoVITS 与 MeloTTS 适配；Android 加入 sherpa-onnx ASR、Kokoro TTS。
- Android 保留 MediaPipe SD1.5 本地头像和千问 Qwen-Image 云端头像。
- PC 网页本地生图入口已移除，只保留已有头像读取。
- 更新复现文档、测试脚本、密钥排除规则和 Android 可安装包。

完整现状与限制见 [项目现状审计](docs/项目现状与网页建议落实审计_2026-10-30.md)，安装细节见 [使用与复现](docs/使用与复现.md)。

## 已实现功能

### 聊天与角色

- 自定义角色、默认角色和角色卡生成草稿。
- 身份、背景、性格、说话方式、关系、互动边界和示例对话。
- 关键词触发世界书，固定事实优先于聊天中的即兴内容。
- 时代、知识截止、世界范围、允许/禁止知识与禁止词边界。
- 每个角色只有一个持续对话。
- PC SSE 流式回答、停止生成、超时与 Provider 错误提示。
- Android 原生流式读取、停止生成与本地历史。
- 图片输入，可交给支持视觉的云端模型。
- 删除、重置角色及其消息、记忆、图片和状态。

### 长期记忆

- 回复保存后自动创建持久后台任务，不阻塞当前聊天。
- 从对话中提取用户事实、角色自述、共同事件、计划和边界。
- 候选事实必须有来源消息 ID、说话人和逐字证据。
- 自动执行新增、合并、更正、疑义和忽略。
- 旧事实保留为历史版本；删除操作创建墓碑，旧原话不能自动复活。
- PC 使用 FTS5 + 可选本地 `BAAI/bge-small-zh-v1.5`。
- Android 使用 FTS4 中文分词，并保留可选语义 Provider 接口。
- 结构化事实、摘要和历史摘录使用统一上下文预算。
- 每个角色每天有独立的后台记忆 Token 预算。

### 内部状态

双端可保存心情、熟悉度、信任度、温暖度、关系阶段和用户边界。状态变化必须绑定证据，受单次变化幅度和自然衰减限制。状态作为自然语言提示影响角色表达，不默认在聊天界面展示数字。

### 语音与图片

| 能力 | PC | Android |
|---|---|---|
| 语音识别 | Whisper | sherpa-onnx 本地 ASR |
| 角色语音 | GPT-SoVITS + MeloTTS 回退 | Kokoro 本地 TTS，多音色 |
| 交互方式 | 录音/上传 → 识别 → 发送 → 回复后播放 | 录音 → 识别 → 发送 → 回复后播放 |
| 图片输入 | 支持 | 支持 |
| 图片生成 | 已移除 | MediaPipe SD1.5 本地生成、千问 Qwen-Image 云端生成 |

语音目前不是实时电话式双向通话，也没有自动打断。

## 快速使用 PC 端

### 环境

- Windows 10/11
- Python 3.11
- Git
- pnpm 或 Node.js 20+
- 可选：NVIDIA GPU。当前主链路使用云端 API，显卡不是必需条件。

### 安装

```powershell
git clone https://github.com/bawdwl/shiyu-qwen3-chat.git
cd shiyu-qwen3-chat

conda create -n shiyu-chat python=3.11 pip -y
conda activate shiyu-chat
python -m pip install -r requirements-lock-windows-py311.txt

pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build
```

仓库同时包含已经构建好的 `web/react-dist/`；修改前端源码后需要重新运行构建。

### 启动文字聊天

```powershell
python launch_app.py --provider opencode --speech-provider off --asr-provider off
```

打开 <http://127.0.0.1:8000/ui/>，在设置中选择 Provider 并填写自己的 API Key。API Key 通过本机服务保存到加密 Secret Store，不会发送回浏览器。

也可以改用 DeepSeek：

```powershell
python launch_app.py --provider deepseek --speech-provider off --asr-provider off
```

### 启用 PC BGE 语义记忆

```powershell
python scripts/fetch_embedding_model.py
```

模型缺失或运行失败时，应用会自动回退到 FTS5 和词法检索。

### 启用 PC 语音

PC 语音依赖单独的 Whisper、MeloTTS 或 GPT-SoVITS 模型与服务，不包含在 Git 仓库中。模型准备和启动方式见 [语音部署](docs/语音部署.md)。

## 安装 Android 端

已构建调试包：[shiyu-android-0.9.4.apk](release/shiyu-android-0.9.4.apk)

- 版本：`0.9.4 / versionCode 14`
- Room schema：10
- APK SHA-256：`EB05102FECF867446F6EF99472E816E9154046DF678C995F0D3A3C387D5B95D7`
- 安装后请在设置中填写自己的 OpenCode Go 或 DeepSeek Key。

APK 包含本地语音运行库和语音模型，因此体积约 441 MiB。MediaPipe SD1.5 生图模型约 1.78 GiB，不随 APK 分发，需要在手机中单独导入。

### 从源码构建 Android

需要 JDK 17、Android SDK 35 和 Android Studio 或等价工具链。

```powershell
$env:JAVA_HOME = "C:\path\to\jdk-17"
$env:ANDROID_HOME = "C:\Users\<you>\AppData\Local\Android\Sdk"
.\scripts\build_android.ps1
```

输出：

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

也可以直接在 Android Studio 中打开 `android/` 并运行 `app` 配置。

MediaPipe 生图模型下载：

```powershell
python scripts/fetch_mediapipe_sd15.py
```

下载后在拾语的“应用设置”中导入。云端头像需要另外填写千问图片 API Key。

## 数据与密钥

- 仓库不包含 API Key、聊天数据库、个人录音、角色生成图片、模型权重或 LoRA Adapter。
- PC 的 Key、数据库和语音档案默认位于 Git 忽略的 `outputs/`。
- Android 的 Key 使用 Android Keystore/EncryptedSharedPreferences。
- 角色、消息、记忆和状态保存在各自设备的 Room 或 SQLite 中。
- PC 与 Android 目前不会自动同步数据。
- 重置角色会删除该角色的消息、长期记忆、图片和状态。

如果发现任何泄露的凭证，请立即在 Provider 控制台撤销并更换。不要通过 Issue 或 Pull Request 上传 API Key。

## 测试

```powershell
python -B scripts/check_project.py
node --test tests/frontend_state.test.cjs
pnpm --dir frontend build
```

Android：

```powershell
cd android
.\gradlew.bat testDebugUnitTest assembleDebug --no-daemon
```

当前发布前验证结果：

- 公开仓库的生产链路 Python 测试：222 项通过。
- 本地完整研究目录此前的全量验证：376 项通过。
- PC 前端状态测试：39 项通过。
- React 生产构建通过。
- Android JVM：40 项通过。
- Android `0.9.4` debug APK 构建并在一加 Ace 5 完成覆盖安装。
- Room schema 10 迁移、FTS4 回填、MediaPipe 本地头像和千问云端头像入口已真机验证。

这些结果证明主要工程流程可运行，不等同于大规模真实聊天质量评估。

## 当前限制

- 还没有全新的留出对话集和独立盲评，不能只用现有测试证明角色自然度。
- Android 没有内置 BGE，语义召回弱于 PC。
- PC 与 Android 的摘要管理尚未完全对等。
- 反思记忆、人物成长、长期目标和剧情 World Clock 未完成。
- 认知边界尚未实现完整 K1–K5 与多角色隐藏事实隔离。
- 没有账号、多用户、跨设备同步、内容审核和生产级并发。
- 还没有完整浏览器 E2E、Android UI instrumentation 和长时间运行测试。
- `src/backend.py` 与 Android `ChatRepository.kt` 仍偏大。

## 目录

```text
launch_app.py                 PC 启动入口
src/                          FastAPI、Provider、记忆、提示词、存储与语音
frontend/                     React + TypeScript 源码
web/react-dist/               已构建 PC 前端
web/portraits/                默认角色图片与来源说明
android/                      Kotlin + Compose + Room Android 项目
tests/                        当前生产链路回归测试
scripts/                      构建、模型下载和检查脚本
docs/                         架构、验收、语音与复现文档
release/                      可安装 Android APK（Git LFS）
```

## 许可证

项目原创代码采用 [AGPL-3.0](LICENSE)。模型、模型转换产物、角色图片、语音模型和其他第三方组件分别遵循各自上游许可，详见 [LICENSE-NOTICE.md](LICENSE-NOTICE.md)。

