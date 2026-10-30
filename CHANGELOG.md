# Changelog

## 2026-10-30

### PC

- 将前端升级为 React + TypeScript + Vite，FastAPI 继续提供接口与本机存储。
- 加入 OpenCode Go、DeepSeek 和保留的本地 Qwen 兼容 Provider。
- 加入角色卡、世界书、知识边界、固定事实和主动剧情开场。
- 加入自动结构化长期记忆、后台持久任务、事实更正、有效期和删除墓碑。
- 加入 FTS5 + 本地 BGE + 词法 + 时间及权重的统一检索。
- 加入内部心情、熟悉、信任、温暖、关系阶段和用户边界。
- 加入 Whisper、GPT-SoVITS 和 MeloTTS 语音链路。
- 撤销四段式角色表演，恢复单条自然聊天回复。
- 移除网页端本地 SD1.5 生图。

### Android

- 发布 Kotlin + Jetpack Compose + Room 客户端 `0.9.4`。
- 支持直接调用 OpenCode Go 与 DeepSeek，不依赖电脑中转。
- 同步角色、世界书、知识边界、自动记忆、FTS4、内部状态和主动开场。
- 加入 WorkManager 持久记忆任务。
- 加入 sherpa-onnx ASR、Kokoro TTS 与多音色播放。
- 加入 MediaPipe SD1.5 本地头像和千问 Qwen-Image 云端头像。
- 新增中文外貌描述和本地生图提示词缓存。
- 修复取消新角色编辑后资料卡串角色的问题。

### Repository

- 重写 README 和双端复现说明。
- 移除 API Key、聊天数据库、个人录音、模型实验和本地输出。
- 换成可用的 `.gitignore` 与 Git LFS 大文件规则。
- 增加 Android 可安装 APK 和 SHA-256。

