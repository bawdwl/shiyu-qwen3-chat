# Android 0.9.3 同步与真机升级

更新：2026-10-30

## 设备与安装

- 设备：OnePlus PKG110（一加 Ace 5），Android 15。
- 覆盖前：`0.8.0 / versionCode 9`。
- 最终版本：`0.9.3 / versionCode 13`。
- Room：schema 9。
- 包名：`com.shiyu.chat`。

新旧 APK 签名均为调试证书，证书 SHA-256：
`52f6cd4bade53b5eee05e5aaf9fb4daab26e9b546ae359bdc5049a410058f439`。

## 数据备份

覆盖前已备份应用数据库、WAL、SHM、角色音色、设置偏好和 0.8.0 APK。

备份目录：`outputs/backups/android-v080-before-092/`。

覆盖前数据库：

- schema 5，完整性 `ok`；
- 6 个角色；
- 70 条消息；
- 44 条结构化记忆；
- 15 个记忆任务和用量记录；
- 4 个角色记忆游标。

## 0.9 同步内容

- P0 自动记忆规则。
- FTS4 中文分词检索、事实有效期、重要度、可信度。
- 结构化事实与旧原话统一上下文预算。
- 内部心情、关系、边界和状态变化日志。
- 自然单条角色回复。
- 创建和重置后的主动剧情开场。
- 历史四段式消息兼容为单气泡。
- 本地 sherpa-onnx 语音识别与 Kokoro 语音输出。
- 本地 MediaPipe SD1.5 模型导入、按钮和生成链路。
- 千问 Qwen-Image 云端生图。
- 角色创建、编辑、删除、重置和长期记忆管理。

## 真机验收

升级结果：

- Room 从 schema 5 直接迁移到 9；
- `PRAGMA integrity_check = ok`；
- 升级时角色、消息、结构化记忆和后台任务均保留；
- FTS4 表存在且已有记录完成 `searchTokens` 回填；
- 角色 `greeting` 和消息 `displayJson` 字段存在；
- 旧聊天继续以单条气泡显示。

MediaPipe SD1.5：

- 设置页识别到本地模型已导入；
- 角色编辑页同时显示“本地生成头像”和“云端生成头像”；
- 对没有头像的“陆景行”实际执行本地生成并成功更新头像；
- 进程保持运行，未发现 MediaPipe/ONNX 原生崩溃；
- 千问云端生图入口保留。

最终设备数据库样本：

- schema 9，完整性 `ok`；
- 7 个角色、79 条消息、51 条结构化记忆；
- 51 条 FTS4 索引；
- 已产生内部状态和变化记录。

## 证据

- `outputs/android-release/shiyu-android-0.9.3.apk`
- `outputs/android-release/0.9.3-postinstall/home.png`
- `outputs/android-release/0.9.3-postinstall/settings-avatar.png`
- `outputs/android-release/0.9.3-postinstall/local-avatar-result.png`
- `outputs/android-release/0.9.3-postinstall/final-summary.json`

## 当前差异

- Android 仍使用 FTS4 和确定性词法排序，没有随 APK 打包 PC 的 BGE/ONNX 语义模型。
- Android 语音使用本地 Kokoro 四音色，不是 PC 的 GPT-SoVITS 参考音色体系。
- Android 继续保留 MediaPipe 本地生图和千问云端生图；PC 网页端本地生图已移除。
- Android 与 PC 的聊天、记忆和设置仍分别存储，不做自动同步。
