# 许可与本地材料边界

更新：2026-10-30。当前项目代码采用根目录LICENSE中的AGPL-3.0。历史MIT许可保存在archive/original-materials/LICENSE-MIT-LEGACY；本次清理不改变旧版本已取得的授权。

拾语使用自己的HTML/CSS/JavaScript页面，参考SillyTavern的角色卡、世界书与提示组织思路。SillyTavern参考源码仍在third_party/SillyTavern，其上游AGPL许可与版权声明保留；当前拾语运行不依赖酒馆服务。角色图片来源记录在web/portraits/README.md。

GPT-SoVITS实际源码已安装在third_party/GPT-SoVITS，本地上游LICENSE为MIT；拾语通过本机API调用它。模型权重、语音参考素材与代码许可分别适用，不能根据源码许可推出任意声音可公开。

PC记忆检索可选用 `BAAI/bge-small-zh-v1.5`。该模型卡当前标注MIT，模型权重不属于拾语代码许可；发布或再分发前仍须重新核对上游模型卡、文件哈希和许可原文。模型保存在被Git忽略的 `outputs/models/`。

Android角色头像使用 `com.google.mediapipe:tasks-vision-image-generator:0.10.26.1`。MediaPipe Tasks运行库遵循其上游许可；可选的 Stable Diffusion 1.5 MediaPipe 转换权重由 `scripts/fetch_mediapipe_sd15.py` 单独下载，采用 CreativeML OpenRAIL-M，并按该许可的使用限制分发。模型不包含在APK、Git仓库或普通发布包中。

Qwen3、MNN/GPTQ衍生产物、Whisper、MeloTTS、PyTorch、Transformers、PEFT以及其他依赖，保留各自上游许可和来源信息。项目许可不覆盖第三方模型或数据。当前公开代码仓库不含模型权重、Adapter、个人录音和聊天数据库。

本机保留LCCC、NaturalConv、RoleBench等数据审查及历史实验，存在来源、研究用途和原作素材边界；原文、派生数据和相关Adapter不因本机保存就获得公开分发授权。旧项目原创教学样例保留其原许可记录，不把该授权扩大到第三方数据。

archive包含旧代码/文档、用户原始材料及历史研究记录，outputs包含私人音频、数据库、模型和环境；两者默认不公开。以后准备新发布时仍须按具体文件、数据与权重复核。旧打包脚本已经归档，不能拿旧ZIP当当前可公开交付包。
