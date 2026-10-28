# 拾语：本地角色聊天

拾语是一个运行在个人电脑上的角色聊天网页。它使用本地的 Qwen3-1.7B 生成回答，网页和 FastAPI 后端负责角色设定、流式聊天与 SQLite 会话记录。模型运行在自己的电脑上，无需购买聊天 API。

## 已实现的功能

- **本地聊天**：Qwen3-1.7B 在电脑上生成回答，网页以流式方式显示文字。使用者可以停止生成、新建会话、查看或删除聊天记录；记录保存在本机 SQLite。
- **角色创建**：提供小满、言川两个默认角色。可以输入关键词让本地模型生成角色卡草稿，检查和修改身份、背景、性格、说话风格等内容后保存；自定义角色同样使用本地模型。
- **世界书和上下文**：角色可配置关键词触发的设定条目，聊天时与角色卡、近期对话一起组成提示。开场情境不再每一轮重复注入；可查看提示预览。
- **同会话记忆**：长对话接近输入预算时，系统可从较早消息中摘取原文，作为后续聊天的参考；支持查看、编辑、刷新和恢复。它可能漏记或选错，新会话不会继承旧会话的动态记忆。详见[角色卡与会话记忆](docs/角色卡与会话记忆.md)。
- **语音输入**：网页支持麦克风录音或上传录音，发送到本地 Whisper 识别后，再由使用者检查和发送文字。模型需要另行准备；网页直接录音的识别准确率曾低于上传已有录音，尚未完成系统评测。
- **角色语音输出**：回答完成后点击播放。后端已接入 GPT-SoVITS 本机服务，可按角色保存参考音频与对应文字，选择不同音色；未配置参考音色的角色可回退到本地 MeloTTS。作者已用自行录制的参考音频验证言川角色的网页朗读链路，试听反馈是可以听清；不代表所有音色都已验证。

语音是分步操作，尚不是边识别边回答的实时语音通话。开发机上 GPT-SoVITS 使用独立的 Python 环境和 CPU 服务，曾有约 7 秒语音需要约 13 秒合成的单次记录；尚未系统评测延迟、音质或 GPU 与聊天模型同时运行的显存。

## 在 Windows 上安装和启动

下面是文字聊天的完整步骤。项目开发环境为 Windows、Python 3.11 和 NVIDIA RTX 4060 Laptop 8GB；其他显卡或 CPU 环境未完整验证。首次安装需要下载 Python 依赖和数 GB 的模型文件，请预留足够磁盘空间。需要能够访问 GitHub、PyTorch 和 Hugging Face。

### 1. 获取代码

安装 Git 和 [Miniconda 或 Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)。打开 **Anaconda Prompt**，选择一个你想保存项目的位置，执行：

```powershell
git clone https://github.com/bawdwl/shiyu-qwen3-chat.git
cd shiyu-qwen3-chat
```

也可以在 GitHub 选择 **Code → Download ZIP**，解压后在终端进入解压出的文件夹。下面所有命令都在项目根目录执行。

### 2. 创建独立环境并安装依赖

```powershell
conda create -n shiyu-chat python=3.11 pip -y
conda activate shiyu-chat
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-lock-windows-py311.txt
```

安装后检查 Python 能否识别 NVIDIA 显卡：

```powershell
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

如果显示 `CUDA 可用: False`，请检查 NVIDIA 驱动和安装的 PyTorch 版本；不需要自行安装完整 CUDA Toolkit 才能使用带 CUDA 的 PyTorch。本项目尚未完成 CPU-only 部署验证。

### 3. 下载基础模型

```powershell
python src/fetch_chat_model.py
```

脚本从 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 下载固定版本到 `outputs/models/qwen3-1.7b/`，并记录文件信息。模型权重不包含在代码仓库里。若下载失败，先检查网络和 Hugging Face 访问；不要把不完整的下载当作模型已就绪。

### 4. 启动网页

```powershell
python launch_app.py --model-profile qwen3-1.7b --speech-provider off --asr-provider off
```

首次加载模型需要等待。看到终端提示打开网址后，访问 **http://127.0.0.1:8000/ui/**，选择角色并发送消息。保持终端窗口打开；在终端按 `Ctrl+C` 停止服务。接口文档位于 http://127.0.0.1:8000/docs。

上面的命令只启动文字聊天。GPT-SoVITS 功能已经在开发机接通，但新机器仍需另外安装官方 GPT-SoVITS、下载相应权重、准备独立运行环境和参考音频；这些文件不随本仓库提供。当前还没有经过验证的完整一键安装流程，**不能按上述命令直接获得语音功能**。

在已经按本项目路径准备好 GPT-SoVITS 服务、CPU 配置和角色参考音频的开发环境中，可使用下面的命令启动按角色音色路由及本地 Whisper：

```powershell
python launch_app.py --model-profile qwen3-1.7b --speech-provider profiles --asr-provider whisper
```

启动器会查找 `third_party/GPT-SoVITS/api_v2.py`、`outputs/gpt-sovits-env/python.exe` 和 `outputs/gpt-sovits-tts-cpu.yaml`；这些是开发机路径约定，而不是新机器的安装步骤。Whisper 与 MeloTTS 的模型也需要单独准备。参考音频及逐字稿必须有权使用，开发者的个人录音不会随仓库发布。[GPT-SoVITS 官方仓库](https://github.com/RVC-Boss/GPT-SoVITS)提供上游安装信息；本项目目前尚未完成从空环境配置到角色播放的复现说明。

## 日常再次使用

```powershell
conda activate shiyu-chat
cd 你的项目目录\shiyu-qwen3-chat
python launch_app.py --model-profile qwen3-1.7b --speech-provider off --asr-provider off
```

其中 `你的项目目录` 换成实际路径。已下载模型的情况下不需要重新下载。聊天记录默认保存在项目下的 `outputs/chat.sqlite3`，请自行备份。

## 常见问题

- **端口 8000 被占用或提示已有服务版本不一致**：先停止占用端口的旧拾语服务，再启动新版；不要同时启动两份模型服务。
- **启动失败或等待超时**：查看 `outputs/app-service.log`；确认模型下载完整、显卡可用且有足够显存。
- **回答不连贯或重复**：这是小模型及提示组织的局限。可以编辑角色卡、精简设定，或开启新对话；会话记忆不能保证完全准确。
- **角色生成内容不合适**：自动生成的是草稿，请在保存前检查并修改。

## 项目结构

```text
launch_app.py             启动入口
src/                      聊天接口、模型调用、角色卡与存储
web/                      网页代码及界面必需的角色资源
tests/                    自动测试
docs/                     功能说明
requirements-lock-windows-py311.txt  Windows 依赖版本
```

项目是单机、单用户原型，没有经过生产环境或全新机器完整安装验收。代码采用 [AGPL-3.0](LICENSE)；模型、角色素材和其他第三方组件各自遵循上游许可。
