# 拾语：本地角色聊天

拾语是一个运行在个人电脑上的角色聊天网页。它使用本地的 Qwen3-1.7B 生成回答，网页和 FastAPI 后端负责角色设定、流式聊天与 SQLite 会话记录。模型运行在自己的电脑上，无需购买聊天 API。

目前已实现：选择或创建角色、编辑角色卡与关键词世界书、多轮流式聊天、保存和删除聊天记录、同一会话的记忆整理。记忆可能遗漏重要信息；新会话从空的动态记忆开始。更多说明见[角色卡与会话记忆](docs/角色卡与会话记忆.md)。

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

上面的命令只启动文字聊天。语音识别与语音合成需要另行准备模型和运行环境，目前没有提供完整的一键安装流程；不要把默认文字启动误认为已经配置好了语音。

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
