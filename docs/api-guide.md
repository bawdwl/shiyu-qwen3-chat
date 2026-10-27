# FastAPI与SQLite后端使用

本文保留后端核心接口说明。后续已完成网页API接入、SSE、协作停止及超时，最新启动与验收见streaming-guide.md。当前app.py只调用后端，可与backend.py同时运行；历史直推入口local_app.py不要与后端同时加载模型。

```powershell
conda activate qwen3-im-chat
python -m pip install -r requirements-api.txt
python backend.py --adapter-run outputs/lora-data-v2
```

启用LoRA还需requirements-train.txt中的PEFT和已存在的Adapter；不传--adapter-run时仅提供Base。服务只监听127.0.0.1:8000，一个进程、一个模型实例，不开reload和多个worker。运行后访问 http://127.0.0.1:8000/docs 查看交互接口文档。Ctrl+C停止。

## 接口

| 方法与路径 | 用途 |
|---|---|
| GET /health | 模型是否就绪、可选模式、是否有生成请求 |
| POST /sessions | 创建会话，指定base或lora |
| GET /sessions?limit=50&offset=0 | 分页列出会话 |
| GET /sessions/{id} | 获取模型模式、时间与全部消息历史 |
| DELETE /sessions/{id} | 永久删除整段会话及其消息 |
| POST /chat | 按会话组织历史、生成并保存问答 |
| DELETE /sessions/{id}/messages | 清空该会话消息，保留会话本身 |
| POST /characters/generate | 调本机 Qwen 生成待审核角色卡草稿，不会自动保存 |
| POST /characters | 保存作者审核过的角色卡及世界书 |
| GET /characters/{id} | 读取已保存的自定义角色卡供编辑 |
| DELETE /characters/{id} | 删除自定义角色卡；既有聊天历史仍保留 |
| GET/PUT /characters/{id}/voice-profile | 读取或保存角色音色名称、风格与参考音频逐字稿 |
| PUT /characters/{id}/voice-profile/reference-audio | 上传该角色的 PCM WAV 参考音频（最多20 MiB） |

## 自定义角色卡与世界书

角色生成与正常聊天共用当前进程内的本地 Qwen；没有使用云端生成服务，也不依赖 SillyTavern API。浏览器调用此本机 FastAPI 接口。

生成草稿：

```json
{"inspiration":"海边小城的夜班电台主持人，喜欢听听众点歌，聊天时自然温和"}
```

请求 `POST /characters/generate`。它会返回 `card` 草稿。模型补充的具体事实默认标为 `proposal`，需作者在网页编辑器中确认后再保存。此接口不写入数据库。角色卡保存使用 `POST /characters`，将生成结果放入 `{"card": ...}`。角色卡和世界书条目存入默认 SQLite 数据库；自定义世界书按关键词命中后才追加到该轮系统提示，单轮最多追加 4 条。`GET /characters` 同时返回内置角色与自定义角色。每段会话绑定创建时的 `character_id`。

删除角色卡只从可选角色列表和运行时注册表移除，不会删除关联的会话记录；读取历史仍可用，但绑定已删除角色的会话不能继续生成，会返回 410，避免静默退化成通用助手。当前没有账号隔离或访问控制，服务只应监听本机地址。

2026-10-23新增会话筛选：`GET /sessions?limit=20&offset=0&nonempty=true&q=关键词`。`nonempty`默认false以兼容旧客户端；true时先过滤无消息会话再分页。`q`最长200字符，去掉首尾空白，对所有已保存用户/助手消息做不区分ASCII大小写的字面子串搜索，百分号、下划线不作通配符。空关键词不筛选。响应保留`sessions`数组，新增`has_more`布尔值。按更新时间倒序、ID升序排列；这是offset分页，多窗口同时改动记录时不是固定快照。

`DELETE /sessions/{id}` 与“清空消息”不同，会删除会话本身以及 SQLite 外键级联的消息；当前有生成任务时返回 409。前端历史卡片提供删除按钮，并在确认后执行此操作。GPT-SoVITS 配置下，播放回答时会按会话角色读取该角色的参考音频档案；默认小满档案需作者提供/授权 WAV 和完全对应的逐字稿。当前 MeloTTS 不使用角色档案，仍以共享音色朗读。

创建会话的JSON请求：

```json
{"model_mode":"base"}
```

聊天请求中session_id替换成创建接口返回的id：

```json
{"session_id":"替换为实际会话UUID","message":"请简单解释什么是模型上下文。"}
```

模式绑定会话，/chat不接受model_mode参数。切换模型时创建新会话，避免混合上下文。当前没有修改会话模式的接口。

当前/chat和/chat/stream可传max_new_tokens（严格整数1–1024，不传仍256）。会话列表新增title，从首条用户问题提取摘要；没有消息时为“新对话”。网页默认512及当前验收见[聊天体验升级记录](聊天体验升级记录.md)。

PowerShell调用示例：

```powershell
$session = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/sessions' -Method Post -ContentType 'application/json' -Body '{"model_mode":"base"}'
$payload = @{session_id=$session.id; message='请记住测试代号：河石。'} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body $payload
Invoke-RestMethod -Uri "http://127.0.0.1:8000/sessions/$($session.id)"
```

## 存储与一致性

默认数据库outputs/chat.sqlite3，可通过--db指定其他本地文件。sessions记录UUID、模式、创建与更新时间；messages记录会话ID、角色、内容和UTC时间。每次操作创建并关闭自己的连接，开启外键并使用参数化SQL。会话消息按自增ID排序。

完整生成成功后，将用户消息和助手回答在一个事务中一起写入；异常、空回答或写入失败不保存半轮问答。生成期间可以读取已完成历史，但不暴露尚未保存的部分。清空会话与生成互斥，避免旧请求结束后把清空前的对话写回来。

服务对生成和清空使用同一非阻塞门锁；有生成请求时，第二个生成或清空返回409，不堆积GPU任务。健康检查仍可响应。门锁仅在一个进程中生效，不是分布式并发控制。SQLite/WAL不等同生产级高并发数据库。

## 错误语义

| 状态码 | 情形 |
|---|---|
| 400 | 请求LoRA会话但服务未加载Adapter |
| 404 | 会话不存在 |
| 409 | 当前忙于生成或清空，请稍后重试 |
| 422 | 空白、超过4096字符、非法字段/模式/UUID，或提示词超过2048 token |
| 500 | 模型生成异常或空回答，未保存该轮历史 |
| 503 | 模型未就绪、显存不足或数据库暂不可用 |

模型加载失败时记录完整终端异常，/health返回503，已有历史仍可读取；不会假装模型已就绪。数据库初始化失败会阻止启动。生成异常详细堆栈保存在终端，接口返回简短提示。

## 测试与验收

```powershell
python -m unittest discover -s tests -p test_backend.py -v
python smoke_api.py
```

单元与集成测试使用临时SQLite文件、显式FakeEngine，无需GPU。17项全部通过：多轮输入构造、不同会话隔离、模式绑定、无效输入、未知会话、清空、分页、模拟生成/上下文/OOM/存储错误、异常不写入、事务回滚、新应用读取持久化记录、忙碌请求冲突和模型启动失败。

smoke_api.py要求真实服务已启动，会新建自己的合成测试会话，向Base与LoRA各发两轮问题，读取4条历史、测试空输入与404，再清空它刚创建的消息。本次两种模式都正确回忆map_73，历史存取与清空均通过；结果见api-smoke-result.json。实际推理显存与耗时是单次功能记录，不是性能基准。模拟故障不是实际GPU或数据库故障。

## 当前边界

当前为无认证的本机单用户开发Demo，无幂等请求、会话授权或多进程锁。新增/chat/stream有协作停止与超时，原同步/chat仍不支持取消。每次最多生成256 token不等于硬执行时间上限；如未收到完成事件，应先读取历史再决定是否重发。不要据功能测试宣称高并发部署经验。

SQLite真实聊天记录属于本地运行数据，数据库与辅助文件已被.gitignore排除，打包脚本也不收录outputs。截图、测试JSON只使用合成内容。

Gradio已连接API，相关流式测试与新增接口详见streaming-guide.md；本文件的17项为最初后端核心测试记录，当前总计25项。

官方实现参考：https://fastapi.tiangolo.com/advanced/events/ 、https://fastapi.tiangolo.com/advanced/testing-events/ 、https://docs.python.org/3/library/sqlite3.html
