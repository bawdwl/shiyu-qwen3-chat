"""启动本地聊天网页：conda activate qwen3-im-chat 后执行 python app.py。"""
import logging
import argparse

import gradio as gr

from conversation import clear_conversation, submit_message, switch_model
from inference import QwenChat


def build_app(engine):
    with gr.Blocks(title='Qwen3 中文聊天') as demo:
        gr.Markdown('# Qwen3 中文聊天\n在本机运行的 Qwen3-0.6B。支持连续追问，回答可能有误。')
        # State 为每个浏览器会话分别保存历史，模型只加载一份。
        history = gr.State([])
        choices = [('Base Qwen3', 'base')]
        if engine.adapter_available:
            choices.append(('LoRA 微调版', 'lora'))
        mode = gr.Radio(choices, value='base', label='当前模型（切换会清空会话）')
        gr.Markdown('LoRA 是小规模实验版本，已观察到部分算术退化；切换后可输入相同问题进行比较。')
        chatbot = gr.Chatbot(label='对话', height=450, buttons=[])
        message = gr.Textbox(label='你的问题', placeholder='例如：请用简单的话解释什么是大语言模型。', lines=2, max_lines=6)
        with gr.Row():
            send = gr.Button('发送', variant='primary')
            clear = gr.Button('清空会话')
        status = gr.Markdown('已就绪。输入问题后点击发送；生成期间请等待。')

        def respond(text, state, selected):
            return submit_message(text, state, engine, selected)

        outputs = [message, history, chatbot, status]
        # 发送和清空共用队列，清空不会在旧请求结束后被旧结果覆盖。
        send.click(respond, [message, history, mode], outputs,
                   concurrency_limit=1, concurrency_id='chat', api_name='send')
        clear.click(clear_conversation, None, outputs,
                    concurrency_limit=1, concurrency_id='chat', api_name='clear')
        chatbot.clear(clear_conversation, None, outputs,
                      concurrency_limit=1, concurrency_id='chat', api_name=False)
        mode.change(switch_model, mode, outputs, concurrency_limit=1,
                    concurrency_id='chat', api_name='switch_model')
    return demo.queue(max_size=8)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-run', default=None,
                        help='训练输出目录，例如 outputs/lora-v2；不传时仅提供基础模型')
    args = parser.parse_args()
    print('正在从本地缓存加载 Qwen3-0.6B，请稍候……', flush=True)
    engine = QwenChat(adapter_run=args.adapter_run)
    build_app(engine).launch(server_name='127.0.0.1', server_port=7860,
                             share=False, inbrowser=False, show_error=False)
