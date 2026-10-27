"""会话更新规则：成功后才加入历史，出错时保留原对话和输入。"""
import logging

import torch


def submit_message(message, history, engine, model_mode='base'):
    history = history or []
    if not isinstance(message, str) or not message.strip():
        return message, history, history, '请输入问题后再发送。'
    if len(message) > 4096:
        return message, history, history, '输入过长，请缩短至 4096 个字符以内。'
    # 创建新列表；不修改 Gradio 为当前浏览器保存的原始状态。
    messages = [dict(item) for item in history]
    messages.append({'role': 'user', 'content': message.strip()})
    try:
        answer, metrics = engine.reply(messages, model_mode=model_mode)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return message, history, history, '显存不足。请关闭其他占用 GPU 的程序，清空会话后重试。'
    except ValueError as exc:
        return message, history, history, str(exc)
    except Exception:
        logging.exception('模型生成失败')
        return message, history, history, '生成失败，请重试；详细原因请查看启动终端。'
    messages.append({'role': 'assistant', 'content': answer})
    name = 'LoRA 微调版' if model_mode == 'lora' else 'Base Qwen3'
    status = f"{name} · 本轮耗时 {metrics['seconds']:.2f} 秒 · 输出 {metrics['output_tokens']} token"
    if not metrics['ended_with_eos']:
        status += ' · 已达到生成长度上限，回答可能未完成'
    return '', messages, messages, status


def clear_conversation():
    return '', [], [], '会话已清空，可以开始新话题。'


def switch_model(model_mode):
    if model_mode not in ('base', 'lora'):
        raise ValueError('模型选择无效')
    name = 'LoRA 微调版' if model_mode == 'lora' else 'Base Qwen3'
    return '', [], [], f'当前模型：{name}。旧会话已清空，请重新输入问题。'
