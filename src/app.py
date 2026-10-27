"""Gradio通过API聊天；只有backend.py加载模型。"""
import argparse
from pathlib import Path
from uuid import uuid4
import gradio as gr
from api_client import BackendClient
from generation_options import ui_timeout


def build_app(client):
    health=client.request('GET','/health')
    choices=[('Base Qwen3','base')]
    if 'lora' in health['model_modes']: choices.append(('LoRA 微调版','lora'))
    def history(session):
        return [dict(role=m['role'],content=m['content']) for m in session['messages']]
    def create(selected):
        return client.request('POST','/sessions',json={'model_mode':selected})['id']

    with gr.Blocks(title='Qwen3 中文聊天') as demo:
        gr.Markdown('# Qwen IM\n聊一聊，也可以用语音。对话保存在这台电脑上。')
        sid=gr.State(None)
        provider=health.get('speech_provider')
        with gr.Row():
            with gr.Column(scale=1,min_width=250):
                new=gr.Button('＋ 新对话',variant='primary')
                sessions=gr.Dropdown(label='历史对话',choices=[],allow_custom_value=False)
                refresh=gr.Button('刷新历史')
                with gr.Accordion('模型与回答设置',open=False):
                    mode=gr.Radio(choices,value='base',label='选择模型（开始新会话）')
                    gr.Markdown('LoRA是微调实验版，可能改变聊天语气，也可能影响原有能力；建议与Base对照体验。')
                    length=gr.Dropdown([('简短 · 最多256 token',256),('标准 · 最多512 token',512),
                                        ('详细 · 最多1024 token',1024)],value=512,label='回答长度')
                    gr.Markdown('token不等于字数。简短/标准最多等待120秒，详细240秒。')
                with gr.Accordion('语音输入',open=False):
                    gr.Markdown('录音或上传 → 识别到输入框 → 修改后发送。识别会替换当前草稿，不会自动发送。最长30秒、8MiB，上传推荐16位PCM WAV。')
                    recording=gr.Audio(label='语音问题',sources=['microphone','upload'],type='filepath',format=None)
                    recognize=gr.Button('识别到输入框（替换草稿）',interactive=bool(health.get('asr_provider')))
                    asr_status=gr.Markdown(health.get('asr_provider') or '语音输入未配置，仍可输入文字。')
                    gr.Markdown('若找不到麦克风，可用Chrome或Edge打开本页。识别可能漏字，请检查文字。')
                with gr.Accordion('当前会话管理',open=False):
                    clear=gr.Button('清空当前会话')
                    gr.Markdown('清空会删除本会话消息；想保留记录请使用“新对话”。')
            with gr.Column(scale=3,min_width=320):
                chatbot=gr.Chatbot(label='对话',height=460,buttons=[])
                status=gr.Markdown('已就绪。输入问题，或从下面选一个话题。')
                message=gr.Textbox(label='你的问题',lines=3,placeholder='今天想聊什么？')
                with gr.Row():
                    send=gr.Button('发送',variant='primary');stop=gr.Button('停止生成');retry=gr.Button('重试未完成的问题')
                gr.Examples(examples=[['今天忙了一天，脑子有点转不动了。'],
                                      ['刚刚路上看到一只胖乎乎的小猫，心情都好了一点。'],
                                      ['周末想找个地方走走，别太赶，有什么想法吗？']],inputs=message)
                with gr.Accordion('收听回答',open=False):
                    read=gr.Button('朗读最近一条回答',interactive=bool(provider))
                    audio=gr.Audio(label='回答语音（点击播放，可暂停）',type='filepath',autoplay=False)
                    voice_status=gr.Markdown('停止生成只停止文字，音频使用播放器暂停。')
                    gr.Markdown(f'朗读：{provider or "未配置"}')

        def read_latest(current_id):
            try:
                if not current_id: return None,'请先发送问题并等待回答完成。'
                current=client.request('GET',f'/sessions/{current_id}')
                answers=[m for m in current['messages'] if m['role']=='assistant']
                if not answers: return None,'当前没有已完成的回答可朗读。'
                content=client.speech(current_id,answers[-1]['id'])
                folder=Path('outputs/audio-preview');folder.mkdir(parents=True,exist_ok=True)
                target=folder/(str(uuid4())+'.wav');target.write_bytes(content)
                return str(target.resolve()),'语音已生成，点击播放器收听。'
            except Exception as exc:
                return None,f'朗读未完成，文字回答保留：{exc}'

        def recognize_audio(path):
            if not path: return gr.skip(),'请先录音或上传PCM WAV文件。'
            try:
                result=client.transcribe(path)
                return result['text'],'识别完成，请检查输入框后点击发送；尚未保存为聊天消息。'
            except Exception as exc: return gr.skip(),f'识别未完成，原草稿保留：{exc}'

        def reset_recording():
            return None,'新会话已清除上一段录音，请重新录音或上传。'

        def send_message(text,current_id,selected,limit,retrying=False):
            if not text or not text.strip():
                yield gr.skip(),gr.skip(),gr.skip(),'请输入问题后再发送。';return
            if len(text)>4096:
                yield gr.skip(),gr.skip(),gr.skip(),'输入超过4096字符，请缩短。';return
            try:
                if not current_id: current_id=create(selected)
                if retrying: client.wait_idle(seconds=15)
                previous=client.request('GET',f'/sessions/{current_id}')
                old=history(previous)
                if retrying and len(old)>=2 and old[-2]==dict(role='user',content=text.strip()) and old[-1]['role']=='assistant':
                    yield '',current_id,old,'这个问题已有保存的回答，已恢复显示，没有重复发送。';return
                client.prepare(current_id)
                yield '',current_id,old+[dict(role='user',content=text)],'正在生成，可点击停止。'
                answer=''
                for event,data in client.stream(current_id,text,timeout=ui_timeout(limit),max_new_tokens=limit):
                    if event=='delta':
                        answer+=data['text']
                        yield '',current_id,old+[dict(role='user',content=text),dict(role='assistant',content=answer)],'正在生成……'
                    elif event=='done':
                        saved=client.request('GET',f'/sessions/{current_id}');m=data['metrics']
                        note=f"{saved['model_mode']} · {m['seconds']:.2f}秒 · {m['output_tokens']}/{limit} token · 已保存"
                        omitted=m.get('context_omitted_turns',0)
                        if omitted:
                            note+=f'\n\n本次未把最早的{omitted}轮对话送给模型，完整记录仍保存在历史中。若要引用较早内容，请在问题里重述。'
                        if not m['ended_with_eos']: note+='\n\n**回答可能没有说完。** 可发送“请继续刚才的回答”，或在“模型与回答设置”中选择更长的回答。'
                        yield '',current_id,history(saved),note
                    elif event=='error':
                        saved=client.request('GET',f'/sessions/{current_id}')
                        yield text,current_id,history(saved),data['detail']+'\n\n问题已放回输入框，可修改后点击“重试未完成的问题”。'
            except Exception as exc:
                try:
                    saved=client.request('GET',f'/sessions/{current_id}') if current_id else None
                    display=history(saved) if saved else []
                    # 完成事件丢失时，以本轮之前的消息前缀确认是否已经成功保存。
                    if 'previous' in locals() and saved and len(saved['messages'])==len(previous['messages'])+2:
                        before=previous['messages'];after=saved['messages']
                        if after[:-2]==before and after[-2]['role']=='user' and after[-2]['content']==text.strip() and after[-1]['role']=='assistant':
                            yield '',current_id,display,'连接中断，但回答已保存，已恢复显示；无需重试。';return
                except Exception: display=gr.skip()
                yield text,current_id,display,f'请求未完成：{exc}\n\n问题已保留，可修改后点击“重试未完成的问题”；重试会先核对是否已有回答。'

        def request_stop(current_id):
            if not current_id: return '当前没有生成请求。'
            try:
                r=client.stop(current_id)
                return '已请求停止，等待当前解码步骤退出。' if r['stop_requested'] else '没有可停止请求，或该轮已完成。'
            except Exception as exc: return f'停止失败：{exc}'

        def clear_session(current_id):
            try:
                if current_id:
                    client.request('POST',f'/sessions/{current_id}/stop');client.wait_idle()
                    client.request('DELETE',f'/sessions/{current_id}/messages')
                return '',[],'当前会话已清空。',None,'当前没有可播放音频。'
            except Exception as exc: return gr.skip(),gr.skip(),f'清空失败：{exc}',gr.skip(),gr.skip()

        def change_mode(selected):
            try: return '',create(selected),[],f'已开始{selected}新会话；旧会话保留。',None,'请先完成文字回答，再点击朗读。'
            except Exception as exc: return gr.skip(),gr.skip(),gr.skip(),f'切换失败：{exc}',gr.skip(),gr.skip()

        def list_sessions(current_id):
            rows=client.request('GET','/sessions')['sessions']
            ids={r['id'] for r in rows}
            return gr.update(choices=[(f"{r.get('title','新对话')} · {r['model_mode']} · {r['id'][:8]}",r['id']) for r in rows],
                             value=current_id if current_id in ids else None)

        def restore_session(selected):
            if not selected: return gr.skip(),gr.skip(),gr.skip(),gr.skip(),gr.skip(),gr.skip()
            try:
                current=client.request('GET',f'/sessions/{selected}')
                return current['id'],current['model_mode'],history(current),'已恢复保存的会话。',None,'已恢复文字，点击朗读可重新生成语音。'
            except Exception as exc: return gr.skip(),gr.skip(),gr.skip(),str(exc),gr.skip(),gr.skip()

        def send_with_audio(*args):
            for update in send_message(*args):
                yield (*update,None,'可朗读当前会话已保存的回答；播放和暂停使用播放器。')

        def retry_message(*args):
            for update in send_message(*args,retrying=True):
                yield (*update,None,'可朗读当前会话已保存的回答；播放和暂停使用播放器。')

        outputs=[message,sid,chatbot,status,audio,voice_status]
        options=dict(concurrency_limit=1,concurrency_id='chat',api_name=False)
        send.click(send_with_audio,[message,sid,mode,length],outputs,**options).then(list_sessions,sid,sessions,**options)
        retry.click(retry_message,[message,sid,mode,length],outputs,**options).then(list_sessions,sid,sessions,**options)
        # 停止必须绕过生成队列。
        stop.click(request_stop,sid,status,queue=False,api_name=False)
        clear.click(clear_session,sid,[message,chatbot,status,audio,voice_status],**options).then(list_sessions,sid,sessions,**options)
        chatbot.clear(clear_session,sid,[message,chatbot,status,audio,voice_status],**options).then(list_sessions,sid,sessions,**options)
        # input只响应用户操作，恢复会话的赋值不会再创建新会话。
        mode.input(change_mode,mode,outputs,**options).then(list_sessions,sid,sessions,**options)
        new.click(change_mode,mode,outputs,**options).then(list_sessions,sid,sessions,**options)
        refresh.click(list_sessions,sid,sessions,**options)
        sessions.input(restore_session,sessions,[sid,mode,chatbot,status,audio,voice_status],**options).then(lambda:'',None,message,**options)
        read.click(read_latest,sid,[audio,voice_status],**options)
        recognize.click(recognize_audio,recording,[message,asr_status],**options)
        for control in (new,clear):
            control.click(reset_recording,None,[recording,asr_status],**options)
        mode.input(reset_recording,None,[recording,asr_status],**options)
        sessions.input(reset_recording,None,[recording,asr_status],**options)
        demo.load(list_sessions,sid,sessions,**options)
    return demo.queue(max_size=8)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--api-url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    build_app(BackendClient(args.api_url)).launch(server_name='127.0.0.1',server_port=7860,share=False,inbrowser=False)
