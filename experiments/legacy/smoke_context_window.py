"""使用自建测试会话验证超长历史真实GPU生成；不改用户已有会话。"""
import json
import argparse
from pathlib import Path
from api_client import BackendClient
from storage import ChatStore

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='docs/context-window-smoke.json')
    args=parser.parse_args()
    client=BackendClient();client.wait_idle();store=ChatStore('outputs/chat.sqlite3');records=[]
    model_id=client.request('GET','/health')['model_id']
    for streaming in (False,True):
        sid=client.request('POST','/sessions',json={})['id']
        try:
            for i in range(8):
                store.save_turn(sid,f'历史轮次{i}：'+('这是用于验证上下文长度的公开合成内容。'*25),
                                '这是一条合成历史回答。'*30)
            before=store.get(sid)['messages']
            question='请用一句话告诉我，学习Python应该先掌握什么？'
            if streaming:
                done=None
                for event,data in client.stream(sid,question,max_new_tokens=32):
                    if event=='error': raise RuntimeError(data)
                    if event=='done': done=data
                assert done is not None;m=done['metrics']
            else:
                result=client.request('POST','/chat',json=dict(session_id=sid,message=question,max_new_tokens=32))
                m=result['metrics']
            after=store.get(sid)['messages']
            assert m['original_input_tokens']>2048 and m['input_tokens']<=2048
            assert m['context_omitted_turns']>0
            assert len(after)==len(before)+2 and after[:-2]==before
            assert after[-1]['content'].strip()
            records.append(dict(model_id=model_id,streaming=streaming,metrics=m,history_rows_before=len(before),
                                history_rows_after=len(after),original_history_unchanged=True,answer=after[-1]['content']))
        finally:
            client.request('DELETE',f'/sessions/{sid}/messages')
    Path(args.output).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
