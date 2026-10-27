"""自然问题原始对话记录；不是质量评分或公平速度基准。"""
import argparse
import json
from pathlib import Path
from api_client import BackendClient

CASES=[('companion','今天学了一下午Python，有点累，不想听大道理，陪我随便聊两句吧。'),
       ('companion','今天我终于独立修好了一个报错，想和你分享一下。'),
       ('study','Python里列表和元组有什么区别？请给一个简单例子。'),
       ('study','我总把学习计划排得太满，最后什么也没完成。今晚只有半小时，可以怎么开始？'),
       ('interview','我想练习大模型开发岗位的面试，一次只问我一个问题。'),
       ('assistant','不知道的问题你会怎么回答？请用两句话说明。')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    c=BackendClient();c.wait_idle();health=c.request('GET','/health');records=[]
    for role,question in CASES:
        sid=c.request('POST','/sessions',json=dict(character_id=role))['id']
        try:
            answer=None
            for event,data in c.stream(sid,question,timeout=120,max_new_tokens=128):
                if event=='error': raise RuntimeError(data)
                if event=='done': answer=data
            assert answer and answer['answer'].strip()
            records.append(dict(character_id=role,question=question,**answer))
            print('Recorded '+role,flush=True)
        finally: c.request('DELETE',f'/sessions/{sid}/messages')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(model_id=health['model_id'],max_new_tokens=128,
        note='Six natural examples, no blind scores; do not infer universal improvement or speed advantage.',
        records=records),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
