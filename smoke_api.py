"""连接真实本地服务，记录GPU聊天和SQLite存取结果。"""
import json
from datetime import datetime
from pathlib import Path

import httpx


def main():
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=120) as client:
        health=client.get('/health');health.raise_for_status()
        traces=[]
        for mode in health.json()['model_modes']:
            response=client.post('/sessions',json={'model_mode':mode});response.raise_for_status()
            sid=response.json()['id'];answers=[]
            for message in ['请记住本次虚构文件代号：map_73。只回复收到。','刚才的文件代号是什么？只输出代号。']:
                r=client.post('/chat',json={'session_id':sid,'message':message});r.raise_for_status()
                result=r.json();assert result['answer'].strip() and result['model_mode']==mode
                answers.append(result)
            saved=client.get('/sessions/'+sid);saved.raise_for_status()
            assert len(saved.json()['messages'])==4
            # 模型质量观察独立于会话正确传递；不假装保证模型必能记住代号。
            remembered='map_73' in answers[-1]['answer']
            invalid=client.post('/chat',json={'session_id':sid,'message':'  '})
            assert invalid.status_code==422
            cleared=client.delete('/sessions/'+sid+'/messages');cleared.raise_for_status()
            assert cleared.json()['messages']==[]
            traces.append(dict(session_id=sid,model_mode=mode,answers=answers,
                               saved_message_count=4,remembered=remembered,
                               empty_input_status=422,cleared=True))
        missing=client.get('/sessions/00000000-0000-0000-0000-000000000000')
        assert missing.status_code==404
        result=dict(timestamp=datetime.now().astimezone().isoformat(),health=health.json(),
                    traces=traces,missing_session_status=404,
                    note='真实HTTP和GPU推理；会话清空仅涉及本脚本刚创建的合成测试记录。')
        Path('docs/api-smoke-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
