"""真实HTTP/GPU验收请求级长度和历史标题，只清理自己创建的消息。"""
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx


def main():
    records = []
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=260) as client:
        for limit, prompt in [(512, '请详细写一份给初学者的Python入门路线，分七天，每天包含目标、练习和常见错误。不要只列标题。'),
                              (1024, '请只回复：你好')]:
            response = client.post('/sessions', json={'model_mode': 'base'})
            response.raise_for_status()
            sid = response.json()['id']
            with client.stream('POST', '/chat/stream', json=dict(session_id=sid, message=prompt,
                               max_new_tokens=limit, timeout_seconds=240)) as stream:
                stream.raise_for_status()
                event, chunks, done = None, [], None
                for line in stream.iter_lines():
                    if line.startswith('event: '):
                        event = line[7:]
                    elif line.startswith('data: '):
                        data = json.loads(line[6:])
                        if event == 'delta':
                            chunks.append(data['text'])
                        elif event == 'done':
                            done = data
                        elif event == 'error':
                            raise RuntimeError(data)
            assert done is not None
            assert done['metrics']['max_new_tokens'] == limit
            assert done['metrics']['output_tokens'] <= limit
            assert ''.join(chunks).strip() == done['answer']
            saved = client.get('/sessions/' + sid)
            saved.raise_for_status()
            assert len(saved.json()['messages']) == 2
            listed = client.get('/sessions')
            listed.raise_for_status()
            title = next(row['title'] for row in listed.json()['sessions'] if row['id'] == sid)
            assert title == ' '.join(prompt[:160].split())[:40]
            records.append(dict(request_limit=limit, prompt=prompt, title=title,
                                answer=done['answer'], metrics=done['metrics'], persisted=True))
            client.delete('/sessions/' + sid + '/messages').raise_for_status()
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), records=records,
                  note='Real HTTP/GPU; tests Base mode and selected limits, not guaranteed semantic quality or maximum 1024-token generation. Only own test messages cleared.')
    Path('docs/chat-controls-smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
