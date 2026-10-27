"""检查冻结指纹、集合近重复、参考算术与部分技术事实，不运行模型。"""
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    root=Path('data/v2')
    manifest=json.loads((root/'freeze-manifest.json').read_text(encoding='utf-8'))
    test,validation=read(root/'test.jsonl'),read(root/'validation.jsonl')
    for key,name,rows in [('test','test.jsonl',test),('validation','validation.jsonl',validation)]:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest[key]['sha256']
        assert len(rows)==manifest[key]['count']
        assert len({r['id'] for r in rows})==len(rows)
    assert Counter(x['category'] for x in test)==dict.fromkeys(
        ['技术解释','指令与格式','多轮上下文','不确定性处理','能力回归'],10)
    from evaluate import grade
    for case in test:
        assert case['prompts'] and all(isinstance(p,str) and p.strip() for p in case['prompts'])
        assert case['content_rubric'] and case['review_required']
        if case['format_check']:
            expected=case['format_check']['expected']
            sample=json.dumps(expected,ensure_ascii=False) if case['format_check']['type']=='json_equal' else expected
            assert grade(sample,case['format_check'])
    arithmetic=[23+46,72-19,14*6,96//8,29%4,3*7+2,84//7,150//5,'低','B']
    assert [x['content_reference'] for x in test[-10:]]==arithmetic
    a=[];a.append([4,5]);b=[];b.extend([4,5])
    assert a==[[4,5]] and b==[4,5]
    a=[[1]];b=a.copy();b[0].append(2)
    assert a[0]==[1,2]
    events=[]
    try:
        raise ValueError('example')
    except ValueError:
        events.append('handled')
    finally:
        events.append('cleanup')
    assert events==['handled','cleanup']
    conn=sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys=ON')
    assert conn.execute('PRAGMA foreign_keys').fetchone()==(1,)
    conn.execute('CREATE TABLE parent(id INTEGER PRIMARY KEY)')
    conn.execute('CREATE TABLE child(pid INTEGER REFERENCES parent(id))')
    try:
        conn.execute('INSERT INTO child VALUES (99)')
        raise AssertionError('foreign key not enforced')
    except sqlite3.IntegrityError:
        conn.rollback()
    conn.execute('INSERT INTO parent VALUES(2)');conn.rollback()
    assert conn.execute('SELECT COUNT(*) FROM parent').fetchone()==(0,)
    assert conn.execute('SELECT NULL=NULL, NULL IS NULL').fetchone()==(None,1)
    conn.close()
    def norm(s):
        return re.sub(r'\W+','',s).lower()
    def grams(s):
        s=norm(s)
        return {s[i:i+3] for i in range(max(1,len(s)-2))}
    train=read('data/train.jsonl')
    sets={
        'historical_train':[(x['id'],x['messages'][0]['content']) for x in train],
        'new_validation':[(x['id'],x['messages'][0]['content']) for x in validation],
        'new_test':[(x['id'],p) for x in test for p in x['prompts']],
    }
    comparisons=[]
    for left,right in [('historical_train','new_validation'),('historical_train','new_test'),('new_validation','new_test')]:
        pairs=[]
        for lid,a in sets[left]:
            for rid,b in sets[right]:
                ga,gb=grams(a),grams(b)
                similarity=len(ga&gb)/len(ga|gb)
                assert norm(a)!=norm(b),f'Exact duplicate: {lid}/{rid}'
                pairs.append(dict(left=lid,right=rid,jaccard=similarity))
        pairs.sort(key=lambda x:x['jaccard'],reverse=True)
        comparisons.append(dict(splits=[left,right],top_pairs=pairs[:3],
                                flagged=[p for p in pairs if p['jaccard']>=0.35]))
    report=dict(status='data-checks-passed',counts=dict(training_retained=32,validation=20,test=50),
                categories=dict(Counter(x['category'] for x in test)),
                frozen_fingerprints_verified=True,arithmetic_references_verified=True,
                sample_facts_checked=['append/extend','shallow copy','except/finally','SQLite FK','rollback','NULL'],
                cross_split_prompt_checks=comparisons,
                limitations=['词面检查不能保证语义无泄漏','验证问答与旧训练涉及相近主题，非完全新知识',
                             '能力回归故意包含旧题相同能力形式的新数字；不是50项独立能力',
                             '未核验基础模型预训练污染','尚未生成新测试答案或训练新Adapter'])
    Path('docs/evaluation-v2-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
