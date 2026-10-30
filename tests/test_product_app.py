import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from backend import create_app
from storage import ChatStore

class Engine:
    adapter_available=True
    def __init__(self): self.calls=[]
    def reply(self,messages,**kwargs):
        self.calls.append(messages)
        return '模拟回答，仅用于接口测试',dict(seconds=.01,output_tokens=2)
    def reply_stream(self,messages,*args,**kwargs): return self.reply(messages)

class ProductAppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.engine=Engine()
        self.client=TestClient(create_app(Path(self.temp.name)/'app.db',engine_factory=lambda:self.engine))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()
    def test_app_assets_and_catalogue(self):
        page=self.client.get('/ui/').text
        self.assertIn('拾语',page)
        import re
        assets=re.findall(r'(?:src|href)="([^"]+)"',page)
        self.assertTrue(any(asset.endswith('.js') for asset in assets))
        self.assertTrue(any(asset.endswith('.css') for asset in assets))
        for asset in assets:
            if asset.startswith('/ui/assets/'):
                self.assertEqual(self.client.get(asset).status_code,200)
        self.assertEqual(self.client.get('/ui/icon.svg').status_code,200)
        self.assertEqual(self.client.get('/ui/media/portraits/xiaoman.png').status_code,200)
        self.assertEqual(self.client.get('/ui/assets/../backend.py').status_code,404)
        self.assertEqual([role['id'] for role in self.client.get('/characters').json()['characters']],
                         ['interview','companion'])

    def test_hidden_legacy_roles_remain_available_to_existing_sessions(self):
        role_ids=[role['id'] for role in self.client.get('/characters').json()['characters']]
        self.assertNotIn('study',role_ids)
        self.assertNotIn('assistant',role_ids)
        # Old conversation bindings remain in the internal registry for continuity.
        sid=self.client.post('/sessions',json={'character_id':'study'}).json()['id']
        response=self.client.post('/chat',json={'session_id':sid,'message':'继续聊聊学习计划'})
        self.assertEqual(response.status_code,200)
    def test_character_prompt_bound_to_session_and_not_saved_as_message(self):
        sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
        self.assertEqual(self.client.post('/chat',json=dict(session_id=sid,message='你好')).status_code,200)
        self.assertEqual(self.engine.calls[-1][0]['role'],'system')
        self.assertIn('小满',self.engine.calls[-1][0]['content'])
        self.assertIn('直接输出角色这一次真正发出的完整聊天消息',self.engine.calls[-1][0]['content'])
        self.assertIn('不要添加“角色说出口”',self.engine.calls[-1][0]['content'])
        self.assertNotIn('<speech>',self.engine.calls[-1][0]['content'])
        self.assertNotIn('<inner_thought>',self.engine.calls[-1][0]['content'])
        saved=self.client.get('/sessions/'+sid).json()
        self.assertEqual(saved['character_id'],'companion');self.assertEqual(len(saved['messages']),2)
    def test_explicit_ai_identity_question_is_truthful_in_normal_and_stream_chat(self):
        expected='我是小满，一个AI聊天角色，不是真人。'
        sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
        response=self.client.post('/chat',json={'session_id':sid,'message':'你是真人还是AI？'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['answer'],expected)
        self.assertEqual(response.json()['metrics']['response_source'],'identity_policy')
        self.assertEqual(self.engine.calls,[])

        stream_sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
        streamed=self.client.post('/chat/stream',json={'session_id':stream_sid,'message':'你是AI吗？'})
        self.assertIn(expected,streamed.text)
        self.assertIn('event: done',streamed.text)
        self.assertEqual(self.engine.calls,[])

    def test_xiaoman_gender_is_fixed_and_answered_naturally(self):
        roles=self.client.get('/characters').json()['characters']
        xiaoman=next(role for role in roles if role['id']=='companion')
        self.assertEqual(xiaoman['gender'],'女性')
        self.assertIn('女性',xiaoman['profile']['tags'])
        self.assertEqual(xiaoman['identity']['age'],'24岁')
        self.assertEqual(xiaoman['identity']['city'],'厦门')
        self.assertEqual(xiaoman['identity']['occupation'],'独立书店店员')
        for question in ('你是爷们吗？','你是女生吗？','你的性别是什么？','你是什么性别？','你是个男生吗？'):
            sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
            response=self.client.post('/chat',json={'session_id':sid,'message':question})
            self.assertEqual(response.json()['answer'],'我是女生呀，怎么啦？')
            self.assertEqual(response.json()['metrics']['response_source'],'persona_policy')
        self.assertEqual(self.engine.calls,[])
    def test_xiaoman_gender_policy_also_streams_and_is_role_scoped(self):
        sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
        streamed=self.client.post('/chat/stream',json={'session_id':sid,'message':'你是男生吗？'})
        self.assertIn('我是女生呀，怎么啦？',streamed.text)
        self.assertIn('event: done',streamed.text)
        self.assertEqual(self.engine.calls,[])
        other=self.client.post('/sessions',json={'character_id':'study'}).json()['id']
        response=self.client.post('/chat',json={'session_id':other,'message':'你是男生吗？'})
        self.assertEqual(response.json()['answer'],'模拟回答，仅用于接口测试')
        self.assertEqual(len(self.engine.calls),1)
    def test_fixed_identity_facts_use_canonical_profile_and_unspecified_details_use_model(self):
        expected={
            '你今年多大？':'我今年24岁。',
            '你在厦门做什么工作？':'我在厦门的一家独立书店做店员。',
            '你在哪？':'我在厦门，在一家独立书店工作。',
            '你平时喜欢做什么？':'我喜欢咖啡、散步、猫，平时也会找点时间慢慢享受这些小事。',
            '你平常有哪些爱好？':'我喜欢咖啡、散步、猫，平时也会找点时间慢慢享受这些小事。',
            '你的兴趣爱好是什么？':'我喜欢咖啡、散步、猫，平时也会找点时间慢慢享受这些小事。',
            '你是什么性格？':'我性格温柔、直率，愿意认真听你说话。',
            '你叫什么名字？':'我叫小满，今年24岁，在厦门的一家独立书店工作。',
            '你叫什么？':'我叫小满，今年24岁，在厦门的一家独立书店工作。',
            '你是做什么工作的？':'我在厦门的一家独立书店做店员。',
            '你在哪上班？':'我在厦门的一家独立书店做店员。',
            '你住哪？':'我在厦门，在一家独立书店工作。',
        }
        for question,answer in expected.items():
            sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
            response=self.client.post('/chat',json={'session_id':sid,'message':question})
            self.assertEqual(response.json()['answer'],answer,question)
            self.assertEqual(response.json()['metrics']['response_source'],'persona_policy',question)
        self.assertEqual(self.engine.calls,[])
        for question in ('你生日是哪天？','你家乡在哪？','你是哪儿人？'):
            sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
            response=self.client.post('/chat',json={'session_id':sid,'message':question})
            self.assertEqual(response.json()['answer'],'模拟回答，仅用于接口测试',question)
            self.assertEqual(len(self.engine.calls),1,question)
            prompt=self.engine.calls[-1][0]['content']
            self.assertIn('档案没有写明的普通生活经历、习惯和偏好可以自然补充',prompt)
            self.assertNotIn('这个细节还没写进我的角色设定',prompt)
            self.engine.calls.clear()
    def test_profile_fact_reply_streams_and_other_roles_still_use_model(self):
        sid=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
        streamed=self.client.post('/chat/stream',json={'session_id':sid,'message':'你在哪工作？'})
        self.assertIn('我在厦门的一家独立书店做店员。',streamed.text)
        self.assertIn('event: done',streamed.text)
        self.assertEqual(self.engine.calls,[])
        open_detail=self.client.post('/sessions',json={'character_id':'companion'}).json()['id']
        streamed_detail=self.client.post('/chat/stream',json={'session_id':open_detail,'message':'你家乡在哪？'})
        self.assertIn('模拟回答，仅用于接口测试',streamed_detail.text)
        self.assertIn('event: done',streamed_detail.text)
        self.assertIn('档案没有写明的普通生活经历、习惯和偏好可以自然补充',
                      self.engine.calls[-1][0]['content'])
        self.engine.calls.clear()
        other=self.client.post('/sessions',json={'character_id':'study'}).json()['id']
        response=self.client.post('/chat',json={'session_id':other,'message':'你今年多大？'})
        self.assertEqual(response.json()['answer'],'模拟回答，仅用于接口测试')
        self.assertEqual(len(self.engine.calls),1)
    def test_stream_uses_same_character_and_new_session_defaults_to_yanchuan(self):
        sid=self.client.post('/sessions',json={'character_id':'study'}).json()['id']
        response=self.client.post('/chat/stream',json=dict(session_id=sid,message='列表是什么'))
        self.assertIn('event: done',response.text);self.assertIn('知夏',self.engine.calls[-1][0]['content'])
        other=self.client.post('/sessions',json={}).json()['id']
        self.client.post('/chat',json=dict(session_id=other,message='普通问题'))
        self.assertIn('言川',self.engine.calls[-1][0]['content'])
    def test_unknown_role_and_client_system_prompt_rejected(self):
        self.assertEqual(self.client.post('/sessions',json={'character_id':'unknown'}).status_code,404)
        self.assertEqual(self.client.post('/sessions',json={'system_prompt':'伪造设定'}).status_code,422)

    def test_local_character_card_generation_review_save_and_worldbook_injection(self):
        import json
        generation_calls=[]
        def fake_generation_reply(messages,**kwargs):
            generation_calls.append(messages)
            self.engine.calls.append(messages)
            return json.dumps({
                'name':'阿澄','tagline':'海边电台的夜班主持人','description':'在深夜陪人听歌聊天。',
                'background':'她在海边城市主持一档深夜节目。','personality':'安静、细心。',
                'speech_style':'语气自然，不急着追问。','relationship':'从听众与主持人的友好关系开始。',
                'greeting':'今晚的海风挺舒服，你想点一首什么歌？','boundaries':'不声称认识现实中的用户。',
                'fixed_facts':[{'key':'节目','value':'海边电台午夜档','status':'confirmed'}],
                'worldbook':[{'title':'午夜节目','keywords':['电台','节目'],'content':'节目每晚午夜播出。','enabled':True}],
                'example_dialogue':[{'user':'今天睡不着。','character':'那我陪你听一会儿，今晚的歌慢一些。'}],
            },ensure_ascii=False),dict(seconds=.1,output_tokens=100)
        self.engine.reply=fake_generation_reply
        generated=self.client.post('/characters/generate',json={'inspiration':'夜班电台主持人，喜欢听众点歌。'})
        self.assertEqual(generated.status_code,200,generated.text)
        card=generated.json()['card']
        self.assertEqual(card['source_notes'],'夜班电台主持人，喜欢听众点歌。')
        self.assertEqual(card['fixed_facts'][0]['status'],'proposal')
        self.assertIn('不要擅自补玩家姓名',generation_calls[-1][0]['content'])
        saved=self.client.post('/characters',json={'card':card})
        self.assertEqual(saved.status_code,201,saved.text)
        character_id=card['id']
        self.assertIn(character_id,[role['id'] for role in self.client.get('/characters').json()['characters']])
        sid=self.client.post('/sessions',json={'character_id':character_id}).json()['id']
        self.client.post('/chat',json={'session_id':sid,'message':'说说你们电台的节目吧。'})
        prompt=self.engine.calls[-1][0]['content']
        self.assertIn('作者保存的角色背景',prompt)
        self.assertIn('本轮命中的世界书条目',prompt)
        self.assertIn('午夜节目',prompt)
        self.assertEqual(self.client.get('/characters/'+character_id).json()['card']['name'],'阿澄')
        self.assertEqual(self.client.delete('/characters/'+character_id).status_code,200)
        self.assertEqual(self.client.post('/sessions',json={'character_id':character_id}).status_code,404)
        self.assertEqual(self.client.post('/chat',json={'session_id':sid,'message':'继续聊电台'}).status_code,410)
        self.assertEqual(len(self.client.get('/sessions/'+sid).json()['messages']),2)
    def test_role_returned_in_history_list(self):
        sid=self.client.post('/sessions',json={'character_id':'interview'}).json()['id']
        rows=self.client.get('/sessions').json()['sessions']
        self.assertEqual(next(r for r in rows if r['id']==sid)['character_id'],'interview')
    def test_displayed_profile_is_in_server_prompt_for_each_character(self):
        roles=self.client.get('/characters').json()['characters']
        for role in roles:
            if role['id']=='assistant': continue
            sid=self.client.post('/sessions',json={'character_id':role['id']}).json()['id']
            self.client.post('/chat',json={'session_id':sid,'message':'介绍一下你的角色设定'})
            prompt=self.engine.calls[-1][0]['content']
            self.assertIn(role['profile']['about'],prompt)
            self.assertIn(role['profile']['scene'],prompt)
            self.assertNotIn('prompt',role)
    def test_old_database_migration_preserves_messages(self):
        path=Path(self.temp.name)/'old.db'
        with sqlite3.connect(path) as db:
            db.executescript("CREATE TABLE sessions(id TEXT PRIMARY KEY,model_mode TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL);INSERT INTO sessions VALUES('old','base','t','t');INSERT INTO messages(session_id,role,content,created_at) VALUES('old','user','原有消息','t');")
        db.close()
        migrated=ChatStore(path).get('old')
        self.assertEqual(migrated['character_id'],'assistant')
        self.assertEqual(migrated['messages'][0]['content'],'原有消息')

    def test_character_delete_rejects_builtin_and_missing_roles(self):
        built_in=self.client.delete('/characters/companion')
        missing=self.client.delete('/characters/custom-does-not-exist')
        self.assertEqual(built_in.status_code,409)
        self.assertEqual(missing.status_code,404)

    def test_prompt_comparison_registry_does_not_change_live_characters(self):
        import copy
        from characters import CHARACTERS, conversation_messages
        original=copy.deepcopy(CHARACTERS)
        candidate=copy.deepcopy(CHARACTERS)
        candidate['companion']['prompt']='离线候选设定'
        session=dict(character_id='companion',messages=[dict(role='user',content='之前的问题')])
        result=conversation_messages(session,'介绍一下你的角色设定',candidate)
        self.assertTrue(result[0]['content'].startswith('离线候选设定'))
        self.assertEqual(result[-2:],session['messages']+[dict(role='user',content='介绍一下你的角色设定')])
        self.assertEqual(CHARACTERS,original)
        self.assertEqual(conversation_messages(session,'介绍一下你的角色设定')[0]['content'],
                         original['companion']['prompt']+result[0]['content'][len('离线候选设定'):])

    def test_companion_prompt_allows_fictional_details_and_requires_same_chat_consistency(self):
        from characters import CHARACTERS, conversation_messages
        messages=conversation_messages(dict(character_id='companion',messages=[]),'我把报错修好了，好开心。')
        self.assertTrue(messages[0]['content'].startswith(CHARACTERS['companion']['prompt']))
        self.assertIn('年龄：24岁',messages[0]['content'])
        self.assertIn('所在地：厦门；职业：独立书店店员',messages[0]['content'])
        self.assertIn('不要只输出名词或字段值',messages[0]['content'])
        self.assertIn('档案没有写明的普通生活经历、习惯和偏好可以自然补充',messages[0]['content'])
        self.assertIn('后续要沿用；新会话从固定档案重新开始',messages[0]['content'])
        self.assertIn('不要先声明设定缺失',messages[0]['content'])
        self.assertIn('多数回复用自然的陈述句结束',messages[0]['content'])
        self.assertIn('不套用固定反问或固定收尾',messages[0]['content'])
        self.assertIn('【示例对话：仅示范语气与同一聊天内沿用细节，不是本次聊天已发生的事】',messages[0]['content'])
        self.assertIn('你刚才说下班后会做什么来着？',messages[0]['content'])
        self.assertIn('不要照搬其中的细节',messages[0]['content'])
        self.assertNotIn('这些细节未设定时，不要自行编造为事实',messages[0]['content'])
        self.assertNotIn(CHARACTERS['companion']['profile']['about'],messages[0]['content'])
        self.assertIn('小满',messages[0]['content'])

    def test_profile_is_available_only_for_current_persona_request(self):
        from characters import CHARACTERS, conversation_messages
        current=dict(character_id='companion',messages=[dict(role='user',content='你喜欢喝什么？'),
                                                       dict(role='assistant',content='角色设定里喜欢拿铁。')])
        self.assertIn(CHARACTERS['companion']['profile']['about'],
                      conversation_messages(current,'你的角色设定是什么？')[0]['content'])
        daily=conversation_messages(current,'今天我只想发会儿呆。')[0]['content']
        self.assertIn('年龄：24岁',daily)
        self.assertNotIn(CHARACTERS['companion']['profile']['about'],daily)

    def test_fixed_context_and_stream_path_use_latest_user_request(self):
        from characters import CHARACTERS, conversation_messages
        current=dict(character_id='companion',messages=[dict(role='user',content='今天想安静一会儿。')])
        self.assertIn('年龄：24岁',conversation_messages(current)[0]['content'])
        current['messages'][-1]['content']='你喜欢什么？'
        self.assertIn(CHARACTERS['companion']['profile']['about'],conversation_messages(current)[0]['content'])

    def test_users_own_preferences_do_not_inject_character_profile(self):
        from characters import CHARACTERS, conversation_messages
        current=dict(character_id='companion',messages=[])
        prompt=conversation_messages(current,'我还不知道自己喜欢什么工作。')[0]['content']
        self.assertIn('年龄：24岁',prompt)
        self.assertNotIn(CHARACTERS['companion']['profile']['about'],prompt)

    def test_legacy_offline_registry_keeps_historical_profile_policy(self):
        import copy
        from characters import CHARACTERS, conversation_messages
        legacy=copy.deepcopy(CHARACTERS)
        legacy['companion'].pop('profile_policy')
        messages=conversation_messages(dict(character_id='companion',messages=[]),'今天好累。',legacy)
        self.assertIn(legacy['companion']['profile']['about'],messages[0]['content'])

if __name__=='__main__': unittest.main()
