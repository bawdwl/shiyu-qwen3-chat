import json
import tempfile
import unittest
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
from backend import create_app
from character_cards import normalize_card, card_to_runtime
from characters import CHARACTERS
from prompt_pipeline import assemble_prompt, reduce_repeated_history
from session_memory import refresh_memory, select_memory, retrieve_covered_history
from storage import ChatStore
from stream_job import GenerationStopped


class SummaryEngine:
    def reply_stream(self, messages, mode, stop, emit, **kwargs):
        rows = json.loads(messages[-1]['content'])
        selected = [dict(message_id=row['id'], quote=row['content'][:60]) for row in rows][-8:]
        return json.dumps(dict(items=selected), ensure_ascii=False), {}


class SessionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ChatStore(Path(self.temp.name) / 'chat.db')
        self.sid = self.store.create('base', 'interview')['id']

    def tearDown(self):
        self.temp.cleanup()

    def populate(self):
        for n in range(10):
            self.store.save_turn(self.sid, f'我把时间改为周日{n}点。', f'我带的杯子是蓝色的，第{n}次提到。')

    def test_summary_uses_older_complete_turns_and_keeps_speaker_sources(self):
        self.populate()
        state = refresh_memory(self.store, self.store.get(self.sid), SummaryEngine())
        self.assertEqual(state['covered_message_id'], 12)
        original = {m['id']: m for m in self.store.get(self.sid)['messages']}
        for item in state['items']:
            self.assertIn(item['quote'], original[item['message_id']]['content'])
            self.assertEqual(item['role'], original[item['message_id']]['role'])
        fresh = self.store.create('base', 'interview')['id']
        self.assertEqual(self.store.get_memory(fresh)['items'], [])

    def test_invented_quote_or_wrong_source_is_rejected(self):
        class BadEngine:
            def reply_stream(self, *args, **kwargs):
                return '{"items":[{"message_id":1,"quote":"用户住在上海"}]}', {}
        with self.assertRaises(ValueError):
            select_memory({}, [dict(id=1, role='user', content='我今天喝了咖啡。'),
                               dict(id=2, role='assistant', content='我喜欢手冲。')], BadEngine())
        self.assertEqual(self.store.get_memory(self.sid)['revision'], 0)

    def test_notes_survive_auto_updates_pause_restore_and_clear(self):
        self.populate()
        initial = self.store.get_memory(self.sid)
        saved = self.store.save_memory(self.sid, dict(notes='约定最后改为周日下午。', auto_enabled=False), initial['revision'])
        self.assertEqual(refresh_memory(self.store, self.store.get(self.sid), SummaryEngine()), saved)
        updated = refresh_memory(self.store, self.store.get(self.sid), SummaryEngine(), force=True)
        self.assertEqual(updated['notes'], saved['notes'])
        restored = self.store.restore_memory(self.sid)
        self.assertEqual(restored['items'] if 'items' in restored else [], [])
        self.store.clear(self.sid)
        self.assertEqual(self.store.get_memory(self.sid)['revision'], 0)

    def test_optimistic_write_and_deletion(self):
        self.store.save_memory(self.sid, dict(notes='一'), 0)
        with self.assertRaises(ValueError):
            self.store.save_memory(self.sid, dict(notes='二'), 0)
        self.store.delete_session(self.sid)
        with self.assertRaises(KeyError):
            self.store.get_memory(self.sid)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM session_memories').fetchone()[0], 0)

    def test_stop_during_summary_does_not_save_memory(self):
        self.populate()
        stop = Event()
        class StopEngine(SummaryEngine):
            def reply_stream(self, *args, **kwargs):
                result = super().reply_stream(*args, **kwargs)
                stop.set()
                return result
        with self.assertRaises(GenerationStopped):
            refresh_memory(self.store, self.store.get(self.sid), StopEngine(), stop)
        self.assertEqual(self.store.get_memory(self.sid)['revision'], 0)

    def test_recent_worldbook_examples_and_character_preset(self):
        card = normalize_card(dict(name='林音', example_dialogue=[dict(user='你好', character='我刚从电台回来，风挺大。')],
            generation_preset=dict(temperature=0.5),
            worldbook=[dict(title='海边电台', keywords=['电台'], content='靠海的夜间节目。')]))
        registry = {**CHARACTERS, card['id']: card_to_runtime(card)}
        history = [dict(id=1, role='user', content='聊聊电台吧'), dict(id=2, role='assistant', content='今晚播轻音乐。')]
        messages, trace = assemble_prompt(dict(character_id=card['id'], messages=history), '那个节目呢？', registry)
        self.assertNotIn('我刚从电台回来，风挺大。', messages[0]['content'])
        self.assertEqual(len(trace['worldbook_ids']), 1)
        self.assertEqual(trace['generation_preset']['temperature'], 0.5)
        self.assertEqual(messages[-1]['content'], '那个节目呢？')

    def test_opening_scene_and_opt_in_examples_do_not_repeat_on_later_turns(self):
        card=normalize_card(dict(name='林浩',scenario='买完午饭准备回宿舍。',use_examples=True,
             example_dialogue=[dict(user='最近玩什么',character='我最近在玩原神。')]))
        registry={**CHARACTERS,card['id']:card_to_runtime(card)}
        first,_=assemble_prompt(dict(character_id=card['id'],messages=[]),'你好',registry)
        self.assertIn('买完午饭准备回宿舍',first[0]['content'])
        self.assertIn('我最近在玩原神',first[0]['content'])
        history=[dict(id=1,role='user',content='你好'),dict(id=2,role='assistant',content='你好。')]
        later,_=assemble_prompt(dict(character_id=card['id'],messages=history),'一起打游戏',registry)
        self.assertNotIn('买完午饭准备回宿舍',later[0]['content'])
        self.assertNotIn('我最近在玩原神',later[0]['content'])
        self.assertEqual(later[-1]['content'],'一起打游戏')

    def test_repeated_assistant_sentences_removed_only_from_input(self):
        messages=[dict(role='system',content='角色设定'),dict(role='user',content='原神，启动'),
            dict(role='assistant',content='哎呀！我刚买完午饭准备回宿舍。你今天过得怎么样？要不要一起打游戏？'),
            dict(role='user',content='一起打原神'),
            dict(role='assistant',content='好呀！我刚买完午饭准备回宿舍。你今天过得怎么样？要不要一起打游戏？'),
            dict(role='user',content='可以')]
        result,count=reduce_repeated_history(messages,'可以')
        self.assertGreaterEqual(count,4)
        self.assertNotIn('买完午饭',result[2]['content'])
        self.assertNotIn('买完午饭',result[4]['content'])
        self.assertIn('买完午饭',messages[2]['content'])
        self.assertEqual(result[-1]['content'],'可以')
        self.assertEqual(reduce_repeated_history(messages,'你刚才说了什么？')[1],0)

    def test_single_repeated_fact_is_not_removed(self):
        messages=[dict(role='system',content='角色'),dict(role='user',content='是什么颜色？'),
            dict(role='assistant',content='我的杯子是蓝色的。'),dict(role='user',content='还是蓝色吗？'),
            dict(role='assistant',content='我的杯子是蓝色的。'),dict(role='user',content='谢谢')]
        self.assertEqual(reduce_repeated_history(messages,'谢谢')[1],0)

    def test_memory_boundary_retains_newer_correction_and_latest_question(self):
        current = dict(character_id='interview', messages=[dict(id=1,role='user',content='周六见'),
            dict(id=2,role='assistant',content='好的'), dict(id=3,role='user',content='改成周日'),
            dict(id=4,role='assistant',content='记住了')])
        messages, _ = assemble_prompt(current, '哪天见？', CHARACTERS,
            dict(covered_message_id=2, items=[dict(message_id=1,role='user',quote='周六见')]))
        self.assertIn('改成周日', [m['content'] for m in messages])
        self.assertEqual(messages[-1]['content'], '哪天见？')
        self.assertEqual(len(messages), 4)

    def test_topic_recalls_covered_source_not_retained_in_summary(self):
        history=[dict(id=1,role='user',content='你今天用什么杯子喝咖啡？'),
                 dict(id=2,role='assistant',content='我用了蓝色陶杯。')]
        found=retrieve_covered_history(history,2,'你之前用的杯子是什么颜色？')
        self.assertTrue(any(item['role']=='assistant' and '蓝色陶杯' in item['quote'] for item in found))
        current=dict(character_id='interview',messages=history)
        messages,_=assemble_prompt(current,'那个杯子是什么颜色？',CHARACTERS,dict(covered_message_id=2,items=[]))
        self.assertIn('蓝色陶杯',messages[0]['content'])
        self.assertIn('角色',messages[0]['content'])

    def test_unreadably_long_early_turn_does_not_advance_memory(self):
        self.store.save_turn(self.sid, '很长的故事' * 500, '我读完了。')
        self.populate()
        with self.assertRaisesRegex(ValueError, '过长'):
            refresh_memory(self.store,self.store.get(self.sid),SummaryEngine(),force=True)
        self.assertEqual(self.store.get_memory(self.sid)['revision'],0)

    def test_disjoint_quotes_cannot_be_joined_into_a_fabricated_source(self):
        class BadEngine:
            def reply_stream(self, *args, **kwargs):
                return '{"items":[{"message_id":1,"quote":"蓝色\\n陶杯"}]}', {}
        previous={'items':[dict(message_id=1,role='assistant',quote='蓝色'),
                           dict(message_id=1,role='assistant',quote='陶杯')]}
        with self.assertRaises(ValueError):
            select_memory(previous,[dict(id=3,role='user',content='你好'),
                                    dict(id=4,role='assistant',content='晚上好')],BadEngine())


class MemoryApiTests(unittest.TestCase):
    def test_edit_preview_restore_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as temp, TestClient(create_app(Path(temp) / 'db', engine_factory=SummaryEngine)) as client:
            sid = client.post('/sessions', json={}).json()['id']
            other = client.post('/sessions', json={}).json()['id']
            url = '/sessions/' + sid
            saved = client.put(url + '/memory', json=dict(notes='杯子是绿色的。',auto_enabled=False,revision=0))
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(client.put(url + '/memory', json=dict(notes='旧版本',revision=0)).status_code, 409)
            self.assertEqual(client.get('/sessions/' + other + '/memory').json()['notes'], '')
            preview = client.get(url + '/prompt', params=dict(message='杯子呢？')).json()
            self.assertIn('杯子是绿色的。', preview['messages'][0]['content'])
            self.assertEqual(preview['messages'][-1]['content'], '杯子呢？')
            self.assertEqual(client.put(url + '/memory', json=dict(notes='x',auto_enabled='yes',revision=1)).status_code, 422)

    def test_removed_quote_is_not_reintroduced_by_retrieval_and_restore_works(self):
        with tempfile.TemporaryDirectory() as temp, TestClient(create_app(Path(temp) / 'db', engine_factory=SummaryEngine)) as client:
            sid=client.post('/sessions',json={}).json()['id']
            store=client.app.state.store
            store.save_turn(sid,'你的杯子是什么颜色？','我用的是蓝色陶杯。')
            messages=store.get(sid)['messages']
            memory=store.save_memory(sid,dict(covered_message_id=messages[-1]['id'],items=[
                dict(message_id=messages[-1]['id'],role='assistant',quote='我用的是蓝色陶杯。')]),0)
            response=client.put(f'/sessions/{sid}/memory',json=dict(notes='',auto_enabled=True,
                                revision=memory['revision'],kept_items=[]))
            self.assertEqual(response.status_code,200)
            preview=client.get(f'/sessions/{sid}/prompt',params=dict(message='之前的杯子是什么颜色？')).json()
            self.assertNotIn('我用的是蓝色陶杯。',preview['messages'][0]['content'])
            self.assertEqual(client.post(f'/sessions/{sid}/memory/restore').status_code,200)
            preview=client.get(f'/sessions/{sid}/prompt',params=dict(message='之前的杯子是什么颜色？')).json()
            self.assertIn('我用的是蓝色陶杯。',preview['messages'][0]['content'])


if __name__ == '__main__':
    unittest.main()
