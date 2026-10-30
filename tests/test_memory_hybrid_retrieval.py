import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from structured_memory import CandidateStore, build_unified_memory_context


ROOT = Path(__file__).resolve().parents[1]
MEMORY_TEST_ROOT = ROOT / 'outputs' / 'memory-research'


class KeywordEmbedder:
    model_name = 'test-keyword-embedder'

    def encode(self, texts):
        vectors = []
        for raw in texts:
            text = str(raw or '')
            vectors.append([
                1.0 if any(word in text for word in ('城市', '杭州', '上海', '工作')) else 0.0,
                1.0 if any(word in text for word in ('咖啡', '饮料', '书店')) else 0.0,
                1.0 if any(word in text for word in ('面试', '周五', '周三')) else 0.0,
            ])
        for vector in vectors:
            length = sum(value * value for value in vector) ** 0.5 or 1.0
            for index in range(len(vector)):
                vector[index] /= length
        return vectors


def candidate(message_id, content, *, value=None, key='user.city'):
    return {
        'status': 'candidate',
        'kind': 'user_report',
        'subject': 'user',
        'fact_key': key,
        'value': value,
        'content': content,
        'evidence': [{
            'message_id': message_id,
            'role': 'user',
            'quote': content,
        }],
    }


class HybridMemoryRetrievalTests(unittest.TestCase):
    def setUp(self):
        MEMORY_TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=MEMORY_TEST_ROOT)
        self.store = CandidateStore(
            Path(self.temp.name) / 'memory.sqlite3',
            embedder=KeywordEmbedder(),
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_fts5_is_enabled_and_indexes_existing_rows(self):
        self.assertTrue(self.store.fts_enabled)
        self.store.auto_ingest(
            'xiaoman',
            candidate(1, '我目前住在杭州生活。', value='杭州', key='user.current.city'),
        )
        rows = self.store.db.execute(
            '''SELECT c.id FROM candidates_fts
               JOIN candidates c ON c.id=candidates_fts.rowid
               WHERE candidates_fts MATCH ?''',
            ('"杭州生"',),
        ).fetchall()
        self.assertEqual(len(rows), 1)

    def test_semantic_aliasing_can_retrieve_without_literal_overlap(self):
        target = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '我已经搬去杭州生活。', value='杭州', key='user.current.city'),
        )['item']
        self.store.auto_ingest(
            'xiaoman',
            candidate(2, '我平时最喜欢喝咖啡。', value='咖啡', key='user.preference.drink'),
        )
        ranked = self.store.search_hybrid('xiaoman', '现在工作在哪个城市？')
        self.assertEqual(ranked[0]['id'], target['id'])
        self.assertGreater(ranked[0]['retrieval']['semantic'], 0)

    def test_unified_context_ranks_fact_before_raw_excerpt_and_deduplicates(self):
        fact = self.store.auto_ingest(
            'xiaoman',
            candidate(4, '我的面试已经改到周五下午。', value='周五下午', key='user.interview.time'),
        )['item']
        text, trace = build_unified_memory_context(
            self.store,
            'xiaoman',
            '我的面试现在是什么时候？',
            excerpts=[
                {'message_id': 9, 'role': 'user', 'quote': '我记得之前聊过面试。'},
                {'message_id': 4, 'role': 'user', 'quote': '我的面试已经改到周五下午。'},
            ],
        )
        self.assertIn('当前用户事实', text)
        self.assertEqual(trace['selected_ids'][0], fact['id'])
        self.assertNotIn(4, trace['selected_excerpt_ids'])

    def test_current_and_historical_values_share_one_context_budget(self):
        old = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '我的面试原本安排在周三。', value='周三', key='user.interview.time'),
        )['item']
        new = self.store.auto_ingest(
            'xiaoman',
            candidate(3, '不是周三了，面试改到周五。', value='周五', key='user.interview.day'),
        )['item']
        self.assertEqual(new['supersedes'], old['id'])
        self.assertIsNotNone(self.store.get('xiaoman', old['id'])['valid_to'])
        text, trace = build_unified_memory_context(
            self.store,
            'xiaoman',
            '我的面试原来是什么时候，现在是什么时候？',
        )
        self.assertIn('当前用户事实', text)
        self.assertIn('历史旧值（已被替代）', text)
        self.assertIn(new['id'], trace['selected_ids'])
        self.assertIn(old['id'], trace['selected_ids'])


if __name__ == '__main__':
    unittest.main()
