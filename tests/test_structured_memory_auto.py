import tempfile
import unittest
from pathlib import Path

from structured_memory import CandidateStore, MemoryWorker


ROOT = Path(__file__).resolve().parents[1]
MEMORY_TEST_ROOT = ROOT / 'outputs' / 'memory-research'


def candidate(message_id, content, value, fact_key='user.interview.time',
              *, operation='add', target_item_id=None):
    return {
        'status': 'candidate',
        'kind': 'user_report',
        'subject': 'user',
        'fact_key': fact_key,
        'value': value,
        'content': content,
        'event_date': None,
        'evidence': [{
            'message_id': message_id,
            'role': 'user',
            'quote': content,
        }],
        'operation': operation,
        'target_item_id': target_item_id,
    }


class AutomaticStructuredMemoryTests(unittest.TestCase):
    def setUp(self):
        MEMORY_TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=MEMORY_TEST_ROOT)
        self.store = CandidateStore(Path(self.temp.name) / 'memory.sqlite3')

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_new_evidence_is_active_without_review(self):
        result = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '我喜欢科幻小说', '科幻小说',
                      'user.preference.genre'),
        )
        self.assertEqual(result['action'], 'added')
        self.assertEqual(result['item']['state'], 'active')
        self.assertTrue(result['item']['semantic_verified'])

    def test_synonym_key_and_explicit_correction_supersede(self):
        first = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '面试在周三', '周三', 'user.interview.date'),
        )['item']
        second = self.store.auto_ingest(
            'xiaoman',
            candidate(3, '面试不是周三，改到周五', '周五',
                      'user.interview.current_day'),
        )
        self.assertEqual(second['action'], 'superseded')
        self.assertEqual(second['replaced_item_id'], first['id'])
        self.assertEqual(self.store.get('xiaoman', first['id'])['state'],
                         'superseded')
        self.assertEqual(second['item']['state'], 'active')
        self.assertEqual(second['item']['fact_key'], 'user.interview.time')
        current = self.store.search_active('xiaoman', '面试什么时候')
        self.assertEqual([item['value'] for item in current], ['周五'])

    def test_same_value_merges_sources(self):
        first = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '我在厦门工作', '厦门', 'user.work.city'),
        )['item']
        merged = self.store.auto_ingest(
            'xiaoman',
            candidate(2, '我现在确实在厦门工作', '厦门', 'user.work.location'),
        )
        self.assertEqual(merged['action'], 'merged')
        self.assertEqual(merged['item']['id'], first['id'])
        self.assertEqual(
            {item['message_id'] for item in merged['item']['evidence']},
            {1, 2},
        )

    def test_conflict_without_correction_remains_contested(self):
        first = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '面试在周三', '周三'),
        )['item']
        conflict = self.store.auto_ingest(
            'xiaoman',
            candidate(2, '面试在周五', '周五'),
        )
        self.assertEqual(conflict['action'], 'contested')
        self.assertEqual(conflict['item']['state'], 'contested')
        self.assertEqual(conflict['conflicts_with'], first['id'])
        self.assertEqual(self.store.get('xiaoman', first['id'])['state'],
                         'active')

    def test_deleted_evidence_is_tombstoned_but_later_statement_is_allowed(self):
        first = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '面试在周三', '周三'),
        )['item']
        self.store.delete('xiaoman', first['id'])
        blocked = self.store.auto_ingest(
            'xiaoman',
            candidate(1, '面试在周三', '周三'),
        )
        self.assertEqual(blocked['action'], 'ignored_deleted')
        later = self.store.auto_ingest(
            'xiaoman',
            candidate(3, '我重新确认，面试在周三', '周三'),
        )
        self.assertEqual(later['action'], 'added')
        self.assertEqual(later['item']['state'], 'active')

    def test_jobs_coalesce_by_character_and_claim_atomically(self):
        first = self.store.enqueue_auto_extraction(
            'xiaoman', 2, ready_in=0
        )
        second = self.store.enqueue_auto_extraction(
            'xiaoman', 4, ready_in=0
        )
        other = self.store.enqueue_auto_extraction(
            'yanchuan', 3, ready_in=0
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        claimed = self.store.claim_next_job()
        self.assertEqual(claimed['id'], first)
        self.assertEqual(claimed['message_end'], 4)
        self.assertEqual(claimed['status'], 'running')
        self.assertEqual(self.store.claim_next_job()['id'], other)
        self.assertIsNone(self.store.claim_next_job())

    def test_deferred_job_does_not_consume_retry_attempt(self):
        job_id = self.store.enqueue_auto_extraction(
            'xiaoman', 2, ready_in=0
        )
        claimed = self.store.claim_next_job()
        self.store.defer_job(job_id, 'waiting for API key', delay_seconds=0)
        row = self.store.db.execute(
            'SELECT status,attempts FROM memory_jobs WHERE id=?', (job_id,)
        ).fetchone()
        self.assertEqual(row['status'], 'queued')
        self.assertEqual(row['attempts'], claimed['attempts'])

    def test_worker_runs_one_job_without_user_approval(self):
        seen = []
        worker = MemoryWorker(
            self.store,
            lambda job: seen.append(job['id']) or {'action': 'ok'},
        )
        job_id = self.store.enqueue_auto_extraction(
            'xiaoman', 2, ready_in=0
        )
        self.assertTrue(worker.run_once())
        self.assertEqual(seen, [job_id])
        row = self.store.db.execute(
            'SELECT status FROM memory_jobs WHERE id=?', (job_id,)
        ).fetchone()
        self.assertEqual(row['status'], 'completed')

    def test_memory_usage_is_separate_and_idempotent_per_job(self):
        first = self.store.record_memory_usage(
            'xiaoman', 'opencode', 'deepseek-v4.1-flash',
            100, 20, usage_day='2026-10-30', job_id=41,
        )
        self.assertEqual(first['total_tokens'], 120)
        repeated = self.store.record_memory_usage(
            'xiaoman', 'opencode', 'deepseek-v4.1-flash',
            110, 25, usage_day='2026-10-30', job_id=41,
        )
        self.assertEqual(repeated['total_tokens'], 135)
        other = self.store.record_memory_usage(
            'yanchuan', 'opencode', 'deepseek-v4.1-flash',
            50, 10, usage_day='2026-10-30', job_id=42,
        )
        self.assertEqual(other['total_tokens'], 60)
        self.assertEqual(
            self.store.memory_usage_for_day('xiaoman', '2026-10-30')['total_tokens'],
            135,
        )


if __name__ == '__main__':
    unittest.main()
