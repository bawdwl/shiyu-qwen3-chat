import json
import tempfile
import unittest
from pathlib import Path

from training_data import read_records


class TrainingRecordIOTests(unittest.TestCase):
    def test_unicode_separators_inside_chat_are_not_record_boundaries(self):
        rows = [{'text': '第一段\u2028第二段\u2029第三段\u0085第四段'}, {'text': '普通文本'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
            self.assertEqual(read_records(path), rows)

    def test_blank_records_and_windows_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            path.write_bytes(b'{"id":1}\r\n\r\n  \r\n{"id":2}\r\n')
            self.assertEqual(read_records(path), [{'id': 1}, {'id': 2}])


if __name__ == '__main__':
    unittest.main()
