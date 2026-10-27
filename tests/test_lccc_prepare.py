import gzip
import json
import tempfile
import unittest
from pathlib import Path

from inspect_lccc import normalize_chinese_spacing
from prepare_lccc import acceptable, prepare
from training_data import audit_lccc


def write_gzip_jsonl(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


class LcccPreparationTests(unittest.TestCase):
    def test_normalizes_jieba_spaces_without_joining_english_words(self):
        text = '我 在 2026 年 喜欢 Chat GPT ， 你 呢 ？'
        self.assertEqual(normalize_chinese_spacing(text), '我在2026年喜欢Chat GPT，你呢？')

    def test_filters_common_contact_details(self):
        normalized, reason = acceptable(['你加我微信', '手机号 13800138000'])
        self.assertIsNone(normalized)
        self.assertEqual(reason, 'contact_pattern')

    def test_prepares_source_split_disjoint_frozen_pilot(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            dialogues = {
                'train': [['train甲问', 'train甲答'],
                          ['train乙问', 'train乙答', 'train乙再问'],
                          ['train丙问', 'train丙答', 'train丙再问', 'train丙再答']],
                'validation': [['valid甲问', 'valid甲答'],
                               ['valid乙问', 'valid乙答', 'valid乙再问'],
                               ['valid丙问', 'valid丙答', 'valid丙再问', 'valid丙再答']],
                'test': [['test甲问', 'test甲答'],
                         ['test乙问', 'test乙答', 'test乙再问'],
                         ['test丙问', 'test丙答', 'test丙再问', 'test丙再答']],
            }
            filenames = {'train': 'lccc_base_train.jsonl.gz',
                         'validation': 'lccc_base_valid.jsonl.gz',
                         'test': 'lccc_base_test.jsonl.gz'}
            for split, rows in dialogues.items():
                write_gzip_jsonl(source / filenames[split], rows)
            audit = {
                'version': 'lccc-base-source-audit-v1',
                'source': 'test-fixture',
                'splits': {
                    split: {'rows': len(rows), 'sha256': __import__('hashlib').sha256(
                        (source / filenames[split]).read_bytes()).hexdigest()}
                    for split, rows in dialogues.items()
                },
            }
            audit_path = root / 'audit.json'
            audit_path.write_text(json.dumps(audit), encoding='utf-8')
            output = root / 'pilot'
            manifest = prepare(source, audit_path, output, 3, 3, 3, 3, 7)
            checked = audit_lccc(output)

            self.assertEqual(manifest['selected_dialogue_counts_by_output'],
                             {'train': 3, 'validation': 3, 'eval': 3})
            self.assertEqual(checked['cross_split_exact_dialogue_overlap'], 0)
            train_rows = [json.loads(line) for line in (output / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(row['messages'][-1]['role'] == 'assistant' for row in train_rows))
            eval_rows = [json.loads(line) for line in (output / 'eval.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(row['messages'][-1]['role'] == 'user' for row in eval_rows))
            self.assertTrue(all(row['reference'] not in [message['content'] for message in row['messages']]
                                for row in eval_rows))


if __name__ == '__main__':
    unittest.main()
