"""Prevent newly downloaded or derived corpora from entering local release zips."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from audit_release import audit
from package_release import PUBLIC_DATA_FILES, allowed_data_path


class ReleaseDataTests(unittest.TestCase):
    def test_original_examples_are_explicitly_allowed(self):
        for name in ('data/train.jsonl', 'data/v2/train.jsonl', 'data/im_chat_development_v2.json'):
            self.assertIn(name, PUBLIC_DATA_FILES)
            self.assertTrue(allowed_data_path(name))

    def test_third_party_and_unknown_corpora_are_excluded(self):
        for name in ('data/LCCC-base/audit/report.json', 'data/lccc-v1-pilot/train.jsonl',
                     'data/NaturalConv/train.json', 'data/naturalconv-v1/validation.jsonl',
                     'data/DuRecDial2/train.json', 'data/future-corpus/train.jsonl'):
            self.assertFalse(allowed_data_path(name), name)

    def test_renaming_into_unknown_subdirectory_does_not_allow_data(self):
        self.assertFalse(allowed_data_path('data/new/train.jsonl'))
        self.assertFalse(allowed_data_path('data/v2/../private.json'))

    def test_auditor_rejects_manifest_valid_but_unapproved_dataset(self):
        content = b'{"text":"private research fixture"}\n'
        relative = 'data/future-corpus/train.jsonl'
        manifest = [dict(path=relative, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'release.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('qwen3-im-chat/' + relative, content)
                archive.writestr('qwen3-im-chat/RELEASE-MANIFEST.json', json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Unapproved dataset'):
                audit(path)


if __name__ == '__main__':
    unittest.main()
