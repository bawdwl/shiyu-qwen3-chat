"""Check the explicit release bundle; never scan unrelated user directories."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from package_release import allowed_data_path


def audit(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entry')
        manifest_name = 'qwen3-im-chat/RELEASE-MANIFEST.json'
        records = json.loads(archive.read(manifest_name))
        expected = {manifest_name}
        forbidden = {'.sqlite3', '.db', '.docx', '.safetensors', '.bin', '.onnx', '.mnn', '.weight', '.pem', '.key', '.pyc'}
        secret = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:ghp_|hf_)[A-Za-z0-9]{25,}\b')
        for record in records:
            relative = PurePosixPath(record['path'])
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
                raise ValueError('Unsafe manifest path')
            if set(relative.parts) & {'outputs', '__pycache__', '.git', '.cache'} or relative.suffix in forbidden:
                raise ValueError('Excluded file: ' + str(relative))
            if relative.parts and relative.parts[0] == 'data' and not allowed_data_path(relative):
                raise ValueError('Unapproved dataset in release: ' + str(relative))
            name = 'qwen3-im-chat/' + str(relative)
            expected.add(name)
            content = archive.read(name)
            if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
                raise ValueError('Manifest mismatch: ' + str(relative))
            if relative.suffix in {'.py', '.js', '.cjs', '.json', '.jsonl', '.md', '.txt', '.html'} and secret.search(content):
                # Report paths only, never print the matched secret.
                raise ValueError('Potential credential in: ' + str(relative))
        if set(names) != expected:
            raise ValueError('Archive contains unlisted files')
        required = {'LICENSE', 'LICENSE-NOTICE.md', 'README.md', 'web/app.js',
                    'tests/frontend_state.test.cjs', 'docs/architecture.md',
                    'docs/api-guide.md', 'docs/求职项目目标与岗位差距.md',
                    'docs/MNN本机实跑记录.md', 'docs/mnn-cpu-smoke.json'}
        if not required.issubset({r['path'] for r in records}):
            raise ValueError('Required release materials missing')
    return dict(archive=str(path), verified_files=len(records),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                limitations='Pattern-based credential check; not a comprehensive privacy or security audit. Any later-added screenshots require visual review.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    print(json.dumps(audit(parser.parse_args().archive), ensure_ascii=False, indent=2))
