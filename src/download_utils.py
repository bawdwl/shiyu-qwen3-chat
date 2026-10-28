"""Download a local model file and verify its expected SHA-256 when available."""
import hashlib
import urllib.request


def fetch(url, target, expected_sha256=None):
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Existing file hash mismatch: ' + str(target))
        return digest
    partial = target.with_suffix(target.suffix + '.part')
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError('Download hash mismatch; partial file retained')
    partial.replace(target)
    return digest.hexdigest()
