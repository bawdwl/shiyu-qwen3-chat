import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend import create_app


class FakeEngine:
    adapter_available = False
    model_id = 'Qwen/Qwen3-1.7B'

    def reply(self, *args, **kwargs):
        return 'test', {'seconds': 0.01, 'output_tokens': 1}


class LanPairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        app = create_app(
            Path(self.temp.name) / 'lan.db',
            engine_factory=FakeEngine,
            lan_enabled=True,
            lan_pair_code='123456',
        )
        self.local = TestClient(app)
        self.remote = TestClient(app, client=('192.0.2.10', 43122))
        self.local.__enter__()

    def tearDown(self):
        self.local.__exit__(None, None, None)
        self.temp.cleanup()

    def test_lan_pairs_once_and_status_requires_token(self):
        self.assertEqual(self.remote.get('/health').status_code, 403)
        self.assertEqual(self.remote.get('/characters').status_code, 403)
        self.assertEqual(self.remote.get('/lan/status').status_code, 401)

        paired = self.remote.post('/lan/pair', json={
            'pair_code': '123456', 'device_name': 'OnePlus Ace 5',
        })
        self.assertEqual(paired.status_code, 200)
        data = paired.json()
        self.assertTrue(data['device_token'])
        self.assertTrue(data['expires_on_restart'])

        status = self.remote.get('/lan/status', headers={
            'Authorization': 'Bearer ' + data['device_token'],
        })
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()['model_id'], 'Qwen/Qwen3-1.7B')
        self.assertEqual(self.remote.post('/lan/pair', json={
            'pair_code': '123456', 'device_name': 'again',
        }).status_code, 403)

        revoked = self.local.post(f"/lan/devices/{data['device_id']}/revoke")
        self.assertEqual(revoked.status_code, 200)
        self.assertEqual(self.remote.get('/lan/status', headers={
            'Authorization': 'Bearer ' + data['device_token'],
        }).status_code, 401)

    def test_lan_defaults_closed(self):
        app = create_app(
            Path(self.temp.name) / 'closed.db',
            engine_factory=FakeEngine,
        )
        client = TestClient(app, client=('192.0.2.10', 43122))
        with client:
            self.assertEqual(client.get('/health').status_code, 403)
            self.assertEqual(client.post('/lan/pair', json={
                'pair_code': '123456', 'device_name': 'phone',
            }).status_code, 403)

    def test_pairing_code_locks_after_five_failed_attempts(self):
        for _ in range(5):
            self.assertEqual(self.remote.post('/lan/pair', json={
                'pair_code': '000000', 'device_name': 'unknown',
            }).status_code, 403)
        self.assertEqual(self.remote.post('/lan/pair', json={
            'pair_code': '123456', 'device_name': 'OnePlus Ace 5',
        }).status_code, 403)


if __name__ == '__main__':
    unittest.main()
