"""覆盖尚未发出SSE请求时点击停止的窗口。"""
import unittest
from unittest.mock import patch
from api_client import BackendClient


class ClientTests(unittest.TestCase):
    def test_stop_before_stream_does_not_start_generation(self):
        client=BackendClient();client.prepare('fixture')
        with patch.object(client,'request',return_value={'stop_requested':False}) as request:
            self.assertTrue(client.stop('fixture')['stop_requested'])
            with patch('api_client.httpx.Client',side_effect=AssertionError('generation request must not start')):
                with self.assertRaisesRegex(RuntimeError,'尚未发送'):
                    list(client.stream('fixture','test'))
        self.assertNotIn('fixture',client._stops)

    def test_idle_stop_does_not_create_pending_generation(self):
        client=BackendClient()
        with patch.object(client,'request',return_value={'stop_requested':False}):
            self.assertFalse(client.stop('fixture')['stop_requested'])
        self.assertEqual(client._stops,{})
