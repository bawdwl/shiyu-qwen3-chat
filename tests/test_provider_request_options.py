import json
import sys
import unittest
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from providers import DeepSeekProvider


class ProviderRequestOptionTests(unittest.TestCase):
    def test_json_response_format_reaches_openai_compatible_payload(self):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    'model': 'deepseek-flash',
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': '{"candidates":[]}',
                        },
                        'finish_reason': 'stop',
                    }],
                    'usage': {
                        'prompt_tokens': 3,
                        'completion_tokens': 3,
                    },
                },
            )

        provider = DeepSeekProvider(
            lambda: 'sk-test',
            transport=httpx.MockTransport(handler),
        )
        provider.reply(
            [{'role': 'user', 'content': 'return json'}],
            generation_overrides={
                'temperature': 0,
                'response_format': {'type': 'json_object'},
            },
        )
        self.assertEqual(
            captured['response_format'],
            {'type': 'json_object'},
        )
        self.assertEqual(captured['temperature'], 0)


if __name__ == '__main__':
    unittest.main()
