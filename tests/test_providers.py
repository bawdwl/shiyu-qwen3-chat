import json
import unittest
from threading import Event

import httpx

from providers import (
    DeepSeekProvider,
    MockProvider,
    OpenCodeProvider,
    ProviderConfigurationError,
    ProviderError,
)


class DeepSeekProviderTests(unittest.TestCase):
    def test_reply_uses_deepseek_flash_and_reports_usage(self):
        seen = {}

        def handler(request):
            seen["authorization"] = request.headers["authorization"]
            seen["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "model": "deepseek-flash",
                    "system_fingerprint": "fp-test",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "测试回答"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "prompt_cache_hit_tokens": 3,
                    },
                },
            )

        provider = DeepSeekProvider(
            lambda: "sk-test", transport=httpx.MockTransport(handler)
        )
        answer, metrics = provider.reply(
            [{"role": "user", "content": "你好"}],
            max_new_tokens=32,
            generation_overrides={"temperature": 0.7},
        )

        self.assertEqual(answer, "测试回答")
        self.assertEqual(seen["authorization"], "Bearer sk-test")
        self.assertEqual(seen["payload"]["model"], "deepseek-flash")
        self.assertEqual(seen["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(seen["payload"]["temperature"], 0.7)
        self.assertEqual(metrics["input_tokens"], 12)
        self.assertEqual(metrics["output_tokens"], 4)
        self.assertEqual(metrics["cached_tokens"], 3)
        self.assertEqual(metrics["system_fingerprint"], "fp-test")

    def test_reply_stream_collects_deltas_and_usage(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["stream_options"], {"include_usage": True})
            body = (
                'data: {"model":"deepseek-flash","choices":[{"delta":{"content":"你"},"finish_reason":null}]}\n\n'
                'data: {"model":"deepseek-flash","choices":[{"delta":{"content":"好"},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )

        provider = DeepSeekProvider(
            lambda: "sk-test", transport=httpx.MockTransport(handler)
        )
        pieces = []
        answer, metrics = provider.reply_stream(
            [{"role": "user", "content": "你好"}],
            Event(),
            pieces.append,
            max_new_tokens=32,
        )

        self.assertEqual(pieces, ["你", "好"])
        self.assertEqual(answer, "你好")
        self.assertEqual(metrics["input_tokens"], 5)
        self.assertEqual(metrics["output_tokens"], 2)

    def test_accepts_multimodal_user_content(self):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "我看到图片了"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        provider = DeepSeekProvider(
            lambda: "sk-test", transport=httpx.MockTransport(handler)
        )
        provider.reply(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是什么？"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ]
        )
        self.assertIsInstance(captured["messages"][0]["content"], list)
        self.assertEqual(captured["messages"][0]["content"][1]["type"], "image_url")

    def test_missing_key_fails_before_network_request(self):
        provider = DeepSeekProvider(lambda: "")
        with self.assertRaises(ProviderConfigurationError):
            provider.reply([{"role": "user", "content": "你好"}])

    def test_network_failure_has_actionable_error(self):
        def handler(request):
            raise httpx.ConnectError("blocked", request=request)

        provider = DeepSeekProvider(
            lambda: "sk-test", transport=httpx.MockTransport(handler)
        )
        with self.assertRaisesRegex(ProviderError, "网络或系统代理"):
            provider.reply([{"role": "user", "content": "你好"}])

    def test_mock_provider_stream_contract(self):
        provider = MockProvider({"你好": "你好呀"})
        pieces = []
        answer, _ = provider.reply_stream(
            [{"role": "user", "content": "你好"}], Event(), pieces.append
        )
        self.assertEqual(answer, "你好呀")
        self.assertEqual(pieces, ["你好呀"])

    def test_opencode_uses_go_endpoint_session_header_and_usage(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["authorization"] = request.headers["authorization"]
            seen["session"] = request.headers["x-opencode-session"]
            seen["user_agent"] = request.headers["user-agent"]
            seen["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-v4.1-flash",
                    "choices": [{
                        "message": {"role": "assistant", "content": "准备好了"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "cost": 0,
                        "opencode": {
                            "billing_source": "go",
                            "list_cost": 0.00001,
                            "allowance": 0.000007,
                        },
                    },
                },
            )

        provider = OpenCodeProvider(
            lambda: "oc_sk_test",
            transport=httpx.MockTransport(handler),
        )
        answer, metrics = provider.reply(
            [{"role": "user", "content": "你好"}],
            generation_overrides={"_opencode_session_id": "session-test"},
        )
        self.assertEqual(answer, "准备好了")
        self.assertEqual(seen["url"], "https://opencode.ai/zen/go/v1/chat/completions")
        self.assertEqual(seen["authorization"], "Bearer oc_sk_test")
        self.assertEqual(seen["session"], "session-test")
        self.assertEqual(seen["user_agent"], "shiyu-chat/0.9.3")
        self.assertEqual(seen["payload"]["model"], "deepseek-v4.1-flash")
        self.assertNotIn("_opencode_session_id", seen["payload"])
        self.assertEqual(metrics["provider"], "opencode")
        self.assertEqual(metrics["billing_source"], "go")
        self.assertEqual(metrics["list_cost"], 0.00001)


if __name__ == "__main__":
    unittest.main()
