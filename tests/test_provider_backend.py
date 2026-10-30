import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from backend import create_app
from providers import DeepSeekProvider, OpenCodeProvider


class DeepSeekBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.requests = []
        self.answer = "那就先歇一会儿，我在。"

        def handler(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            if payload.get("stream"):
                event = {
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "delta": {"content": self.answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 5,
                    },
                }
                return httpx.Response(
                    200,
                    text=f"data: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-flash",
                    "system_fingerprint": "fp-test",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": self.answer,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 5,
                        "prompt_cache_hit_tokens": 2,
                    },
                },
            )

        provider = DeepSeekProvider(
            lambda: "sk-test", transport=httpx.MockTransport(handler)
        )
        self.client = TestClient(
            create_app(
                Path(self.temp.name) / "chat.sqlite3",
                engine_factory=lambda: None,
                default_provider="deepseek",
                deepseek_provider=provider,
            )
        )
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_key_configuration_and_health(self):
        before = self.client.get("/health")
        self.assertEqual(before.status_code, 503)
        self.assertFalse(before.json()["providers"]["deepseek"]["configured"])

        response = self.client.put(
            "/settings/deepseek-key", json={"api_key": "sk-very-secret-test"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("sk-very-secret-test", response.text)
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["providers"]["deepseek"]["configured"])

    def test_single_role_conversation_and_deepseek_chat_usage(self):
        self.client.put(
            "/settings/deepseek-key", json={"api_key": "sk-very-secret-test"}
        )
        conversation = self.client.put(
            "/characters/companion/conversation", params={"provider": "deepseek"}
        )
        self.assertEqual(conversation.status_code, 200)
        sid = conversation.json()["id"]
        self.assertEqual(conversation.json()["messages"][0]["role"], "assistant")
        self.assertEqual(
            conversation.json()["messages"][0]["content"],
            "今天过得怎么样？不用组织好语言，随便说说就好。",
        )
        same = self.client.put(
            "/characters/companion/conversation", params={"provider": "deepseek"}
        )
        self.assertEqual(same.json()["id"], sid)
        self.assertEqual(len(same.json()["messages"]), 1)

        response = self.client.post(
            "/chat",
            json={
                "session_id": sid,
                "message": "今天有点累，陪我聊两句",
                "max_new_tokens": 64,
                "request_id": "523d0116-12d1-4ed3-820f-4b1872e50367",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["provider"], "deepseek")
        self.assertEqual(response.json()["answer"], "那就先歇一会儿，我在。")
        self.assertEqual(self.requests[-1]["model"], "deepseek-flash")
        self.assertEqual(self.requests[-1]["thinking"], {"type": "disabled"})
        usage = self.client.get("/health").json()["deepseek_usage"]
        self.assertEqual(usage["total_tokens"], 14)
        self.assertEqual(usage["cached_tokens"], 2)

    def test_configurable_request_limit_blocks_large_call(self):
        self.client.put(
            "/settings/deepseek-key", json={"api_key": "sk-very-secret-test"}
        )
        limits = self.client.put(
            "/settings/token-limits",
            json={"request_token_limit": 256, "daily_token_limit": 5000},
        )
        self.assertEqual(limits.status_code, 200)
        sid = self.client.put("/characters/companion/conversation").json()["id"]
        response = self.client.post(
            "/chat",
            json={
                "session_id": sid,
                "message": "长" * 1200,
                "max_new_tokens": 64,
            },
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("request_token_limit", response.text)

    def test_image_upload_is_sent_multimodal_and_removed_on_reset(self):
        self.client.put(
            "/settings/deepseek-key", json={"api_key": "sk-very-secret-test"}
        )
        sid = self.client.put("/characters/companion/conversation").json()["id"]
        image = Image.new("RGB", (64, 48), "white")
        output = BytesIO()
        image.save(output, format="PNG")

        uploaded = self.client.post(
            "/images",
            params={"session_id": sid},
            content=output.getvalue(),
            headers={"content-type": "image/png"},
        )
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        image_id = uploaded.json()["id"]
        self.assertEqual(uploaded.json()["mime_type"], "image/webp")

        response = self.client.post(
            "/chat",
            json={
                "session_id": sid,
                "message": "这张图里有什么？",
                "image_id": image_id,
                "max_new_tokens": 64,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        sent_content = self.requests[-1]["messages"][-1]["content"]
        self.assertIsInstance(sent_content, list)
        self.assertEqual(sent_content[1]["type"], "image_url")
        saved = self.client.get(f"/sessions/{sid}").json()["messages"]
        image_message = next(message for message in saved if message.get("image"))
        self.assertEqual(image_message["image"]["id"], image_id)

        reset = self.client.delete(
            "/characters/companion/conversation/messages"
        )
        self.assertEqual(reset.status_code, 200, reset.text)
        self.assertEqual(len(reset.json()["messages"]), 1)
        self.assertEqual(reset.json()["messages"][0]["role"], "assistant")
        self.assertEqual(
            reset.json()["messages"][0]["content"],
            "今天过得怎么样？不用组织好语言，随便说说就好。",
        )
        self.assertEqual(self.client.get(
            f"/images/{image_id}", params={"session_id": sid}
        ).status_code, 404)

    def test_strict_historical_boundary_buffers_and_replaces_leaking_answer(self):
        self.client.put(
            "/settings/deepseek-key", json={"api_key": "sk-very-secret-test"}
        )
        card = {
            "id": "custom-libai",
            "name": "李白",
            "tagline": "唐代诗人",
            "description": "身处唐代的原创化角色",
            "background": "生活在唐代。",
            "personality": "豪放、敏锐。",
            "speech_style": "古代口吻，但不做作。",
            "relationship": "初次交谈。",
            "greeting": "你从何处来？",
            "knowledge_boundary": {
                "character_kind": "historical",
                "era": "唐代",
                "knowledge_cutoff": "唐代",
                "universe": "中国古代",
                "allowed_knowledge": "诗词、饮酒、游历",
                "forbidden_knowledge": "现代科技",
                "forbidden_terms": ["python", "电脑"],
                "can_learn_from_user": True,
                "allow_everyday_invention": True,
            },
        }
        saved = self.client.post("/characters", json={"card": card})
        self.assertEqual(saved.status_code, 201, saved.text)
        sid = self.client.put(
            "/characters/custom-libai/conversation"
        ).json()["id"]
        self.answer = "Python 是一种现代编程语言。"

        streamed = self.client.post(
            "/chat/stream",
            json={
                "session_id": sid,
                "message": "李白，Python 是什么？",
                "max_new_tokens": 64,
                "timeout_seconds": 20,
            },
        )
        self.assertEqual(streamed.status_code, 200, streamed.text)
        self.assertIn("此事我未曾听闻", streamed.text)
        self.assertNotIn("现代编程语言", streamed.text)
        self.assertIn("knowledge_guard", streamed.text)

    def test_web_character_avatar_generation_is_removed(self):
        card = {
            "id": "custom-avatar-test",
            "name": "头像测试",
            "tagline": "测试角色",
            "description": "用于验证本地头像接口",
            "background": "",
            "personality": "安静",
            "speech_style": "简短",
            "relationship": "朋友",
            "greeting": "你好",
        }
        saved = self.client.post("/characters", json={"card": card})
        self.assertEqual(saved.status_code, 201, saved.text)
        generated = self.client.post("/characters/custom-avatar-test/avatar")
        self.assertEqual(generated.status_code, 405)
        self.assertEqual(
            self.client.get("/characters/custom-avatar-test/avatar").status_code,
            404,
        )


class OpenCodeBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.requests = []
        self.answer = "我在听。"

        def handler(request):
            payload = json.loads(request.content)
            self.requests.append({
                "path": request.url.path,
                "headers": dict(request.headers),
                "payload": payload,
            })
            if payload.get("stream"):
                event = {
                    "model": "deepseek-v4.1-flash",
                    "choices": [{
                        "delta": {"content": self.answer},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 5,
                        "opencode": {
                            "billing_source": "go",
                            "list_cost": 0.00001,
                            "allowance": 0.000007,
                        },
                    },
                }
                return httpx.Response(
                    200,
                    text=f"data: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-v4.1-flash",
                    "choices": [{
                        "message": {"role": "assistant", "content": self.answer},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 5,
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
        self.client = TestClient(
            create_app(
                Path(self.temp.name) / "chat.sqlite3",
                engine_factory=lambda: None,
                default_provider="opencode",
                opencode_provider=provider,
            )
        )
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_key_health_chat_headers_and_usage(self):
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(
            self.client.put(
                "/settings/opencode-key", json={"api_key": "sk-not-opencode"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.put(
                "/settings/opencode-key", json={"api_key": "oc_sk_valid_test"}
            ).status_code,
            200,
        )
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["providers"]["opencode"]["configured"])

        conversation = self.client.put(
            "/characters/companion/conversation",
            params={"provider": "opencode"},
        )
        self.assertEqual(conversation.status_code, 200)
        self.assertEqual(conversation.json()["provider"], "opencode")
        self.assertEqual(conversation.json()["model_id"], "deepseek-v4.1-flash")
        sid = conversation.json()["id"]

        response = self.client.post(
            "/chat",
            json={
                "session_id": sid,
                "message": "你好",
                "max_new_tokens": 32,
                "request_id": "3e9aa595-b487-4475-a1ed-929df33ae2a0",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["provider"], "opencode")
        self.assertEqual(self.requests[-1]["path"], "/zen/go/v1/chat/completions")
        self.assertEqual(self.requests[-1]["payload"]["model"], "deepseek-v4.1-flash")
        self.assertEqual(
            self.requests[-1]["headers"]["x-opencode-session"], sid
        )
        self.assertEqual(
            self.requests[-1]["headers"]["user-agent"], "shiyu-chat/0.9.3"
        )
        usage = self.client.get("/health").json()["opencode_usage"]
        self.assertEqual(usage["total_tokens"], 14)

    def test_existing_role_conversation_switches_provider_without_losing_messages(self):
        self.client.put(
            "/settings/opencode-key", json={"api_key": "oc_sk_valid_test"}
        )
        existing = self.client.put(
            "/characters/companion/conversation",
            params={"provider": "opencode"},
        ).json()
        self.client.post(
            "/chat",
            json={"session_id": existing["id"], "message": "保留这段聊天"},
        )
        switched = self.client.put(
            "/characters/companion/conversation",
            params={"provider": "opencode"},
        ).json()
        self.assertEqual(switched["id"], existing["id"])
        self.assertEqual(len(switched["messages"]), 3)
        self.assertEqual(switched["provider"], "opencode")
        self.assertEqual(switched["model_id"], "deepseek-v4.1-flash")


if __name__ == "__main__":
    unittest.main()
