"""Provider-neutral model clients used by the chat application.

The local Qwen engine predates this module. DeepSeek is implemented with the
same reply/reply_stream contract so the chat orchestration layer does not need
to know which model backend produced an answer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Callable, Iterable, Protocol

import httpx


class ProviderError(RuntimeError):
    """Base error for model-provider failures."""


class ProviderConfigurationError(ProviderError):
    """The selected provider is missing credentials or configuration."""


class ProviderResponseError(ProviderError):
    """The provider returned an invalid or incomplete response."""


@dataclass(frozen=True)
class ProviderCapabilities:
    text: bool = True
    images: bool = False
    streaming: bool = True
    json_output: bool = False
    thinking_control: bool = False


@dataclass
class ProviderReply:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    extra: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        result = {
            "seconds": self.seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "provider": self.provider,
            "model": self.model,
        }
        if self.cached_tokens:
            result["cached_tokens"] = self.cached_tokens
        if self.system_fingerprint:
            result["system_fingerprint"] = self.system_fingerprint
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        result.update(self.extra)
        return result


class ChatProvider(Protocol):
    name: str
    model_id: str
    capabilities: ProviderCapabilities

    def reply(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        ...

    def reply_stream(
        self,
        messages: list[dict],
        stop_event: Event,
        emit: Callable[[str], None],
        *,
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        ...


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _normalize_messages(messages: Iterable[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = [dict(part) for part in content if isinstance(part, dict)]
        else:
            content = _content_text(content)
        normalized.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("messages 至少需要一条 user 消息")
    return normalized


def _usage_metrics(usage: dict | None) -> tuple[int, int, int]:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cached = int(usage.get("prompt_cache_hit_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = max(cached, int(details.get("cached_tokens") or 0))
    return prompt, completion, cached


class DeepSeekProvider:
    """OpenAI-compatible DeepSeek chat client.

    The model name intentionally remains the documented rolling alias
    ``deepseek-flash``. Every response records the concrete model and
    system_fingerprint returned by the provider for later reproducibility.
    """

    name = "deepseek"
    display_name = "DeepSeek"
    model_id = "deepseek-flash"
    capabilities = ProviderCapabilities(
        text=True,
        images=True,
        streaming=True,
        json_output=True,
        thinking_control=True,
    )

    def __init__(
        self,
        api_key_provider: Callable[[], str | None],
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout_seconds: float = 120,
        transport: httpx.BaseTransport | None = None,
    ):
        self._api_key_provider = api_key_provider
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport

    def api_key(self) -> str:
        key = (self._api_key_provider() or "").strip()
        if not key:
            raise ProviderConfigurationError(f"请先在设置中填写 {self.display_name} API Key")
        return key

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    def _payload(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        generation_overrides: dict | None,
        stream: bool,
        thinking: bool,
    ) -> dict:
        overrides = dict(generation_overrides or {})
        payload = {
            "model": self.model_id,
            "messages": _normalize_messages(messages),
            "max_tokens": int(max_new_tokens),
            "stream": bool(stream),
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        for key in ("temperature", "top_p"):
            value = overrides.get(key)
            if value is not None:
                payload[key] = value
        response_format = overrides.get("response_format")
        if isinstance(response_format, dict):
            payload["response_format"] = response_format
        return payload

    def _request_headers(
        self,
        *,
        stream: bool,
        generation_overrides: dict | None = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key()}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def _usage_extra(self, usage: dict | None) -> dict:
        return {}

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = ""
        try:
            body = response.json()
            detail = str((body.get("error") or {}).get("message") or body.get("detail") or "")
        except Exception:
            detail = response.text[:500]
        if response.status_code in (401, 403):
            raise ProviderConfigurationError(f"{self.display_name} API Key 无效或没有访问权限")
        if response.status_code == 429:
            raise ProviderError(f"{self.display_name} 请求过于频繁或余额不足，请稍后重试")
        raise ProviderResponseError(
            f"{self.display_name} 请求失败（HTTP {response.status_code}）"
            + (f"：{detail}" if detail else "")
        )

    def reply(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        started = time.perf_counter()
        headers = self._request_headers(
            stream=False, generation_overrides=generation_overrides
        )
        payload = self._payload(
            messages,
            max_new_tokens=max_new_tokens,
            generation_overrides=generation_overrides,
            stream=False,
            thinking=False,
        )
        try:
            with self._client() as client:
                response = client.post("/chat/completions", headers=headers, json=payload)
        except httpx.RequestError as exc:
            raise ProviderError(
                f"无法连接 {self.display_name} API，请检查网络或系统代理设置。"
            ) from exc
        self._raise_for_response(response)
        try:
            body = response.json()
            choice = body["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(f"{self.display_name} 返回了无法解析的回答") from exc
        if not text.strip():
            raise ProviderResponseError(f"{self.display_name} 返回了空回答")
        prompt_tokens, completion_tokens, cached_tokens = _usage_metrics(body.get("usage"))
        result = ProviderReply(
            text=text,
            provider=self.name,
            model=str(body.get("model") or self.model_id),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            seconds=time.perf_counter() - started,
            system_fingerprint=body.get("system_fingerprint"),
            finish_reason=choice.get("finish_reason"),
            extra=self._usage_extra(body.get("usage")),
        )
        return result.text, result.metrics()

    def reply_stream(
        self,
        messages: list[dict],
        stop_event: Event,
        emit: Callable[[str], None],
        *,
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        started = time.perf_counter()
        headers = self._request_headers(
            stream=True, generation_overrides=generation_overrides
        )
        payload = self._payload(
            messages,
            max_new_tokens=max_new_tokens,
            generation_overrides=generation_overrides,
            stream=True,
            thinking=False,
        )
        pieces: list[str] = []
        model = self.model_id
        fingerprint = None
        finish_reason = None
        usage = None
        try:
            with self._client() as client:
                with client.stream(
                    "POST", "/chat/completions", headers=headers, json=payload
                ) as response:
                    self._raise_for_response(response)
                    for line in response.iter_lines():
                        if stop_event.is_set():
                            break
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise ProviderResponseError(
                                f"{self.display_name} 流式响应包含无效 JSON"
                            ) from exc
                        model = str(event.get("model") or model)
                        fingerprint = event.get("system_fingerprint") or fingerprint
                        usage = event.get("usage") or usage
                        for choice in event.get("choices") or []:
                            finish_reason = choice.get("finish_reason") or finish_reason
                            delta = choice.get("delta") or {}
                            text = delta.get("content")
                            if text:
                                pieces.append(text)
                                emit(text)
        except httpx.RequestError as exc:
            raise ProviderError(
                f"无法连接 {self.display_name} API，请检查网络或系统代理设置。"
            ) from exc
        text = "".join(pieces)
        if not text.strip() and not stop_event.is_set():
            raise ProviderResponseError(f"{self.display_name} 返回了空回答")
        prompt_tokens, completion_tokens, cached_tokens = _usage_metrics(usage)
        result = ProviderReply(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            seconds=time.perf_counter() - started,
            system_fingerprint=fingerprint,
            finish_reason=finish_reason,
            extra={"stopped": bool(stop_event.is_set()), **self._usage_extra(usage)},
        )
        return result.text, result.metrics()


class OpenCodeProvider(DeepSeekProvider):
    """OpenCode Go's OpenAI-compatible DeepSeek V4.1 endpoint."""

    name = "opencode"
    display_name = "OpenCode Go"
    model_id = "deepseek-v4.1-flash"

    def __init__(
        self,
        api_key_provider: Callable[[], str | None],
        *,
        base_url: str = "https://opencode.ai/zen/go/v1",
        model: str = "deepseek-v4.1-flash",
        timeout_seconds: float = 120,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = "shiyu-chat/0.9.3",
    ):
        super().__init__(
            api_key_provider,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        self.user_agent = user_agent

    def _request_headers(
        self,
        *,
        stream: bool,
        generation_overrides: dict | None = None,
    ) -> dict:
        headers = super()._request_headers(
            stream=stream, generation_overrides=generation_overrides
        )
        headers["User-Agent"] = self.user_agent
        session_id = (generation_overrides or {}).get("_opencode_session_id")
        if isinstance(session_id, str) and session_id.strip():
            headers["x-opencode-session"] = session_id.strip()
        return headers

    def _usage_extra(self, usage: dict | None) -> dict:
        usage = usage or {}
        opencode = usage.get("opencode") or {}
        extra = {}
        if isinstance(opencode, dict):
            if opencode.get("billing_source") is not None:
                extra["billing_source"] = opencode.get("billing_source")
            if opencode.get("list_cost") is not None:
                extra["list_cost"] = opencode.get("list_cost")
            if opencode.get("allowance") is not None:
                extra["allowance"] = opencode.get("allowance")
        if usage.get("cost") is not None:
            extra["cost"] = usage.get("cost")
        return extra


def estimate_tokens(messages: list[dict]) -> int:
    """Conservative local estimate used before exact provider usage is known."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += max(1, (len(content) + 1) // 2)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") in ("text", "input_text"):
                    total += max(1, (len(str(part.get("text", ""))) + 1) // 2)
                elif part.get("type") in ("image_url", "input_image"):
                    total += 1024
        total += 8
    return total


class DeepSeekChatEngine:
    """Adapter matching the local QwenChat interface used by the web backend."""

    adapter_available = False
    tokenizer = None
    supports_vision = True

    def __init__(
        self,
        provider: DeepSeekProvider,
        *,
        input_budget: int = 16384,
    ):
        self.provider = provider
        self.provider_name = getattr(provider, "name", "deepseek")
        self.display_name = getattr(provider, "display_name", self.provider_name)
        self.model_id = provider.model_id
        self.input_budget = int(input_budget)
        self.same_chat_memory = False

    def reply(
        self,
        messages: list[dict],
        model_mode: str = "base",
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        if model_mode != "base":
            raise ValueError(f"{self.display_name} 模式不支持本地 LoRA Adapter")
        return self.provider.reply(
            messages,
            max_new_tokens=max_new_tokens,
            generation_overrides=generation_overrides,
        )

    def reply_stream(
        self,
        messages: list[dict],
        model_mode: str,
        stop_event: Event,
        emit: Callable[[str], None],
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        if model_mode != "base":
            raise ValueError(f"{self.display_name} 模式不支持本地 LoRA Adapter")
        return self.provider.reply_stream(
            messages,
            stop_event,
            emit,
            max_new_tokens=max_new_tokens,
            generation_overrides=generation_overrides,
        )


class MockProvider:
    """Deterministic provider for local tests and offline UI development."""

    name = "mock"
    model_id = "mock-character"
    capabilities = ProviderCapabilities(text=True, images=True)

    def __init__(self, answers: dict[str, str] | None = None, default: str = "我在。"):
        self.answers = answers or {}
        self.default = default
        self.calls: list[list[dict]] = []

    def _answer(self, messages: list[dict]) -> str:
        last = next(
            (
                _content_text(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        return self.answers.get(last, self.default)

    def reply(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        self.calls.append([dict(message) for message in messages])
        text = self._answer(messages)
        return text, {
            "seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": len(text),
            "provider": self.name,
            "model": self.model_id,
        }

    def reply_stream(
        self,
        messages: list[dict],
        stop_event: Event,
        emit: Callable[[str], None],
        *,
        max_new_tokens: int = 512,
        generation_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        text, metrics = self.reply(
            messages,
            max_new_tokens=max_new_tokens,
            generation_overrides=generation_overrides,
        )
        if not stop_event.is_set():
            emit(text)
        return text, metrics

