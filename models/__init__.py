"""
APEX v2.0 Model Layer
Multi-model with auth profiles, failover, circuit breaker (from OpenClaw patterns).
Combines Hermes v1 simplicity with OpenClaw's provider reliability system.
"""
from __future__ import annotations

import asyncio
import time
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from apex.config import ModelRouterConfig, ModelProfile

logger = logging.getLogger("apex.models")


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role | str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""
    images: list[dict[str, Any]] = field(default_factory=list)  # for vision models

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": str(self.role.value if hasattr(self.role, 'value') else self.role),
            "content": self.content,
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    finish_reason: str = "stop"
    error: str = ""
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def success(self) -> bool:
        return not self.error


class CircuitBreaker:
    """Circuit breaker for provider failover (from OpenClaw failover-policy)."""

    def __init__(self, threshold: int = 3, cooldown: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def record_failure(self, provider: str):
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= self.threshold:
            self._opened_at[provider] = time.time()
            logger.warning(f"Circuit breaker OPEN for {provider}")

    def record_success(self, provider: str):
        self._failures.pop(provider, None)
        self._opened_at.pop(provider, None)

    def is_open(self, provider: str) -> bool:
        if provider not in self._opened_at:
            return False
        elapsed = time.time() - self._opened_at[provider]
        if elapsed >= self.cooldown:
            # Half-open: allow one request through
            self._opened_at.pop(provider, None)
            self._failures.pop(provider, None)
            logger.info(f"Circuit breaker HALF-OPEN for {provider}")
            return False
        return True


class BaseProvider(ABC):
    """Abstract base for all model providers."""

    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.name = profile.model
        self.provider_name = profile.provider

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        ...


class OllamaProvider(BaseProvider):
    """Ollama local model provider."""

    def __init__(self, profile: ModelProfile):
        super().__init__(profile)
        self.api_base = profile.api_base or "http://localhost:11434"

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
                   **kwargs: Any) -> ModelResponse:
        import aiohttp
        start = time.time()
        url = f"{self.api_base}/api/chat"
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.profile.temperature),
                "num_predict": kwargs.get("max_tokens", self.profile.max_tokens),
            },
        }
        if tools:
            payload["format"] = "json"  # Ollama JSON mode for tool calling

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return ModelResponse(
                        content="", tool_calls=[], model=self.name,
                        provider=self.provider_name, error=f"HTTP {resp.status}: {error_text[:200]}",
                        latency=time.time() - start,
                    )
                data = await resp.json()

        latency = time.time() - start
        msg = data.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = []

        # Parse Ollama tool calls from content if present
        if tools and "<tool_call>" in content:
            try:
                import re
                tc_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
                if tc_match:
                    tc_data = json.loads(tc_match.group(1))
                    tool_calls = [{"id": f"call_{int(time.time())}", "type": "function",
                                   "function": {"name": tc_data["name"],
                                                "arguments": json.dumps(tc_data.get("arguments", {}))}}]
                    content = content[:tc_match.start()].strip()
            except Exception:
                pass

        return ModelResponse(
            content=content, tool_calls=tool_calls,
            model=self.name, provider=self.provider_name,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency=latency, raw=data,
        )

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
                     **kwargs: Any) -> AsyncIterator[str]:
        import aiohttp
        url = f"{self.api_base}/api/chat"
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {"temperature": kwargs.get("temperature", self.profile.temperature)},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                async for line in resp.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI / OpenAI-compatible provider (Anthropic, Google, etc.)."""

    def __init__(self, profile: ModelProfile):
        super().__init__(profile)
        self.api_base = profile.api_base or "https://api.openai.com/v1"

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
                   **kwargs: Any) -> ModelResponse:
        import aiohttp
        start = time.time()
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.profile.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.profile.temperature),
            "max_tokens": kwargs.get("max_tokens", self.profile.max_tokens),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error = data.get("error", {}).get("message", f"HTTP {resp.status}")
                    return ModelResponse(
                        content="", tool_calls=[], model=self.name,
                        provider=self.provider_name, error=error,
                        latency=time.time() - start,
                    )

        latency = time.time() - start
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        # Normalize tool call format
        normalized_tcs = []
        for tc in tool_calls:
            normalized_tcs.append({
                "id": tc.get("id", f"call_{int(time.time())}"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            })

        usage = data.get("usage", {})
        return ModelResponse(
            content=content, tool_calls=normalized_tcs,
            model=self.name, provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency=latency,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
                     **kwargs: Any) -> AsyncIterator[str]:
        import aiohttp
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.profile.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "temperature": kwargs.get("temperature", self.profile.temperature),
        }
        if tools:
            payload["tools"] = tools

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: ") and text != "data: [DONE]":
                        try:
                            chunk = json.loads(text[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass


# ── Provider Registry ──

PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": OpenAICompatibleProvider,
    "google": OpenAICompatibleProvider,
}


class ModelRouter:
    """
    Multi-model router (from OpenClaw model-fallback + auth-profiles pattern).
    - Auth profiles with named model configurations
    - Auto-failover with circuit breaker
    - Cooldown before retrying failed providers
    - Cost tracking
    """

    def __init__(self, config: ModelRouterConfig):
        self.config = config
        self.providers: dict[str, BaseProvider] = {}
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            cooldown=config.cooldown_seconds,
        )
        self.cost_log: list[dict[str, Any]] = []
        self._register_profiles(config)

    def _register_profiles(self, config: ModelRouterConfig):
        for name, profile in config.profiles.items():
            cls = PROVIDER_CLASSES.get(profile.provider)
            if not cls:
                logger.warning(f"Unknown provider: {profile.provider}")
                continue
            self.providers[name] = cls(profile)
            logger.info(f"Registered model profile: {name} ({profile.provider}/{profile.model})")

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        profile: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send chat request with auto-failover."""
        # Determine order: requested → primary → fallbacks
        order: list[str] = []
        if profile and profile in self.providers:
            order.append(profile)
        if self.config.primary in self.providers and self.config.primary not in order:
            order.append(self.config.primary)
        for fb in self.config.fallbacks:
            if fb in self.providers and fb not in order:
                order.append(fb)
        # If still empty, use all registered providers
        if not order:
            order = list(self.providers.keys())

        # Filter out providers with open circuits
        order = [p for p in order if not self.circuit_breaker.is_open(p)]

        if not order:
            return ModelResponse(
                content="", tool_calls=[], model="", provider="",
                error="All model providers are unavailable (circuit breaker open)",
            )

        last_error = ""
        for name in order:
            provider = self.providers[name]
            try:
                logger.debug(f"Trying {name} ({provider.provider_name}/{provider.name})")
                resp = await provider.chat(messages, tools, **kwargs)
                if resp.success:
                    self.circuit_breaker.record_success(name)
                    self._log_cost(resp)
                    return resp
                else:
                    logger.warning(f"Provider {name} returned error: {resp.error}")
                    self.circuit_breaker.record_failure(name)
                    last_error = resp.error
            except Exception as e:
                logger.warning(f"Provider {name} failed: {e}")
                self.circuit_breaker.record_failure(name)
                last_error = str(e)

        return ModelResponse(
            content="", tool_calls=[], model="", provider="",
            error=f"All providers failed. Last error: {last_error}",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        profile: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat with auto-failover."""
        order: list[str] = []
        if profile and profile in self.providers:
            order.append(profile)
        if self.config.primary in self.providers and self.config.primary not in order:
            order.append(self.config.primary)
        for fb in self.config.fallbacks:
            if fb in self.providers and fb not in order:
                order.append(fb)
        if not order:
            order = list(self.providers.keys())

        order = [p for p in order if not self.circuit_breaker.is_open(p)]

        for name in order:
            provider = self.providers[name]
            try:
                async for chunk in provider.stream(messages, tools, **kwargs):
                    yield chunk
                self.circuit_breaker.record_success(name)
                return
            except Exception as e:
                logger.warning(f"Stream from {name} failed: {e}")
                self.circuit_breaker.record_failure(name)
                continue

    def _log_cost(self, resp: ModelResponse):
        self.cost_log.append({
            "timestamp": time.time(),
            "provider": resp.provider,
            "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency": resp.latency,
        })

    def get_status(self) -> dict[str, Any]:
        """Get status of all providers."""
        return {
            name: {
                "provider": p.provider_name,
                "model": p.name,
                "circuit_open": self.circuit_breaker.is_open(name),
            }
            for name, p in self.providers.items()
        }
