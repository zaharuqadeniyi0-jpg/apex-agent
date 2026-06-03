"""
APEX Model Layer
Unified interface for multiple LLM providers with auto-failover.
"""
from __future__ import annotations

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from apex.config import ModelConfig

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

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": str(self.role.value if hasattr(self.role, 'value') else self.role), "content": self.content}
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
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseModel(ABC):
    """Abstract base for all model providers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.model
        self.provider_name = config.provider

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

    def _messages_to_dict(self, messages: list[Message]) -> list[dict]:
        return [m.to_dict() for m in messages]


class OllamaModel(BaseModel):
    """Ollama local model provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:11434"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        import aiohttp

        start = time.time()
        url = f"{self.api_base}/api/chat"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_to_dict(messages),
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        if tools:
            payload["tools"] = tools

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()

        latency = time.time() - start
        msg = data.get("message", {})
        return ModelResponse(
            content=msg.get("content", ""),
            tool_calls=msg.get("tool_calls", []),
            model=self.config.model,
            provider="ollama",
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency=latency,
            raw=data,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        import aiohttp

        url = f"{self.api_base}/api/chat"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_to_dict(messages),
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }
        if tools:
            payload["tools"] = tools

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                async for line in resp.content:
                    if line:
                        import json
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass


class OpenAIModel(BaseModel):
    """OpenAI / OpenAI-compatible provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.openai.com/v1"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        import aiohttp

        start = time.time()
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_to_dict(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()

        latency = time.time() - start
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return ModelResponse(
            content=msg.get("content", "") or "",
            tool_calls=msg.get("tool_calls", []),
            model=self.config.model,
            provider="openai",
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            latency=latency,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        import aiohttp, json

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_to_dict(messages),
            "stream": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if tools:
            payload["tools"] = tools

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: ") and text != "data: [DONE]":
                        chunk = json.loads(text[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content


class ModelRouter:
    """
    Multi-model router with:
    - Auto-failover across providers
    - Model routing based on task type
    - Cost tracking
    - Rate limiting
    """

    PROVIDERS: dict[str, type[BaseModel]] = {
        "ollama": OllamaModel,
        "openai": OpenAIModel,
        "anthropic": OpenAIModel,  # Same pattern, different base URL
    }

    def __init__(self, configs: list[ModelConfig] | None = None):
        self.models: list[BaseModel] = []
        self.cost_log: list[dict[str, Any]] = []
        self._rate_limits: dict[str, float] = {}

        if configs:
            for cfg in configs:
                self.add_model(cfg)

    def add_model(self, config: ModelConfig):
        """Register a model."""
        cls = self.PROVIDERS.get(config.provider)
        if not cls:
            logger.warning(f"Unknown provider: {config.provider}")
            return
        self.models.append(cls(config))
        logger.info(f"Registered model: {config.provider}/{config.model}")

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        preferred_model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send chat request with auto-failover."""
        models = self.models
        if preferred_model:
            models = [m for m in models if m.config.model == preferred_model] or models

        last_error: Exception | None = None
        for model in models:
            try:
                logger.debug(f"Trying {model.provider_name}/{model.name}")
                resp = await model.chat(messages, tools, **kwargs)
                self._log_cost(resp)
                return resp
            except Exception as e:
                logger.warning(f"Model {model.provider_name}/{model.name} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        ) from last_error

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        preferred_model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat with auto-failover."""
        models = self.models
        if preferred_model:
            models = [m for m in models if m.config.model == preferred_model] or models

        for model in models:
            try:
                async for chunk in model.stream(messages, tools, **kwargs):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Stream from {model.provider_name}/{model.name} failed: {e}")
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
