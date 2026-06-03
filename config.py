"""
APEX Configuration
Central config loaded from ~/.apex/config.yaml
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


APEX_HOME = Path(os.environ.get("APEX_HOME", Path.home() / ".apex"))
CONFIG_PATH = APEX_HOME / "config.yaml"


@dataclass
class ModelConfig:
    provider: str = "ollama"
    model: str = "llama3"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    fallback: list[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    sandbox: bool = True
    timeout: int = 300
    max_output: int = 50_000
    allowed_commands: list[str] = field(default_factory=lambda: ["*"])
    blocked_commands: list[str] = field(default_factory=lambda: ["rm -rf /", "mkfs", "dd"])


@dataclass
class MemoryConfig:
    backend: str = "chroma"  # chroma, sqlite, hybrid
    persist_dir: str = ""
    auto_summarize: bool = True
    max_session_msgs: int = 100


@dataclass
class GatewayConfig:
    telegram_token: str = ""
    discord_token: str = ""
    slack_token: str = ""
    web_port: int = 8080


@dataclass
class Config:
    version: str = "0.1.0"
    name: str = "APEX"
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    skills_dir: str = ""
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load config from YAML file, falling back to defaults."""
        cfg_path = Path(path) if path else CONFIG_PATH
        if not cfg_path.exists():
            config = cls()
            config._create_default(cfg_path)
            return config

        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}

        model_raw = raw.pop("model", {})
        tools_raw = raw.pop("tools", {})
        memory_raw = raw.pop("memory", {})
        gateway_raw = raw.pop("gateway", {})

        return cls(
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
            model=ModelConfig(**model_raw) if model_raw else ModelConfig(),
            tools=ToolConfig(**tools_raw) if tools_raw else ToolConfig(),
            memory=MemoryConfig(**memory_raw) if memory_raw else MemoryConfig(),
            gateway=GatewayConfig(**gateway_raw) if gateway_raw else GatewayConfig(),
        )

    def _create_default(self, path: Path):
        """Create default config file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "name": "APEX",
            "model": {
                "provider": "ollama",
                "model": "llama3",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "tools": {
                "sandbox": True,
                "timeout": 300,
            },
            "memory": {
                "backend": "hybrid",
                "auto_summarize": True,
            },
            "gateway": {
                "web_port": 8080,
            },
            "log_level": "INFO",
        }
        with open(path, "w") as f:
            yaml.dump(default, f, default_flow_style=False)
        print(f"[APEX] Created default config at {path}")

    def save(self, path: str | Path | None = None):
        """Save current config to YAML file."""
        import dataclasses
        cfg_path = Path(path) if path else CONFIG_PATH
        with open(cfg_path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
