# 🦅 APEX — Autonomous Personal EXecutor

A powerful AI agent framework built from scratch.

## Architecture

```
apex/
├── core/        # Agent loop + multi-agent orchestrator
├── models/      # Multi-model abstraction (Ollama, OpenAI, Anthropic)
├── tools/       # Shell, file, web, code execution tools
├── memory/      # Hybrid: SQLite + ChromaDB vector memory
├── skills/      # Versioned, dependency-aware skill system
├── gateway/     # Telegram, CLI, Web API gateways
├── config.py    # YAML configuration
└── main.py      # Entry point
```

## Features

- **Async-first** — asyncio throughout
- **Multi-model** — Ollama, OpenAI with auto-failover
- **7 built-in tools** — shell, file R/W/patch, web search/extract, code exec
- **Hybrid memory** — SQLite structured + ChromaDB semantic search
- **DAG orchestrator** — auto-decompose goals into parallel subagent tasks
- **Skill system** — YAML-frontmatter skills with categories
- **29 tests** — all passing

## Quick Start

```bash
pip install -e ".[dev]"
apex --chat
```

## License
MIT
