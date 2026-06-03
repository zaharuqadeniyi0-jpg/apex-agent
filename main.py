"""
APEX — Main Entry Point
Initializes and runs the full agent system.
"""
from __future__ import annotations

import sys
import asyncio
import logging
import argparse
from pathlib import Path

from apex.config import Config, ModelConfig
from apex.models import ModelRouter
from apex.tools import ToolRegistry
from apex.memory import HybridMemory
from apex.skills import SkillManager
from apex.core.agent import Agent
from apex.core.orchestrator import Orchestrator
from apex.gateway import GatewayManager


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_agent(config: Config) -> Agent:
    """Factory: create a fully wired agent."""
    # Models
    router = ModelRouter([config.model])
    if config.model.fallback:
        for fb in config.model.fallback:
            router.add_model(ModelConfig(provider=fb.get("provider", "ollama"), model=fb.get("model", "llama3")))

    # Tools
    tools = ToolRegistry()

    # Memory
    data_dir = Path(config.memory.persist_dir) if config.memory.persist_dir else Path.home() / ".apex" / "data"
    memory = HybridMemory(config.memory, data_dir)

    return Agent(config=config, model_router=router, tools=tools, memory=memory)


async def main():
    parser = argparse.ArgumentParser(description="APEX — Autonomous Personal EXecutor")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--provider", "-p", help="Model provider")
    parser.add_argument("--web", action="store_true", help="Start web gateway")
    parser.add_argument("--telegram", action="store_true", help="Start Telegram gateway")
    parser.add_argument("--chat", action="store_true", help="Start interactive CLI chat")
    parser.add_argument("--message", help="Send a single message and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Load config
    config = Config.load(args.config)
    if args.debug:
        config.debug = True
        config.log_level = "DEBUG"
    setup_logging(config.log_level)

    if args.model:
        config.model.model = args.model
    if args.provider:
        config.model.provider = args.provider

    # Create agent
    agent = create_agent(config)

    # Single message mode
    if args.message:
        result = await agent.run(args.message)
        print(result.content)
        return

    # Interactive CLI
    if args.chat or (not args.web and not args.telegram):
        from apex.gateway import CLIGateway
        gateway = CLIGateway(agent)
        await gateway.start()
        return

    # Gateway mode
    manager = GatewayManager(config, lambda: create_agent(config))
    if args.web:
        from apex.gateway import WebGateway
        manager.gateways.append(WebGateway(agent, config.gateway.web_port))
    if args.telegram:
        from apex.gateway import TelegramGateway
        if config.gateway.telegram_token:
            manager.gateways.append(TelegramGateway(agent, config.gateway.telegram_token))
        else:
            print("Error: No Telegram token in config")
            sys.exit(1)

    await manager.start_all()


if __name__ == "__main__":
    asyncio.run(main())
