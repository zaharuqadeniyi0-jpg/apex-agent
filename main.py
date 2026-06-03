"""
APEX v2.0 — Main Entry Point
Initializes and runs the full agent system.
"""
from __future__ import annotations

import sys
import asyncio
import logging
import argparse
from pathlib import Path

from apex.config import Config
from apex.models import ModelRouter
from apex.tools import ToolRegistry
from apex.memory import MemoryManager
from apex.skills import SkillManager
from apex.core.agent import Agent
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
    router = ModelRouter(config.model)

    # Tools
    tools = ToolRegistry()

    # Memory
    data_dir = Path(config.memory.tree.persist_dir) if config.memory.tree.persist_dir else (
        Path.home() / ".apex" / "data"
    )
    memory = MemoryManager(config.memory, data_dir)

    # Skills
    skill_dirs = [Path(d).expanduser() for d in config.skills.dirs] if config.skills.dirs else [
        Path.home() / ".apex" / "skills",
    ]
    skills = SkillManager(dirs=skill_dirs, clawhub_enabled=config.skills.clawhub_enabled)
    skills.discover()

    return Agent(
        config=config, model_router=router, tools=tools,
        memory=memory, skills=skills,
    )


async def main():
    parser = argparse.ArgumentParser(description="APEX v2.0 — Autonomous Personal EXecutor")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--model", "-m", help="Model profile name")
    parser.add_argument("--agent", "-a", help="Agent definition name")
    parser.add_argument("--web", action="store_true", help="Start web API")
    parser.add_argument("--telegram", action="store_true", help="Start Telegram")
    parser.add_argument("--discord", action="store_true", help="Start Discord")
    parser.add_argument("--chat", action="store_true", help="Interactive CLI")
    parser.add_argument("--message", help="Single message mode")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.debug:
        config.debug = True
        config.log_level = "DEBUG"
    if args.model:
        config.model.primary = args.model

    setup_logging(config.log_level)
    agent = create_agent(config)

    if args.message:
        result = await agent.run(args.message, agent_name=args.agent)
        print(result.content)
        return

    if args.chat or (not args.web and not args.telegram and not args.discord):
        from apex.gateway import CLIChannel
        channel = CLIChannel(
            type("Cfg", (), {"platform": "cli", "token": "", "bot_token": "",
                              "app_token": "", "webhook_url": "", "webhook_secret": "",
                              "allow_from": [], "dm_policy": "open", "group_policy": "mention",
                              "streaming": True, "max_message_length": 4096, "extra": {}})(),
            agent,
        )
        await channel.start()
        return

    manager = GatewayManager(config, lambda: create_agent(config))
    await manager.start_all()


if __name__ == "__main__":
    asyncio.run(main())
