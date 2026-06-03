"""
APEX Gateway — Unified message gateway.
Handles Telegram, Discord, Slack, CLI, and Web API.
"""
from __future__ import annotations

import asyncio
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Awaitable

from apex.config import Config
from apex.core.agent import Agent, AgentResult

logger = logging.getLogger("apex.gateway")


@dataclass
class IncomingMessage:
    platform: str  # telegram, discord, slack, cli, web
    chat_id: str
    user_id: str
    content: str
    message_id: str = ""
    thread_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    chat_id: str
    content: str
    platform: str = "cli"
    thread_id: str = ""
    reply_to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGateway(ABC):
    """Abstract gateway for a messaging platform."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self._message_handler: Callable[[IncomingMessage], Awaitable[OutgoingMessage]] | None = None

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[OutgoingMessage]]):
        self._message_handler = handler

    @abstractmethod
    async def start(self):
        ...

    @abstractmethod
    async def stop(self):
        ...

    @abstractmethod
    async def send(self, message: OutgoingMessage):
        ...

    async def handle_message(self, msg: IncomingMessage) -> OutgoingMessage:
        """Process an incoming message through the agent."""
        session_id = f"{msg.platform}:{msg.chat_id}"

        result = await self.agent.run(
            user_message=msg.content,
            session_id=session_id,
        )

        return OutgoingMessage(
            chat_id=msg.chat_id,
            content=result.content,
            platform=msg.platform,
            reply_to=msg.message_id,
        )


class TelegramGateway(BaseGateway):
    """Telegram bot gateway using python-telegram-bot."""

    def __init__(self, agent: Agent, token: str):
        super().__init__(agent)
        self.token = token

    async def start(self):
        try:
            from telegram import Update
            from telegram.ext import Application, MessageHandler, filters

            async def handler(update: Update, context):
                if update.message and update.message.text:
                    msg = IncomingMessage(
                        platform="telegram",
                        chat_id=str(update.message.chat_id),
                        user_id=str(update.message.from_user.id),
                        content=update.message.text,
                        message_id=str(update.message.message_id),
                    )
                    response = await self.handle_message(msg)
                    await update.message.reply_text(response.content)

            app = Application.builder().token(self.token).build()
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler))
            logger.info("Telegram gateway starting...")
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            self._app = app
        except ImportError:
            logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")

    async def stop(self):
        if hasattr(self, "_app"):
            await self._app.stop()

    async def send(self, message: OutgoingMessage):
        if hasattr(self, "_app"):
            await self._app.bot.send_message(
                chat_id=message.chat_id,
                text=message.content[:4096],
            )


class CLIGateway(BaseGateway):
    """Command-line interface gateway."""

    def __init__(self, agent: Agent):
        super().__init__(agent)
        self._running = False

    async def start(self):
        import sys
        self._running = True
        print("\n🦅 APEX Agent — Type your message (or 'quit' to exit)\n")
        print("=" * 50)

        while self._running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\n[You] > ")
                )
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input.strip():
                    continue

                print("\n[APEX] Thinking...")
                response = await self.handle_message(IncomingMessage(
                    platform="cli",
                    chat_id="cli:main",
                    user_id="user",
                    content=user_input,
                ))
                print(f"\n[APEX] {response.content}")

            except (EOFError, KeyboardInterrupt):
                break

        print("\n🦅 APEX shutting down.")

    async def stop(self):
        self._running = False

    async def send(self, message: OutgoingMessage):
        print(f"\n[APEX → {message.chat_id}] {message.content}")


class WebGateway(BaseGateway):
    """HTTP/WebSocket API gateway."""

    def __init__(self, agent: Agent, port: int = 8080):
        super().__init__(agent)
        self.port = port

    async def start(self):
        try:
            from aiohttp import web

            async def handle_chat(request: web.Request) -> web.Response:
                data = await request.json()
                msg = IncomingMessage(
                    platform="web",
                    chat_id=data.get("chat_id", "default"),
                    user_id=data.get("user_id", "anonymous"),
                    content=data.get("message", ""),
                )
                response = await self.handle_message(msg)
                return web.json_response({
                    "response": response.content,
                    "platform": "web",
                })

            async def handle_health(request: web.Request) -> web.Response:
                return web.json_response({"status": "ok", "agent": "APEX"})

            app = web.Application()
            app.router.add_post("/api/chat", handle_chat)
            app.router.add_get("/api/health", handle_health)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()
            logger.info(f"Web gateway running on port {self.port}")
            self._runner = runner
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")

    async def stop(self):
        if hasattr(self, "_runner"):
            await self._runner.cleanup()

    async def send(self, message: OutgoingMessage):
        logger.info(f"Web send to {message.chat_id}: {message.content[:100]}")


class GatewayManager:
    """Manages all gateway instances."""

    def __init__(self, config: Config, agent_factory: Callable[[], Agent]):
        self.config = config
        self.agent_factory = agent_factory
        self.gateways: list[BaseGateway] = []

        # Initialize configured gateways
        if config.gateway.telegram_token:
            self.gateways.append(
                TelegramGateway(self.agent_factory(), config.gateway.telegram_token)
            )
        if config.gateway.web_port:
            self.gateways.append(
                WebGateway(self.agent_factory(), config.gateway.web_port)
            )

    def add_cli(self):
        self.gateways.append(CLIGateway(self.agent_factory()))

    async def start_all(self):
        await asyncio.gather(*(g.start() for g in self.gateways))

    async def stop_all(self):
        await asyncio.gather(*(g.stop() for g in self.gateways))
