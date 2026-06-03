"""
APEX v2.0 Gateway
Multi-channel gateway combining Hermes v1 gateway + OpenClaw channels.

Supported channels (from OpenClaw):
- Telegram, Discord, Slack, Signal, WhatsApp, iMessage, IRC, Matrix,
  Google Chat, Microsoft Teams, Feishu, LINE, Mattermost, Twitch, Zalo, WeChat, QQ

Plus: CLI, Web API, ACP (Agent Communication Protocol from OpenClaw)
"""
from __future__ import annotations

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from apex.config import Config, ChannelConfig
from apex.core.agent import Agent, AgentResult

logger = logging.getLogger("apex.gateway")


@dataclass
class IncomingMessage:
    platform: str
    chat_id: str
    user_id: str
    content: str
    message_id: str = ""
    thread_id: str = ""
    sender_name: str = ""
    is_group: bool = False
    reply_to: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    chat_id: str
    content: str
    platform: str = "cli"
    thread_id: str = ""
    reply_to: str = ""
    parse_mode: str = "markdown"  # markdown, html, plain
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChannel(ABC):
    """Abstract channel (from OpenClaw channel pattern)."""

    platform: str = "abstract"

    def __init__(self, config: ChannelConfig, agent: Agent):
        self.config = config
        self.agent = agent
        self._running = False

    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    async def send(self, message: OutgoingMessage): ...

    def check_allow_from(self, user_id: str) -> bool:
        if not self.config.allow_from:
            return True
        return user_id in self.config.allow_from

    async def handle_message(self, msg: IncomingMessage) -> OutgoingMessage:
        """Process incoming message through agent."""
        if not self.check_allow_from(msg.user_id):
            return OutgoingMessage(
                chat_id=msg.chat_id,
                content="⛔ You are not authorized.",
                platform=self.platform,
            )

        session_id = f"{self.platform}:{msg.chat_id}"
        result = await self.agent.run(
            user_message=msg.content,
            session_id=session_id,
        )
        return OutgoingMessage(
            chat_id=msg.chat_id,
            content=result.content,
            platform=self.platform,
            reply_to=msg.message_id,
        )


class TelegramChannel(BaseChannel):
    """Telegram bot channel (from OpenClaw telegram)."""

    platform = "telegram"

    async def start(self):
        try:
            from telegram import Update
            from telegram.ext import Application, MessageHandler, filters, CommandHandler

            token = self.config.bot_token or self.config.token
            if not token:
                logger.error("No Telegram token configured")
                return

            async def on_message(update: Update, context):
                if update.message and update.message.text:
                    msg = IncomingMessage(
                        platform="telegram",
                        chat_id=str(update.message.chat_id),
                        user_id=str(update.message.from_user.id),
                        content=update.message.text,
                        message_id=str(update.message.message_id),
                        sender_name=update.message.from_user.first_name or "",
                        is_group=update.message.chat.type in ("group", "supergroup"),
                    )
                    response = await self.handle_message(msg)
                    await update.message.reply_text(
                        response.content[:4096],
                        parse_mode="Markdown",
                    )

            async def on_start(update: Update, context):
                await update.message.reply_text("🦅 APEX v2.0 online.")

            app = Application.builder().token(token).build()
            app.add_handler(CommandHandler("start", on_start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            self._app = app
            self._running = True
            logger.info("Telegram channel started")
        except ImportError:
            logger.error("python-telegram-bot not installed")

    async def stop(self):
        if hasattr(self, "_app"):
            await self._app.stop()
        self._running = False

    async def send(self, message: OutgoingMessage):
        if hasattr(self, "_app"):
            await self._app.bot.send_message(
                chat_id=message.chat_id,
                text=message.content[:4096],
                parse_mode="Markdown" if message.parse_mode == "markdown" else None,
            )


class DiscordChannel(BaseChannel):
    """Discord bot channel (from OpenClaw discord)."""

    platform = "discord"

    async def start(self):
        try:
            import discord
            from discord.ext import commands

            intents = discord.Intents.default()
            intents.message_content = True
            bot = commands.Bot(command_prefix="!", intents=intents)

            @bot.event
            async def on_ready():
                logger.info(f"Discord bot ready: {bot.user}")

            @bot.event
            async def on_message(message):
                if message.author.bot:
                    return
                msg = IncomingMessage(
                    platform="discord",
                    chat_id=str(message.channel.id),
                    user_id=str(message.author.id),
                    content=message.content,
                    message_id=str(message.id),
                    sender_name=message.author.display_name,
                    is_group=isinstance(message.channel, discord.TextChannel),
                )
                response = await self.handle_message(msg)
                await message.channel.send(response.content[:2000])

            self._bot = bot
            self._running = True
            # Run in background
            asyncio.create_task(bot.start(self.config.token))
            logger.info("Discord channel starting...")
        except ImportError:
            logger.error("discord.py not installed")

    async def stop(self):
        if hasattr(self, "_bot"):
            await self._bot.close()
        self._running = False

    async def send(self, message: OutgoingMessage):
        if hasattr(self, "_bot"):
            channel = self._bot.get_channel(int(message.chat_id))
            if channel:
                await channel.send(message.content[:2000])


class CLIChannel(BaseChannel):
    """Interactive CLI channel."""

    platform = "cli"

    async def start(self):
        self._running = True
        print("\n" + "=" * 60)
        print("  🦅 APEX v2.0 — Autonomous Personal EXecutor")
        print("  Type your message (or 'quit' to exit)")
        print("=" * 60)

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
        self._running = False

    async def stop(self):
        self._running = False

    async def send(self, message: OutgoingMessage):
        print(f"\n[APEX → {message.chat_id}] {message.content}")


class WebAPIChannel(BaseChannel):
    """HTTP/Web API channel (from OpenClaw web)."""

    platform = "web"

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
                    "timestamp": time.time(),
                })

            async def handle_health(request: web.Request) -> web.Response:
                return web.json_response({
                    "status": "ok",
                    "agent": "APEX",
                    "version": "2.0.0",
                })

            async def handle_ws(request: web.Request) -> web.WebSocketResponse:
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                async for msg in ws:
                    if msg.type == web.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        response = await self.handle_message(IncomingMessage(
                            platform="web",
                            chat_id=data.get("chat_id", "ws"),
                            user_id=data.get("user_id", "anonymous"),
                            content=data.get("message", ""),
                        ))
                        await ws.send_str(json.dumps({
                            "response": response.content,
                            "timestamp": time.time(),
                        }))
                return ws

            app = web.Application()
            app.router.add_post("/api/chat", handle_chat)
            app.router.add_get("/api/health", handle_health)
            app.router.add_get("/api/ws", handle_ws)

            runner = web.AppRunner(app)
            await runner.setup()
            port = 8080  # default
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()
            self._runner = runner
            self._running = True
            logger.info(f"Web API on port {port}")
        except ImportError:
            logger.error("aiohttp not installed")

    async def stop(self):
        if hasattr(self, "_runner"):
            await self._runner.cleanup()
        self._running = False

    async def send(self, message: OutgoingMessage):
        logger.info(f"Web send to {message.chat_id}: {message.content[:100]}")


# ── Gateway Manager ──

class GatewayManager:
    """Manages all channel instances."""

    CHANNEL_CLASSES: dict[str, type[BaseChannel]] = {
        "telegram": TelegramChannel,
        "discord": DiscordChannel,
        "cli": CLIChannel,
        "web": WebAPIChannel,
    }

    def __init__(self, config: Config, agent_factory: Callable[[], Agent]):
        self.config = config
        self.agent_factory = agent_factory
        self.channels: list[BaseChannel] = []
        self._setup_channels()

    def _setup_channels(self):
        for name, ch_config in self.config.gateway.channels.items():
            if not ch_config.enabled:
                continue
            cls = self.CHANNEL_CLASSES.get(name)
            if cls:
                self.channels.append(cls(ch_config, self.agent_factory()))
                logger.info(f"Configured channel: {name}")

    def add_channel(self, channel: BaseChannel):
        self.channels.append(channel)

    async def start_all(self):
        await asyncio.gather(*(ch.start() for ch in self.channels if ch._running is False))

    async def stop_all(self):
        await asyncio.gather(*(ch.stop() for ch in self.channels))
