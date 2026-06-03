"""
APEX Memory System
Multi-layered memory: session, persistent (SQLite), vector (ChromaDB), and graph.
"""
from __future__ import annotations

import json
import time
import sqlite3
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex.config import MemoryConfig

logger = logging.getLogger("apex.memory")


@dataclass
class MemoryEntry:
    key: str
    content: str
    category: str = "general"  # general, user_pref, env_fact, skill_note
    importance: float = 0.5  # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    async def retrieve(self, key: str) -> MemoryEntry | None: ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]: ...

    @abstractmethod
    async def list_all(self, category: str | None = None) -> list[MemoryEntry]: ...

    @abstractmethod
    async def delete(self, key: str) -> bool: ...

    @abstractmethod
    async def clear(self) -> None: ...


class SQLiteMemory(MemoryBackend):
    """Structured memory in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT DEFAULT '[]',
                    summary TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    async def store(self, entry: MemoryEntry) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (key, content, category, importance, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.key, entry.content, entry.category,
                    entry.importance, entry.created_at, entry.updated_at,
                    json.dumps(entry.metadata),
                ),
            )
            conn.commit()

    async def retrieve(self, key: str) -> MemoryEntry | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT key, content, category, importance, created_at, updated_at, metadata FROM memories WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return MemoryEntry(
            key=row[0], content=row[1], category=row[2],
            importance=row[3], created_at=row[4], updated_at=row[5],
            metadata=json.loads(row[6]),
        )

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        with sqlite3.connect(self.db_path) as conn:
            # Simple FTS-like search
            rows = conn.execute(
                """SELECT key, content, category, importance, created_at, updated_at, metadata
                   FROM memories WHERE content LIKE ?
                   ORDER BY importance DESC LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
        return [
            MemoryEntry(
                key=r[0], content=r[1], category=r[2],
                importance=r[3], created_at=r[4], updated_at=r[5],
                metadata=json.loads(r[6]),
            )
            for r in rows
        ]

    async def list_all(self, category: str | None = None) -> list[MemoryEntry]:
        with sqlite3.connect(self.db_path) as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY updated_at DESC",
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC"
                ).fetchall()
        return [
            MemoryEntry(
                key=r[0], content=r[1], category=r[2],
                importance=r[3], created_at=r[4], updated_at=r[5],
                metadata=json.loads(r[6]),
            )
            for r in rows
        ]

    async def delete(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    async def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories")
            conn.commit()

    # Session management
    async def save_session(self, session_id: str, messages: list[dict], summary: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, messages, summary, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, json.dumps(messages), summary, time.time(), time.time()),
            )
            conn.commit()

    async def load_session(self, session_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT messages, summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"messages": json.loads(row[0]), "summary": row[1]}

    async def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, summary, messages FROM sessions
                   WHERE summary LIKE ? OR messages LIKE ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [
            {"session_id": r[0], "summary": r[1], "message_count": len(json.loads(r[2]))}
            for r in rows
        ]


class VectorMemory(MemoryBackend):
    """Semantic vector memory using ChromaDB."""

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._available = False
        try:
            import chromadb
            self._client = chromadb.Client(
                chromadb.config.Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.persist_dir),
                    anonymized_telemetry=False,
                )
            )
            self._collection = self._client.get_or_create_collection(
                name="apex_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info("ChromaDB vector memory initialized")
        except ImportError:
            logger.warning("chromadb not available, vector memory disabled")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")

    async def store(self, entry: MemoryEntry) -> None:
        if not self._available:
            return
        self._collection.upsert(
            ids=[entry.key],
            documents=[entry.content],
            metadatas=[{
                "category": entry.category,
                "importance": entry.importance,
                "created_at": entry.created_at,
            }],
        )

    async def retrieve(self, key: str) -> MemoryEntry | None:
        if not self._available:
            return None
        result = self._collection.get(ids=[key])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0] if result["metadatas"] else {}
        return MemoryEntry(
            key=result["ids"][0],
            content=result["documents"][0],
            category=meta.get("category", "general"),
            importance=meta.get("importance", 0.5),
            created_at=meta.get("created_at", time.time()),
        )

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        if not self._available:
            return []
        results = self._collection.query(query_texts=[query], n_results=limit)
        entries = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            metas = results.get("metadatas", [[]])[0]
            meta = metas[i] if i < len(metas) else {}
            ids = results.get("ids", [[]])[0]
            entries.append(MemoryEntry(
                key=ids[i] if i < len(ids) else f"vec_{i}",
                content=doc,
                category=meta.get("category", "general"),
                importance=meta.get("importance", 0.5),
                created_at=meta.get("created_at", time.time()),
            ))
        return entries

    async def list_all(self, category: str | None = None) -> list[MemoryEntry]:
        if not self._available:
            return []
        results = self._collection.get()
        entries = []
        for i, doc in enumerate(results.get("documents", [])):
            ids = results.get("ids", [])
            metas = results.get("metadatas", [])
            meta = metas[i] if i < len(metas) else {}
            if category and meta.get("category") != category:
                continue
            entries.append(MemoryEntry(
                key=ids[i] if i < len(ids) else f"vec_{i}",
                content=doc,
                category=meta.get("category", "general"),
                importance=meta.get("importance", 0.5),
            ))
        return entries

    async def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            self._collection.delete(ids=[key])
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        if not self._available:
            return
        self._client.delete_collection("apex_memory")
        self._collection = self._client.get_or_create_collection(
            name="apex_memory",
            metadata={"hnsw:space": "cosine"},
        )


class HybridMemory:
    """
    Combines SQLite (structured) + ChromaDB (semantic) memory.
    Falls back gracefully if ChromaDB is unavailable.
    """

    def __init__(self, config: MemoryConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        sqlite_path = data_dir / "memory.db"
        vector_dir = data_dir / "vector"

        self.structured = SQLiteMemory(str(sqlite_path))
        self.vector = VectorMemory(str(vector_dir))
        self._session_cache: dict[str, list[dict]] = {}

    async def remember(
        self,
        key: str,
        content: str,
        category: str = "general",
        importance: float = 0.5,
    ):
        """Store a memory in both backends."""
        entry = MemoryEntry(
            key=key,
            content=content,
            category=category,
            importance=importance,
        )
        await self.structured.store(entry)
        await self.vector.store(entry)

    async def recall(self, key: str) -> str:
        """Retrieve a memory by key."""
        entry = await self.structured.retrieve(key)
        if entry:
            return entry.content
        entry = await self.vector.retrieve(key)
        if entry:
            return entry.content
        return ""

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search memories using both backends."""
        # Combine results from both
        structured = await self.structured.search(query, limit=limit)
        vector = await self.vector.search(query, limit=limit)

        # Deduplicate by key
        seen = {e.key for e in structured}
        combined = list(structured)
        for e in vector:
            if e.key not in seen:
                combined.append(e)
        return combined[:limit]

    async def forget(self, key: str) -> bool:
        """Delete a memory."""
        r1 = await self.structured.delete(key)
        r2 = await self.vector.delete(key)
        return r1 or r2

    async def list_all(self, category: str | None = None) -> list[MemoryEntry]:
        return await self.structured.list_all(category)

    # Session management (delegated to SQLite backend)
    async def save_session(self, session_id: str, messages: list[dict], summary: str = ""):
        await self.structured.save_session(session_id, messages, summary)

    async def load_session(self, session_id: str) -> dict | None:
        return await self.structured.load_session(session_id)

    async def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        return await self.structured.search_sessions(query, limit)
