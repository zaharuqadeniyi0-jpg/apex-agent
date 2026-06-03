"""
APEX v2.0 Memory System
Combines Hermes key-value + OpenHuman memory tree + OpenClaw SQLite state.

Memory layers:
1. Session transcript (messages in current conversation)
2. SQLite state DB (structured facts, preferences, environment)
3. Vector DB (semantic search via ChromaDB)
4. Memory tree (from OpenHuman): hierarchical summary tree
   - Source trees: conversations, files, web, integrations
   - Entity index: extracted people, projects, concepts
   - Automatic summarization and entity extraction
5. Markdown vault (from OpenHuman): Obsidian-compatible markdown files
"""
from __future__ import annotations

import json
import time
import sqlite3
import logging
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex.config import MemoryConfig, MemoryTreeConfig

logger = logging.getLogger("apex.memory")


# ── Core Types ──

@dataclass
class MemoryEntry:
    key: str
    content: str
    category: str = "general"
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "conversation"  # conversation, file, web, integration
    entities: list[str] = field(default_factory=list)  # extracted entity names


@dataclass
class Entity:
    """Extracted entity from memory content (from OpenHuman memory_entities)."""
    name: str
    type: str  # person, project, tool, concept, location, org
    mentions: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    relations: dict[str, list[str]] = field(default_factory=dict)  # relation_type -> [entity_names]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryLeaf:
    """A leaf node in the memory tree (from OpenHuman memory_tree)."""
    id: str
    source: str          # conversations, files, web, whatsapp, etc.
    content: str
    timestamp: float = field(default_factory=time.time)
    embedding: list[float] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryBucket:
    """A bucket (branch node) in the memory tree containing leaves + summary."""
    id: str
    source: str
    label: str
    leaves: list[str] = field(default_factory=list)    # leaf IDs
    children: list[str] = field(default_factory=list)  # child bucket IDs
    summary: str = ""
    last_summarized: float = 0.0
    token_count: int = 0


# ── SQLite Backend (from OpenClaw state DB pattern) ──

class SQLiteBackend:
    """
    Structured state storage using SQLite (from OpenClaw state/openclaw.sqlite pattern).
    Stores: memories, entities, sessions, plugin KV, agent state.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance REAL DEFAULT 0.5,
                    source TEXT DEFAULT 'conversation',
                    entities TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    mentions INTEGER DEFAULT 1,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    relations TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}',
                    PRIMARY KEY (name, type)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_name TEXT DEFAULT 'default',
                    messages TEXT DEFAULT '[]',
                    summary TEXT DEFAULT '',
                    token_usage TEXT DEFAULT '{"prompt":0,"completion":0}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plugin_kv (
                    plugin_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (plugin_id, key)
                );

                CREATE TABLE IF NOT EXISTS memory_leaves (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    embedding TEXT DEFAULT '[]',
                    entities TEXT DEFAULT '[]',
                    importance REAL DEFAULT 0.5,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS memory_buckets (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    label TEXT NOT NULL,
                    leaves TEXT DEFAULT '[]',
                    children TEXT DEFAULT '[]',
                    summary TEXT DEFAULT '',
                    last_summarized REAL DEFAULT 0,
                    token_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS agent_state (
                    agent_id TEXT PRIMARY KEY,
                    state TEXT DEFAULT '{}',
                    updated_at REAL NOT NULL
                );

                -- FTS5 for full-text search across memories
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    key, content, category,
                    content='memories',
                    content_rowid='rowid'
                );

                -- Triggers to keep FTS index in sync
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, key, content, category)
                    VALUES (new.rowid, new.key, new.content, new.category);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, content, category)
                    VALUES ('delete', old.rowid, old.key, old.content, old.category);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, content, category)
                    VALUES ('delete', old.rowid, old.key, old.content, old.category);
                    INSERT INTO memories_fts(rowid, key, content, category)
                    VALUES (new.rowid, new.key, new.content, new.category);
                END;
            """)
            conn.commit()

    # ── Memories CRUD ──

    def store_memory(self, entry: MemoryEntry):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (key, content, category, importance, source, entities,
                    created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.key, entry.content, entry.category,
                    entry.importance, entry.source,
                    json.dumps(entry.entities),
                    entry.created_at, entry.updated_at,
                    json.dumps(entry.metadata),
                ),
            )
            conn.commit()

    def retrieve_memory(self, key: str) -> MemoryEntry | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        return MemoryEntry(
            key=row[0], content=row[1], category=row[2],
            importance=row[3], source=row[4],
            entities=json.loads(row[5]),
            created_at=row[6], updated_at=row[7],
            metadata=json.loads(row[8]),
        )

    def search_memories(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Full-text search via FTS5."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                rows = conn.execute(
                    """SELECT m.key, m.content, m.category, m.importance,
                              m.source, m.entities, m.created_at, m.updated_at, m.metadata
                       FROM memories_fts f
                       JOIN memories m ON m.rowid = f.rowid
                       WHERE memories_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, limit),
                ).fetchall()
            except Exception:
                # FTS5 query may fail on special chars, fall back to LIKE
                rows = conn.execute(
                    "SELECT * FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
        return [
            MemoryEntry(
                key=r[0], content=r[1], category=r[2],
                importance=r[3], source=r[4], entities=json.loads(r[5]),
                created_at=r[6], updated_at=r[7], metadata=json.loads(r[8]),
            )
            for r in rows
        ]

    def list_memories(self, category: str | None = None, limit: int = 100) -> list[MemoryEntry]:
        with sqlite3.connect(self.db_path) as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [MemoryEntry(
            key=r[0], content=r[1], category=r[2], importance=r[3],
            source=r[4], entities=json.loads(r[5]),
            created_at=r[6], updated_at=r[7], metadata=json.loads(r[8]),
        ) for r in rows]

    def delete_memory(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    # ── Entity CRUD ──

    def store_entity(self, entity: Entity):
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT mentions, relations FROM entities WHERE name = ? AND type = ?",
                (entity.name, entity.type),
            ).fetchone()
            if existing:
                entity.mentions += existing[0]
                old_relations = json.loads(existing[1])
                for k, v in entity.relations.items():
                    old_relations.setdefault(k, []).extend(v)
                entity.relations = old_relations
            conn.execute(
                """INSERT OR REPLACE INTO entities
                   (name, type, mentions, first_seen, last_seen, relations, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity.name, entity.type, entity.mentions,
                    entity.first_seen, entity.last_seen,
                    json.dumps(entity.relations), json.dumps(entity.metadata),
                ),
            )
            conn.commit()

    def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? ORDER BY mentions DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [Entity(
            name=r[0], type=r[1], mentions=r[2],
            first_seen=r[3], last_seen=r[4],
            relations=json.loads(r[5]), metadata=json.loads(r[6]),
        ) for r in rows]

    def get_entity(self, name: str, type: str = "") -> Entity | None:
        with sqlite3.connect(self.db_path) as conn:
            if type:
                row = conn.execute(
                    "SELECT * FROM entities WHERE name = ? AND type = ?", (name, type)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM entities WHERE name = ? ORDER BY mentions DESC LIMIT 1", (name,)
                ).fetchone()
        if not row:
            return None
        return Entity(
            name=r[0], type=r[1], mentions=r[2],
            first_seen=r[3], last_seen=r[4],
            relations=json.loads(r[5]), metadata=json.loads(r[6]),
        )

    # ── Session CRUD ──

    def save_session(self, session_id: str, messages: list[dict], summary: str = "",
                     token_usage: dict[str, int] | None = None, agent_name: str = "default"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, agent_name, messages, summary, token_usage, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, agent_name,
                    json.dumps(messages), summary,
                    json.dumps(token_usage or {"prompt": 0, "completion": 0}),
                    time.time(), time.time(),
                ),
            )
            conn.commit()

    def load_session(self, session_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT messages, summary, token_usage, agent_name FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "messages": json.loads(row[0]),
            "summary": row[1],
            "token_usage": json.loads(row[2]),
            "agent_name": row[3],
        }

    def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT s.session_id, s.agent_name, s.summary, s.messages
                   FROM sessions s
                   WHERE s.summary LIKE ? OR s.messages LIKE ?
                   ORDER BY s.updated_at DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [
            {"session_id": r[0], "agent_name": r[1], "summary": r[2],
             "message_count": len(json.loads(r[3]))}
            for r in rows
        ]

    # ── Memory tree leaves/buckets ──

    def store_leaf(self, leaf: MemoryLeaf):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memory_leaves
                   (id, source, content, timestamp, embedding, entities, importance, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    leaf.id, leaf.source, leaf.content, leaf.timestamp,
                    json.dumps(leaf.embedding), json.dumps(leaf.entities),
                    leaf.importance, json.dumps(leaf.metadata),
                ),
            )
            conn.commit()

    def get_leaves_by_source(self, source: str, limit: int = 50) -> list[MemoryLeaf]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM memory_leaves WHERE source = ? ORDER BY timestamp DESC LIMIT ?",
                (source, limit),
            ).fetchall()
        return [MemoryLeaf(
            id=r[0], source=r[1], content=r[2], timestamp=r[3],
            embedding=json.loads(r[4]), entities=json.loads(r[5]),
            importance=r[6], metadata=json.loads(r[7]),
        ) for r in rows]

    def store_bucket(self, bucket: MemoryBucket):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memory_buckets
                   (id, source, label, leaves, children, summary, last_summarized, token_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    bucket.id, bucket.source, bucket.label,
                    json.dumps(bucket.leaves), json.dumps(bucket.children),
                    bucket.summary, bucket.last_summarized, bucket.token_count,
                ),
            )
            conn.commit()

    def get_bucket(self, bucket_id: str) -> MemoryBucket | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memory_buckets WHERE id = ?", (bucket_id,)
            ).fetchone()
        if not row:
            return None
        return MemoryBucket(
            id=r[0], source=r[1], label=r[2],
            leaves=json.loads(r[3]), children=json.loads(r[4]),
            summary=r[5], last_summarized=r[6], token_count=r[7],
        )

    # ── Plugin KV ──

    def set_plugin_kv(self, plugin_id: str, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO plugin_kv (plugin_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (plugin_id, key, value, time.time()),
            )
            conn.commit()

    def get_plugin_kv(self, plugin_id: str, key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM plugin_kv WHERE plugin_id = ? AND key = ?",
                (plugin_id, key),
            ).fetchone()
        return row[0] if row else None

    # ── Agent state ──

    def save_agent_state(self, agent_id: str, state: dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO agent_state (agent_id, state, updated_at)
                   VALUES (?, ?, ?)""",
                (agent_id, json.dumps(state), time.time()),
            )
            conn.commit()

    def load_agent_state(self, agent_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state FROM agent_state WHERE agent_id = ?", (agent_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None


# ── Vector Backend (ChromaDB) ──

class VectorBackend:
    """Semantic vector search (from Hermes v1, kept for compatibility)."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
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
                name="apex_memory_v2",
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info("ChromaDB vector memory initialized")
        except ImportError:
            logger.info("chromadb not available, vector search disabled")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def upsert(self, id: str, content: str, metadata: dict[str, Any] | None = None):
        if not self._available:
            return
        self._collection.upsert(
            ids=[id], documents=[content],
            metadatas=[metadata or {}],
        )

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self._available:
            return []
        results = self._collection.query(query_texts=[query], n_results=limit)
        items = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            ids = results.get("ids", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            items.append({
                "id": ids[i] if i < len(ids) else f"vec_{i}",
                "content": doc,
                "metadata": metas[i] if i < len(metas) else {},
            })
        return items

    def delete(self, id: str):
        if not self._available:
            return
        try:
            self._collection.delete(ids=[id])
        except Exception:
            pass


# ── Markdown Vault (from OpenHuman) ──

class MarkdownVault:
    """
    Obsidian-compatible markdown vault for persistent memory.
    Stores memories as markdown files with YAML frontmatter.
    """

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.vault_path.mkdir(parents=True, exist_ok=True)
        (self.vault_path / "entities").mkdir(exist_ok=True)
        (self.vault_path / "conversations").mkdir(exist_ok=True)
        (self.vault_path / "daily").mkdir(exist_ok=True)

    def write_memory(self, entry: MemoryEntry):
        """Write a memory as a markdown file."""
        safe_name = re.sub(r'[^\w\-.]', '_', entry.key)[:80]
        path = self.vault_path / f"{safe_name}.md"
        frontmatter = {
            "key": entry.key,
            "category": entry.category,
            "importance": entry.importance,
            "source": entry.source,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(entry.created_at)),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(entry.updated_at)),
            "entities": entry.entities,
        }
        yaml_header = yaml.dump(frontmatter, default_flow_style=False)
        content = f"---\n{yaml_header}---\n\n{entry.content}\n"
        path.write_text(content, encoding="utf-8")

    def write_entity(self, entity: Entity):
        """Write an entity as a markdown file (Obsidian-style)."""
        safe_name = re.sub(r'[^\w\-.]', '_', entity.name)[:80]
        path = self.vault_path / "entities" / f"{safe_name}.md"
        frontmatter = {
            "name": entity.name,
            "type": entity.type,
            "mentions": entity.mentions,
            "first_seen": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(entity.first_seen)),
            "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(entity.last_seen)),
        }
        yaml_header = yaml.dump(frontmatter, default_flow_style=False)
        relations_md = "\n".join(
            f"- **{k}**: {', '.join(v)}"
            for k, v in entity.relations.items()
        )
        content = f"---\n{yaml_header}---\n\n## Relations\n{relations_md}\n"
        path.write_text(content, encoding="utf-8")

    def write_daily_note(self, date: str, content: str):
        """Write a daily note (from OpenHuman daily pattern)."""
        path = self.vault_path / "daily" / f"{date}.md"
        path.write_text(f"# {date}\n\n{content}\n", encoding="utf-8")

    def search_vault(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """Search vault markdown files."""
        results = []
        for md_file in self.vault_path.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8", errors="replace")
            if query.lower() in text.lower():
                results.append({
                    "file": str(md_file.relative_to(self.vault_path)),
                    "content": text[:500],
                })
                if len(results) >= limit:
                    break
        return results


# ── Unified Memory Manager ──

class MemoryManager:
    """
    Unified memory system combining all backends.
    API matches what the agent loop expects.
    """

    def __init__(self, config: MemoryConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir

        # SQLite (always available)
        db_path = data_dir / "state" / "apex.sqlite"
        self.sqlite = SQLiteBackend(db_path)

        # Vector (optional)
        vector_dir = data_dir / "vector"
        self.vector = VectorBackend(vector_dir)

        # Markdown vault (optional, from OpenHuman)
        vault_path = Path(config.tree.vault_path) if config.tree.vault_path else (data_dir / "vault")
        self.vault = MarkdownVault(vault_path) if config.tree.enabled else None

    async def remember(self, key: str, content: str, category: str = "general",
                       importance: float = 0.5, source: str = "conversation",
                       entities: list[str] | None = None):
        """Store a memory across all backends."""
        entry = MemoryEntry(
            key=key, content=content, category=category,
            importance=importance, source=source,
            entities=entities or [],
        )
        self.sqlite.store_memory(entry)
        self.vector.upsert(key, content, {"category": category, "importance": importance})
        if self.vault:
            self.vault.write_memory(entry)

    async def recall(self, key: str) -> str:
        """Retrieve a memory by key."""
        entry = self.sqlite.retrieve_memory(key)
        return entry.content if entry else ""

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search across SQLite FTS5 + vector."""
        results = self.sqlite.search_memories(query, limit=limit)
        if self.vector.available:
            vec_results = self.vector.search(query, limit=limit)
            seen_keys = {r.key for r in results}
            for vr in vec_results:
                if vr["id"] not in seen_keys:
                    results.append(MemoryEntry(
                        key=vr["id"], content=vr["content"],
                        category=vr.get("metadata", {}).get("category", "general"),
                    ))
        return results[:limit]

    async def forget(self, key: str) -> bool:
        """Delete a memory."""
        self.vector.delete(key)
        return self.sqlite.delete_memory(key)

    async def list_all(self, category: str | None = None) -> list[MemoryEntry]:
        return self.sqlite.list_memories(category=category)

    # Entity management
    async def store_entity(self, entity: Entity):
        self.sqlite.store_entity(entity)
        if self.vault:
            self.vault.write_entity(entity)

    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        return self.sqlite.search_entities(query, limit=limit)

    # Session management
    async def save_session(self, session_id: str, messages: list[dict],
                           summary: str = "", agent_name: str = "default"):
        self.sqlite.save_session(session_id, messages, summary, agent_name=agent_name)

    async def load_session(self, session_id: str) -> dict | None:
        return self.sqlite.load_session(session_id)

    async def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        return self.sqlite.search_sessions(query, limit)

    # Memory tree (from OpenHuman)
    async def ingest_leaf(self, source: str, content: str,
                          entities: list[str] | None = None) -> str:
        """Ingest a leaf into the memory tree."""
        leaf_id = hashlib.sha256(f"{source}:{content[:100]}:{time.time()}".encode()).hexdigest()[:16]
        leaf = MemoryLeaf(
            id=leaf_id, source=source, content=content,
            entities=entities or [],
        )
        self.sqlite.store_leaf(leaf)
        return leaf_id

    async def get_leaves(self, source: str, limit: int = 50) -> list[MemoryLeaf]:
        return self.sqlite.get_leaves_by_source(source, limit=limit)

    # Plugin KV
    async def plugin_set(self, plugin_id: str, key: str, value: str):
        self.sqlite.set_plugin_kv(plugin_id, key, value)

    async def plugin_get(self, plugin_id: str, key: str) -> str | None:
        return self.sqlite.get_plugin_kv(plugin_id, key)

    # Agent state
    async def save_agent_state(self, agent_id: str, state: dict):
        self.sqlite.save_agent_state(agent_id, state)

    async def load_agent_state(self, agent_id: str) -> dict | None:
        return self.sqlite.load_agent_state(agent_id)
