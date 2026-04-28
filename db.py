import json
import os
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any

import aiosqlite


@dataclass(frozen=True)
class StoredMessage:
    chat_id: int
    message_id: int
    from_user_id: int
    from_username: str | None
    created_at_ts: int
    text: str | None
    caption: str | None
    media: dict[str, Any] | None


@dataclass(frozen=True)
class Project:
    id: int
    chat_id: int
    slug: str
    name: str
    status: str
    created_at_ts: int


CREATE_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  message_id INTEGER NOT NULL,
  from_user_id INTEGER NOT NULL,
  from_username TEXT,
  created_at_ts INTEGER NOT NULL,
  text TEXT,
  caption TEXT,
  media_json TEXT,
  media_type TEXT,
  UNIQUE(chat_id, message_id)
);
"""

CREATE_DAILY_SUMMARIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  date_key TEXT NOT NULL,
  start_ts INTEGER NOT NULL,
  end_ts INTEGER NOT NULL,
  summary_text TEXT NOT NULL,
  created_at_ts INTEGER NOT NULL,
  UNIQUE(chat_id, date_key)
);
"""

CREATE_ASK_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ask_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  created_at_ts INTEGER NOT NULL,
  question TEXT NOT NULL,
  selected_lines_json TEXT NOT NULL,
  bot_answer TEXT NOT NULL DEFAULT ''
);
"""

CREATE_BOT_MESSAGE_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bot_message_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  bot_message_id INTEGER NOT NULL,
  session_id INTEGER NOT NULL,
  created_at_ts INTEGER NOT NULL,
  UNIQUE(chat_id, bot_message_id)
);
"""

CREATE_PROJECTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  slug TEXT NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at_ts INTEGER NOT NULL,
  UNIQUE(chat_id, slug)
);
"""

CREATE_PROJECT_MEMORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS project_memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL,
  section TEXT NOT NULL,
  content TEXT NOT NULL,
  updated_at_ts INTEGER NOT NULL,
  UNIQUE(project_id, section)
);
"""

CREATE_CHAT_ACTIVE_PROJECT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_active_project (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL UNIQUE,
  project_id INTEGER NOT NULL,
  updated_at_ts INTEGER NOT NULL
);
"""

CREATE_CHAT_REGISTRY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_registry (
  chat_id INTEGER PRIMARY KEY,
  chat_title TEXT NOT NULL,
  chat_type TEXT NOT NULL,
  updated_at_ts INTEGER NOT NULL
);
"""

CREATE_CHAT_MEMBERS_SEEN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_members_seen (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  last_seen_ts INTEGER NOT NULL,
  UNIQUE(chat_id, user_id)
);
"""

CREATE_LONG_TERM_FACTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS long_term_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  fact_kind TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at_ts INTEGER NOT NULL
);
"""


class Database:
    def __init__(self, path: str) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        parent = os.path.dirname(self._path) or "."
        os.makedirs(parent, exist_ok=True)

        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(
            CREATE_MESSAGES_TABLE_SQL
            + CREATE_DAILY_SUMMARIES_TABLE_SQL
            + CREATE_ASK_SESSIONS_TABLE_SQL
            + CREATE_BOT_MESSAGE_SESSIONS_TABLE_SQL
            + CREATE_PROJECTS_TABLE_SQL
            + CREATE_PROJECT_MEMORY_TABLE_SQL
            + CREATE_CHAT_ACTIVE_PROJECT_TABLE_SQL
            + CREATE_CHAT_REGISTRY_TABLE_SQL
            + CREATE_CHAT_MEMBERS_SEEN_TABLE_SQL
            + CREATE_LONG_TERM_FACTS_TABLE_SQL
        )
        await self._conn.commit()
        # Migration: add bot_answer column if it doesn't exist yet (old DBs).
        try:
            await self._conn.execute(
                "ALTER TABLE ask_sessions ADD COLUMN bot_answer TEXT NOT NULL DEFAULT ''"
            )
            await self._conn.commit()
        except Exception:
            pass

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def insert_message(self, msg: StoredMessage) -> None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        await self._conn.execute(
            """
            INSERT OR IGNORE INTO messages (
              chat_id, message_id, from_user_id, from_username, created_at_ts,
              text, caption, media_json, media_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.chat_id,
                msg.message_id,
                msg.from_user_id,
                msg.from_username,
                msg.created_at_ts,
                msg.text,
                msg.caption,
                json.dumps(msg.media) if msg.media is not None else None,
                (msg.media or {}).get("type") if msg.media is not None else None,
            ),
        )
        await self._conn.commit()

    async def insert_daily_summary(
        self,
        *,
        chat_id: int,
        date_key: str,
        start_ts: int,
        end_ts: int,
        summary_text: str,
    ) -> None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        created_at_ts = int(time.time())
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO daily_summaries (
              chat_id, date_key, start_ts, end_ts, summary_text, created_at_ts
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                date_key,
                start_ts,
                end_ts,
                summary_text,
                created_at_ts,
            ),
        )
        await self._conn.commit()

    async def get_latest_daily_summary(self, chat_id: int) -> tuple[str, int, int] | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT date_key, start_ts, end_ts, summary_text
            FROM daily_summaries
            WHERE chat_id = ?
            ORDER BY date_key DESC
            LIMIT 1
            """,
            (chat_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return row["summary_text"], int(row["start_ts"]), int(row["end_ts"])

    async def get_distinct_chat_ids_in_range(
        self,
        start_ts: int,
        end_ts: int,
        allowed_chat_ids: list[int] | None,
    ) -> list[int]:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        params: list[Any] = [start_ts, end_ts]
        allowed_sql = ""
        if allowed_chat_ids:
            placeholders = ",".join(["?"] * len(allowed_chat_ids))
            allowed_sql = f" AND chat_id IN ({placeholders}) "
            params.extend(allowed_chat_ids)

        cur = await self._conn.execute(
            f"""
            SELECT DISTINCT chat_id
            FROM messages
            WHERE created_at_ts >= ?
              AND created_at_ts < ?
              {allowed_sql}
            """,
            tuple(params),
        )
        rows = await cur.fetchall()
        return [int(r["chat_id"]) for r in rows]

    async def get_messages_for_chat_in_range(
        self, chat_id: int, start_ts: int, end_ts: int, limit: int
    ) -> list[StoredMessage]:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT chat_id, message_id, from_user_id, from_username, created_at_ts,
                   text, caption, media_json
            FROM messages
            WHERE chat_id = ?
              AND created_at_ts >= ?
              AND created_at_ts < ?
            ORDER BY created_at_ts ASC
            LIMIT ?
            """,
            (chat_id, start_ts, end_ts, limit),
        )
        rows = await cur.fetchall()

        out: list[StoredMessage] = []
        for r in rows:
            media_json = r["media_json"]
            media = json.loads(media_json) if media_json else None
            out.append(
                StoredMessage(
                    chat_id=int(r["chat_id"]),
                    message_id=int(r["message_id"]),
                    from_user_id=int(r["from_user_id"]),
                    from_username=r["from_username"],
                    created_at_ts=int(r["created_at_ts"]),
                    text=r["text"],
                    caption=r["caption"],
                    media=media,
                )
            )
        return out

    async def get_recent_messages_for_chat(
        self, chat_id: int, min_ts: int, limit: int
    ) -> list[StoredMessage]:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT chat_id, message_id, from_user_id, from_username, created_at_ts,
                   text, caption, media_json
            FROM messages
            WHERE chat_id = ?
              AND created_at_ts >= ?
            ORDER BY created_at_ts DESC
            LIMIT ?
            """,
            (chat_id, min_ts, limit),
        )
        rows = await cur.fetchall()
        out: list[StoredMessage] = []
        for r in rows:
            media_json = r["media_json"]
            media = json.loads(media_json) if media_json else None
            out.append(
                StoredMessage(
                    chat_id=int(r["chat_id"]),
                    message_id=int(r["message_id"]),
                    from_user_id=int(r["from_user_id"]),
                    from_username=r["from_username"],
                    created_at_ts=int(r["created_at_ts"]),
                    text=r["text"],
                    caption=r["caption"],
                    media=media,
                )
            )
        return out

    async def create_ask_session(
        self,
        *,
        chat_id: int,
        user_id: int,
        question: str,
        selected_lines: list[str],
    ) -> int:
        """
        Stores the context used for answering, so button callbacks can run
        deterministic “find numbers / analysis / show messages” actions.
        """
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            INSERT INTO ask_sessions (chat_id, user_id, created_at_ts, question, selected_lines_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                user_id,
                int(time.time()),
                question,
                json.dumps(selected_lines, ensure_ascii=False),
            ),
        )
        await self._conn.commit()
        return int(cur.lastrowid)

    async def get_ask_session(
        self,
        *,
        session_id: int,
        chat_id: int,
        user_id: int,
    ) -> tuple[str, list[str]] | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT question, selected_lines_json
            FROM ask_sessions
            WHERE id = ? AND chat_id = ? AND user_id = ?
            LIMIT 1
            """,
            (session_id, chat_id, user_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None

        question = str(row["question"])
        selected_lines_json = row["selected_lines_json"] or "[]"
        selected_lines = json.loads(selected_lines_json)
        if not isinstance(selected_lines, list):
            selected_lines = []
        selected_lines_str = [str(x) for x in selected_lines]
        return question, selected_lines_str

    async def get_ask_session_by_id(
        self, *, session_id: int
    ) -> tuple[str, list[str], str] | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT question, selected_lines_json, bot_answer
            FROM ask_sessions
            WHERE id = ?
            LIMIT 1
            """,
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None

        question = str(row["question"])
        selected_lines_json = row["selected_lines_json"] or "[]"
        selected_lines = json.loads(selected_lines_json)
        if not isinstance(selected_lines, list):
            selected_lines = []
        selected_lines_str = [str(x) for x in selected_lines]
        bot_answer = str(row["bot_answer"] or "")
        return question, selected_lines_str, bot_answer

    async def update_session_bot_answer(
        self, *, session_id: int, bot_answer: str
    ) -> None:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        await self._conn.execute(
            "UPDATE ask_sessions SET bot_answer = ? WHERE id = ?",
            (bot_answer, session_id),
        )
        await self._conn.commit()

    async def map_bot_message_to_session(
        self, *, chat_id: int, bot_message_id: int, session_id: int
    ) -> None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        created_at_ts = int(time.time())
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO bot_message_sessions (chat_id, bot_message_id, session_id, created_at_ts)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, bot_message_id, session_id, created_at_ts),
        )
        await self._conn.commit()

    async def get_session_by_bot_message(
        self, *, chat_id: int, bot_message_id: int
    ) -> int | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT session_id
            FROM bot_message_sessions
            WHERE chat_id = ?
              AND bot_message_id = ?
            LIMIT 1
            """,
            (chat_id, bot_message_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return int(row["session_id"])

    async def create_project(self, *, chat_id: int, slug: str, name: str) -> int:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        created_at_ts = int(time.time())
        cur = await self._conn.execute(
            """
            INSERT INTO projects (chat_id, slug, name, status, created_at_ts)
            VALUES (?, ?, ?, 'active', ?)
            """,
            (chat_id, slug, name, created_at_ts),
        )
        await self._conn.commit()
        return int(cur.lastrowid)

    async def list_projects(self, *, chat_id: int) -> list[Project]:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT id, chat_id, slug, name, status, created_at_ts
            FROM projects
            WHERE chat_id = ?
            ORDER BY created_at_ts DESC
            """,
            (chat_id,),
        )
        rows = await cur.fetchall()
        out: list[Project] = []
        for row in rows:
            out.append(
                Project(
                    id=int(row["id"]),
                    chat_id=int(row["chat_id"]),
                    slug=str(row["slug"]),
                    name=str(row["name"]),
                    status=str(row["status"]),
                    created_at_ts=int(row["created_at_ts"]),
                )
            )
        return out

    async def get_project_by_ref(self, *, chat_id: int, project_ref: str) -> Project | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        project_ref_clean = (project_ref or "").strip()
        if not project_ref_clean:
            return None

        row = None
        if project_ref_clean.isdigit():
            cur = await self._conn.execute(
                """
                SELECT id, chat_id, slug, name, status, created_at_ts
                FROM projects
                WHERE chat_id = ? AND id = ?
                LIMIT 1
                """,
                (chat_id, int(project_ref_clean)),
            )
            row = await cur.fetchone()
        if row is None:
            cur = await self._conn.execute(
                """
                SELECT id, chat_id, slug, name, status, created_at_ts
                FROM projects
                WHERE chat_id = ? AND slug = ?
                LIMIT 1
                """,
                (chat_id, project_ref_clean.lower()),
            )
            row = await cur.fetchone()
        if row is None:
            return None

        return Project(
            id=int(row["id"]),
            chat_id=int(row["chat_id"]),
            slug=str(row["slug"]),
            name=str(row["name"]),
            status=str(row["status"]),
            created_at_ts=int(row["created_at_ts"]),
        )

    async def set_active_project(self, *, chat_id: int, project_id: int) -> bool:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        # Guard: project must belong to this chat.
        cur = await self._conn.execute(
            """
            SELECT id
            FROM projects
            WHERE id = ? AND chat_id = ?
            LIMIT 1
            """,
            (project_id, chat_id),
        )
        row = await cur.fetchone()
        if row is None:
            return False

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO chat_active_project (chat_id, project_id, updated_at_ts)
            VALUES (?, ?, ?)
            """,
            (chat_id, project_id, int(time.time())),
        )
        await self._conn.commit()
        return True

    async def get_active_project(self, *, chat_id: int) -> Project | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT p.id, p.chat_id, p.slug, p.name, p.status, p.created_at_ts
            FROM chat_active_project cap
            JOIN projects p ON p.id = cap.project_id
            WHERE cap.chat_id = ?
              AND p.chat_id = ?
            LIMIT 1
            """,
            (chat_id, chat_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None

        return Project(
            id=int(row["id"]),
            chat_id=int(row["chat_id"]),
            slug=str(row["slug"]),
            name=str(row["name"]),
            status=str(row["status"]),
            created_at_ts=int(row["created_at_ts"]),
        )

    async def upsert_project_memory(
        self, *, chat_id: int, project_id: int, section: str, content: str
    ) -> bool:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT id
            FROM projects
            WHERE id = ? AND chat_id = ?
            LIMIT 1
            """,
            (project_id, chat_id),
        )
        row = await cur.fetchone()
        if row is None:
            return False

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO project_memory (project_id, section, content, updated_at_ts)
            VALUES (?, ?, ?, ?)
            """,
            (project_id, section, content, int(time.time())),
        )
        await self._conn.commit()
        return True

    async def get_project_memory(
        self, *, chat_id: int, project_id: int
    ) -> dict[str, str]:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT pm.section, pm.content
            FROM project_memory pm
            JOIN projects p ON p.id = pm.project_id
            WHERE pm.project_id = ?
              AND p.chat_id = ?
            ORDER BY pm.section ASC
            """,
            (project_id, chat_id),
        )
        rows = await cur.fetchall()
        out: dict[str, str] = {}
        for row in rows:
            out[str(row["section"])] = str(row["content"])
        return out

    async def upsert_chat_registry(
        self, *, chat_id: int, chat_title: str, chat_type: str
    ) -> None:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO chat_registry (chat_id, chat_title, chat_type, updated_at_ts)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, chat_title, chat_type, int(time.time())),
        )
        await self._conn.commit()

    async def get_chat_registry_entry(self, *, chat_id: int) -> tuple[int, str, str] | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        cur = await self._conn.execute(
            """
            SELECT chat_id, chat_title, chat_type
            FROM chat_registry
            WHERE chat_id = ?
            LIMIT 1
            """,
            (chat_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return int(row["chat_id"]), str(row["chat_title"]), str(row["chat_type"])

    async def mark_chat_member_seen(self, *, chat_id: int, user_id: int) -> None:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO chat_members_seen (chat_id, user_id, last_seen_ts)
            VALUES (?, ?, ?)
            """,
            (chat_id, user_id, int(time.time())),
        )
        await self._conn.commit()

    async def has_user_seen_chat(self, *, chat_id: int, user_id: int) -> bool:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        cur = await self._conn.execute(
            """
            SELECT id
            FROM chat_members_seen
            WHERE chat_id = ? AND user_id = ?
            LIMIT 1
            """,
            (chat_id, user_id),
        )
        row = await cur.fetchone()
        return row is not None

    async def has_cross_chat_access(
        self, *, user_id: int, current_chat_id: int, target_chat_id: int
    ) -> bool:
        left = await self.has_user_seen_chat(chat_id=current_chat_id, user_id=user_id)
        if not left:
            return False
        right = await self.has_user_seen_chat(chat_id=target_chat_id, user_id=user_id)
        return right

    async def find_chat_by_title_fuzzy(
        self,
        *,
        title_query: str,
        min_score: float,
        max_candidates: int,
    ) -> tuple[int, str, float] | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        cur = await self._conn.execute(
            """
            SELECT chat_id, chat_title
            FROM chat_registry
            ORDER BY updated_at_ts DESC
            LIMIT 500
            """
        )
        rows = await cur.fetchall()
        q = (title_query or "").strip().lower()
        if not q:
            return None

        scored: list[tuple[int, str, float]] = []
        for row in rows:
            chat_id = int(row["chat_id"])
            title = str(row["chat_title"])
            title_l = title.lower()
            ratio = SequenceMatcher(None, q, title_l).ratio()
            if q in title_l:
                ratio = max(ratio, 0.95)
            if ratio >= min_score:
                scored.append((chat_id, title, ratio))

        if not scored:
            return None
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[: max(1, max_candidates)][0]

    async def cleanup_old_messages(self, *, older_than_ts: int) -> int:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        cur = await self._conn.execute(
            """
            DELETE FROM messages
            WHERE created_at_ts < ?
            """,
            (older_than_ts,),
        )
        await self._conn.commit()
        return int(cur.rowcount or 0)

    async def has_messages_older_than(self, *, chat_id: int, older_than_ts: int) -> bool:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        cur = await self._conn.execute(
            """
            SELECT id
            FROM messages
            WHERE chat_id = ?
              AND created_at_ts < ?
            LIMIT 1
            """,
            (chat_id, older_than_ts),
        )
        row = await cur.fetchone()
        return row is not None

    async def get_messages_for_chat_topic(
        self,
        *,
        chat_id: int,
        topic_query: str,
        min_ts: int,
        limit: int,
    ) -> list[StoredMessage]:
        if self._conn is None:
            raise RuntimeError("DB not connected")
        topic = (topic_query or "").strip().lower()
        if not topic:
            return await self.get_recent_messages_for_chat(chat_id=chat_id, min_ts=min_ts, limit=limit)

        pattern = f"%{topic}%"
        cur = await self._conn.execute(
            """
            SELECT chat_id, message_id, from_user_id, from_username, created_at_ts,
                   text, caption, media_json
            FROM messages
            WHERE chat_id = ?
              AND created_at_ts >= ?
              AND (
                LOWER(COALESCE(text, '')) LIKE ?
                OR LOWER(COALESCE(caption, '')) LIKE ?
              )
            ORDER BY created_at_ts DESC
            LIMIT ?
            """,
            (chat_id, min_ts, pattern, pattern, limit),
        )
        rows = await cur.fetchall()
        out: list[StoredMessage] = []
        for r in rows:
            media_json = r["media_json"]
            media = json.loads(media_json) if media_json else None
            out.append(
                StoredMessage(
                    chat_id=int(r["chat_id"]),
                    message_id=int(r["message_id"]),
                    from_user_id=int(r["from_user_id"]),
                    from_username=r["from_username"],
                    created_at_ts=int(r["created_at_ts"]),
                    text=r["text"],
                    caption=r["caption"],
                    media=media,
                )
            )
        return out

