import json
import os
import time
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
  selected_lines_json TEXT NOT NULL
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
        )
        await self._conn.commit()

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
    ) -> tuple[str, list[str]] | None:
        if self._conn is None:
            raise RuntimeError("DB not connected")

        cur = await self._conn.execute(
            """
            SELECT question, selected_lines_json
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
        return question, selected_lines_str

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

