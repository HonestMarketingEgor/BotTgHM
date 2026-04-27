from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from db import StoredMessage


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_COMMAND_RE = re.compile(r"^/([A-Za-z0-9_]+)(?:@[A-Za-z0-9_]+)?(?:\s+(.*))?$")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "you",
    "your",
    "are",
    "was",
    "were",
    "but",
    "not",
    "have",
    "has",
    "had",
    "will",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "about",
    "into",
    "out",
    "our",
    "we",
    "i",
    "it",
    "as",
    "at",
    "be",
    "on",
    "in",
    "of",
    "to",
    "is",
    "im",
    "ive",
    "dont",
    "do",
    "did",
}


def _humanize_command_text(content: str) -> str:
    raw = (content or "").strip()
    m = _COMMAND_RE.match(raw)
    if not m:
        return raw

    command = (m.group(1) or "").lower()
    args = (m.group(2) or "").strip()
    args_hint = f": {args[:120]}{'…' if len(args) > 120 else ''}" if args else ""

    action_map = {
        "start": "запустил бота и запросил стартовую инструкцию",
        "help": "запросил список возможностей бота",
        "ask": "запустил запрос на анализ",
        "mode": "изменил или проверил режим работы бота",
        "reset": "сбросил текущий режим бота",
        "chat_info": "запросил данные о текущем чате",
        "vkmatch": "запустил сопоставление лидов (VkMatch)",
        "daily_summary": "запросил дневную сводку по чату",
        "summary": "запросил дневную сводку по чату",
        "project_new": "создал новый проект",
        "project_list": "запросил список проектов",
        "project_use": "сменил активный проект",
        "project_show": "запросил карточку активного проекта",
        "project_set": "обновил секцию памяти проекта",
    }
    action = action_map.get(command, f"выполнил команду /{command}")
    return f"[ДЕЙСТВИЕ] {action}{args_hint}"


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in tokens if t not in _STOPWORDS]


def message_to_excerpt(msg: StoredMessage) -> str:
    dt = datetime.fromtimestamp(msg.created_at_ts, tz=timezone.utc)
    user = msg.from_username or f"user:{msg.from_user_id}"

    content = msg.text or msg.caption or ""
    if msg.media:
        media_type = msg.media.get("type") or "media"
        # Prefer a marker for images so the bot can download & analyze.
        if (not content) and media_type == "photo" and msg.media.get("file_id"):
            content = f"[TELEGRAM_PHOTO_FILE_ID] {msg.media.get('file_id')}"
        elif not content:
            content = f"[{media_type}]"

        # If we have any "name"/caption-like metadata, include it.
        name = msg.media.get("name")
        if name and name not in content:
            content += f" {name}"

    content = (content or "").strip()
    if content.startswith("/"):
        content = _humanize_command_text(content)
    if not content:
        content = "[empty]"

    return f"{dt.strftime('%Y-%m-%d %H:%M')} @{user}: {content}"


def rank_messages(question: str, messages: Sequence[StoredMessage]) -> list[StoredMessage]:
    q_tokens = tokenize(question)
    if not q_tokens:
        return list(messages[:])

    def score(msg: StoredMessage) -> int:
        hay = (msg.text or msg.caption or "").lower()
        if not hay:
            if msg.media:
                hay = str(msg.media).lower()
            else:
                hay = ""
        return sum(1 for t in q_tokens if t in hay)

    sorted_msgs = sorted(messages, key=score, reverse=True)
    # If no message matched anything, return in recency order (input is already DESC).
    if sorted_msgs and score(sorted_msgs[0]) == 0:
        return list(messages)
    return sorted_msgs


def budgeted_join(lines: Iterable[str], *, max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for line in lines:
        if not line:
            continue
        extra = len(line) + 1
        if total + extra > max_chars:
            break
        out.append(line)
        total += extra
    return "\n".join(out)


def build_project_memory_lines(memory: dict[str, str]) -> list[str]:
    preferred_order = [
        "brief",
        "kpi",
        "constraints",
        "audience",
        "hypotheses",
        "notes",
    ]
    out: list[str] = []
    used: set[str] = set()
    for section in preferred_order:
        content = (memory.get(section) or "").strip()
        if content:
            out.append(f"[PROJECT:{section}] {content}")
            used.add(section)
    for section, content in memory.items():
        if section in used:
            continue
        c = (content or "").strip()
        if c:
            out.append(f"[PROJECT:{section}] {c}")
    return out


def prepend_project_context(
    *,
    project_lines: Sequence[str],
    context_lines: Sequence[str],
    max_chars: int,
) -> list[str]:
    out: list[str] = []
    total = 0
    for line in project_lines:
        if not line:
            continue
        extra = len(line) + 1
        if total + extra > max_chars:
            break
        out.append(line)
        total += extra

    for line in context_lines:
        if not line:
            continue
        extra = len(line) + 1
        if total + extra > max_chars:
            break
        out.append(line)
        total += extra
    return out

