from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

from db import Database, StoredMessage
from llm import LLMClient
from retrieval import message_to_excerpt


@dataclass(frozen=True)
class DailyRunParams:
    daily_hour: int
    daily_minute: int
    timezone_name: str
    max_context_chars: int
    messages_limit_for_summary: int
    lookback_hours: int


def _tz(tz_name: str) -> ZoneInfo:
    return ZoneInfo(tz_name)


async def _sleep_until_next_target(
    *, hour: int, minute: int, timezone_name: str
) -> None:
    tz = _tz(timezone_name)
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    seconds = (target - now).total_seconds()
    await asyncio.sleep(max(0.0, seconds))


def _date_key_in_tz(d: datetime, tz: ZoneInfo) -> str:
    return d.astimezone(tz).strftime("%Y-%m-%d")


_TELEGRAM_SAFE_LEN = 4000


def _sanitize_daily_summary_for_telegram(raw: str) -> str:
    """
    Telegram is sent without parse_mode; strip common Markdown artifacts
    models still sometimes emit.
    """
    t = (raw or "").strip()
    if not t:
        return t
    # Headings like ### Foo
    t = re.sub(r"(?m)^#{1,6}\s*", "", t)
    # Bold / italic markers
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return t.strip()


async def _send_plain_text_chunks(bot, chat_id: int, text: str) -> None:
    t = text or ""
    while t:
        part = t[:_TELEGRAM_SAFE_LEN]
        t = t[_TELEGRAM_SAFE_LEN:]
        await bot.send_message(
            chat_id, part, disable_web_page_preview=True
        )


async def daily_summary_loop(
    *,
    bot,
    db: Database,
    llm: LLMClient | None,
    allowed_chat_ids: list[int],
    params: DailyRunParams,
) -> None:
    """
    Runs forever:
    - waits until next wall time (hour:minute) in configured timezone (default Europe/Moscow)
    - summarizes messages from the last N hours (default 24)
    - posts summary to each chat with new messages
    """
    while True:
        try:
            await _sleep_until_next_target(
                hour=params.daily_hour,
                minute=params.daily_minute,
                timezone_name=params.timezone_name,
            )

            tz = _tz(params.timezone_name)
            end_dt = datetime.now(tz)
            start_dt = end_dt - timedelta(hours=params.lookback_hours)
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            date_key = _date_key_in_tz(end_dt, tz)

            distinct_chats = await db.get_distinct_chat_ids_in_range(
                start_ts=start_ts, end_ts=end_ts, allowed_chat_ids=allowed_chat_ids
            )
            if not distinct_chats:
                continue

            sem = asyncio.Semaphore(3)

            async def summarize_one(chat_id: int) -> None:
                async with sem:
                    try:
                        msgs: Sequence[StoredMessage] = (
                            await db.get_messages_for_chat_in_range(
                                chat_id=chat_id,
                                start_ts=start_ts,
                                end_ts=end_ts,
                                limit=params.messages_limit_for_summary,
                            )
                        )
                        if not msgs:
                            return

                        excerpts = [message_to_excerpt(m) for m in msgs]
                        selected: list[str] = []
                        total_chars = 0
                        for line in excerpts:
                            extra = len(line) + 1
                            if total_chars + extra > params.max_context_chars:
                                break
                            selected.append(line)
                            total_chars += extra

                        if not selected:
                            return

                        if llm is not None:
                            try:
                                daily = await llm.daily_summary(
                                    chat_id=chat_id,
                                    start_ts=start_ts,
                                    end_ts=end_ts,
                                    message_bullets=selected,
                                )

                                summary_text = daily.text.strip()
                                if not summary_text:
                                    summary_text = (
                                        "Ежедневная сводка: модель не вернула текст."
                                    )
                            except Exception:
                                summary_lines = selected[:10]
                                summary_text = (
                                    "Ежедневная сводка (упрощённая, без LLM из-за ошибки OpenAI):\n\n"
                                    + "\n".join([f"— {line}" for line in summary_lines])
                                )
                        else:
                            summary_lines = selected[:10]
                            summary_text = (
                                "Ежедневная сводка (упрощённая, без LLM):\n\n"
                                + "\n".join([f"— {line}" for line in summary_lines])
                            )

                        summary_text = _sanitize_daily_summary_for_telegram(summary_text)

                        header = f"📅 Сводка за {date_key} (период ~{params.lookback_hours} ч)\n\n"
                        out_text = header + summary_text

                        await db.insert_daily_summary(
                            chat_id=chat_id,
                            date_key=date_key,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            summary_text=out_text,
                        )

                        await _send_plain_text_chunks(bot, chat_id, out_text)
                    except Exception as e:
                        print(
                            f"[daily] skip/fail chat_id={chat_id}: {type(e).__name__}: {e}"
                        )

            await asyncio.gather(*(summarize_one(cid) for cid in distinct_chats))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[daily] loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(60)
