from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

from config import load_config
from daily_loop import DailyRunParams, daily_summary_loop
from db import Database, StoredMessage
from formatter import (
    build_analysis_fallback,
    build_assistant_fallback,
    build_help_text,
    build_help_redirect,
)
from llm import LLMClient
from message_extract import extract_media_metadata
from retrieval import message_to_excerpt, rank_messages

HELP_MODE = "help_mode"
ASSISTANT_MODE = "assistant_mode"
ANALYSIS_MODE = "analysis_mode"

VALID_CHAT_MODES = {ASSISTANT_MODE, ANALYSIS_MODE}


@dataclass
class ChatRuntimeState:
    mode: str


def _is_help_intent(text: str) -> bool:
    q = (text or "").strip().lower().replace("ё", "е")
    markers = [
        "что ты умеешь",
        "что умеет бот",
        "как пользоваться",
        "инструкция",
        "help",
        "start",
        "команды",
    ]
    return any(m in q for m in markers)


def _is_analysis_intent(text: str) -> bool:
    q = (text or "").strip().lower()
    markers = [
        "проанализ",
        "разбер",
        "анализ",
        "kpi",
        "метрик",
        "cpl",
        "roi",
        "сводк",
        "по чату",
        "по переписке",
        "контекст",
        "дай вывод",
    ]
    return any(m in q for m in markers)


def _normalize_mode(raw: str) -> str | None:
    t = (raw or "").strip().lower()
    aliases = {
        "assistant": ASSISTANT_MODE,
        "assistant_mode": ASSISTANT_MODE,
        "analysis": ANALYSIS_MODE,
        "analysis_mode": ANALYSIS_MODE,
    }
    return aliases.get(t)


def _parse_command_args(text: str | None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _openai_failure_reply(exc: BaseException) -> str | None:
    err = f"{type(exc).__name__}: {exc}".lower()
    if "401" in err or "invalid_api_key" in err or "authentication" in err:
        return "Не удалось обратиться к OpenAI. Проверь OPENAI_API_KEY в .env."
    if "403" in err and ("model" in err or "access" in err):
        return "Модель недоступна для текущего ключа. Проверь OPENAI_MODEL."
    if "429" in err or "insufficient_quota" in err or "rate limit" in err:
        return "Достигнут лимит OpenAI. Попробуй позже или проверь биллинг."
    if "timeout" in err or "connection" in err or "network" in err:
        return "Проблема сети при запросе к OpenAI. Попробуй позже."
    return None


async def main() -> None:
    cfg = load_config()

    bot = Bot(token=cfg.telegram_bot_token)
    dp = Dispatcher()
    router = Router()
    dp.include_router(router)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

    db = Database(cfg.db_path)
    await db.connect()

    llm: LLMClient | None = None
    if cfg.openai_api_key:
        llm = LLMClient(
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
            base_url=cfg.openai_base_url or None,
        )

    me = await bot.get_me()
    bot_id = me.id
    bot_username = (me.username or "").strip().lower()

    runtime_state_by_chat: dict[int, ChatRuntimeState] = {}
    default_mode = _normalize_mode(cfg.default_mode) or ASSISTANT_MODE

    def _chat_state(chat_id: int) -> ChatRuntimeState:
        st = runtime_state_by_chat.get(chat_id)
        if st is None:
            st = ChatRuntimeState(mode=default_mode)
            runtime_state_by_chat[chat_id] = st
        return st

    def is_bot_mentioned(message: Message) -> bool:
        if not bot_username:
            return False
        raw = (message.text or message.caption or "").lower()
        return f"@{bot_username}" in raw

    def strip_bot_mention(text: str) -> str:
        if not bot_username:
            return text.strip()
        pattern = rf"@{re.escape(bot_username)}\b"
        return re.sub(pattern, "", text or "", flags=re.IGNORECASE).strip()

    async def answer_user_query(
        *,
        message: Message,
        question: str,
        forced_mode: str | None = None,
    ) -> None:
        q = (question or "").strip()
        if not q:
            await message.reply("Напиши вопрос после команды или упоминания.")
            return

        if _is_help_intent(q):
            await message.reply(build_help_redirect(bot_username))
            return

        if forced_mode:
            mode = forced_mode
        elif _is_analysis_intent(q):
            mode = ANALYSIS_MODE
        else:
            mode = _chat_state(message.chat.id).mode

        if mode not in VALID_CHAT_MODES:
            mode = ASSISTANT_MODE

        context_lines: list[str] = []
        if mode == ANALYSIS_MODE and cfg.enable_chat_analysis:
            now = datetime.now().astimezone()
            min_ts = int((now - timedelta(days=7)).timestamp())
            recent = await db.get_recent_messages_for_chat(
                chat_id=message.chat.id, min_ts=min_ts, limit=300
            )
            ranked = rank_messages(q, recent)
            selected = ranked[: cfg.max_messages_for_ask]

            total_chars = 0
            for msg in selected:
                line = message_to_excerpt(msg)
                extra = len(line) + 1
                if total_chars + extra > cfg.max_context_chars:
                    break
                context_lines.append(line)
                total_chars += extra

        if llm is None:
            if mode == ANALYSIS_MODE:
                out = build_analysis_fallback(q, context_lines)
            else:
                out = build_assistant_fallback(q)
            await message.reply(out, disable_web_page_preview=True)
            return

        try:
            result = await llm.answer(
                mode=mode,
                question=q,
                context_messages=context_lines if mode == ANALYSIS_MODE else None,
            )
            text = (result.text or "").strip() or "Не удалось сформировать ответ."
        except Exception as e:
            hint = _openai_failure_reply(e)
            if hint:
                await message.reply(hint, disable_web_page_preview=True)
                return
            if mode == ANALYSIS_MODE:
                text = build_analysis_fallback(q, context_lines)
            else:
                text = build_assistant_fallback(q)
        await message.reply(text, disable_web_page_preview=True)

    @router.message(Command("start"))
    async def on_start(message: Message) -> None:
        await message.reply(build_help_text(bot_username))

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.reply(build_help_text(bot_username))

    @router.message(Command("mode"))
    async def on_mode(message: Message) -> None:
        if message.chat is None:
            return
        arg = _parse_command_args(message.text)
        state = _chat_state(message.chat.id)
        if not arg:
            await message.reply(
                f"Текущий режим: {state.mode}\n"
                "Изменить: /mode assistant или /mode analysis"
            )
            return
        normalized = _normalize_mode(arg)
        if normalized is None:
            await message.reply("Доступные режимы: assistant, analysis")
            return
        state.mode = normalized
        await message.reply(f"Режим обновлен: {state.mode}")

    @router.message(Command("reset"))
    async def on_reset(message: Message) -> None:
        if message.chat is None:
            return
        runtime_state_by_chat[message.chat.id] = ChatRuntimeState(mode=default_mode)
        await message.reply(f"Сброс выполнен. Режим по умолчанию: {default_mode}")

    @router.message(Command("ask"))
    async def on_ask(message: Message) -> None:
        if message.chat is None:
            return
        q = _parse_command_args(message.text)
        if not q:
            await message.reply("Формат: /ask <вопрос>")
            return
        await answer_user_query(message=message, question=q, forced_mode=ANALYSIS_MODE)

    @router.message()
    async def on_text(message: Message) -> None:
        if message.chat is None or message.from_user is None:
            return
        if message.from_user.id == bot_id:
            return
        if message.text and message.text.strip().startswith("/"):
            return

        text = message.text or None
        caption = getattr(message, "caption", None) or None
        media = extract_media_metadata(message)
        if text or caption or media:
            stored = StoredMessage(
                chat_id=message.chat.id,
                message_id=message.message_id,
                from_user_id=message.from_user.id,
                from_username=message.from_user.username,
                created_at_ts=int(message.date.timestamp()),
                text=text,
                caption=caption,
                media=media,
            )
            await db.insert_message(stored)

        is_group = message.chat.type in {"group", "supergroup"}
        if is_group and not is_bot_mentioned(message):
            return

        raw = text or caption or ""
        q = strip_bot_mention(raw) if is_group else raw.strip()
        if not q:
            return
        await answer_user_query(message=message, question=q, forced_mode=None)

    params = DailyRunParams(
        daily_hour=cfg.daily_hour,
        daily_minute=cfg.daily_minute,
        timezone_name=cfg.daily_timezone,
        max_context_chars=cfg.max_context_chars,
        messages_limit_for_summary=800,
        lookback_hours=24,
    )

    daily_task = asyncio.create_task(
        daily_summary_loop(
            bot=bot,
            db=db,
            llm=llm,
            allowed_chat_ids=cfg.allowed_chat_ids or [],
            params=params,
        )
    )

    try:
        await dp.start_polling(bot)
    finally:
        daily_task.cancel()
        await db.close()

