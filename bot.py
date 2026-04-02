from __future__ import annotations

import asyncio as _bootstrap_asyncio
from bot_v2 import main as _bot_v2_main

if __name__ == "__main__":
    _bootstrap_asyncio.run(_bot_v2_main())
    raise SystemExit(0)

import asyncio
import json
import re
import time
import traceback
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from config import load_config
from daily_loop import DailyRunParams, daily_summary_loop
from db import Database, StoredMessage
from llm import LLMClient, classify_response_mode
from message_extract import extract_media_metadata
from retrieval import (
    budgeted_join,
    build_project_memory_lines,
    message_to_excerpt,
    prepend_project_context,
    rank_messages,
)
from formatter import (
    build_numbers_answer,
    build_marketing_suggestions_fallback,
    build_tasks_answer,
    build_fallback_answer,
    build_freeform_answer,
    build_help_text,
    build_help_redirect,
)
from links import extract_urls, fetch_url_text, fetch_google_drive_folder_image_urls
import httpx
from table_kpis import build_numbers_from_link_context

_MAXON_JOKE_REPLY = (
    "Медведь увидел в лесу горящую машину, сел в неё — и тоже сгорел."
)
_PROJECT_SECTIONS = {"brief", "kpi", "constraints", "audience", "hypotheses", "notes"}
_PROJECT_SECTIONS_ORDER = ["brief", "kpi", "constraints", "audience", "hypotheses", "notes"]

_DEBUG_LOG_PATH = "/Users/pelemenio/telegram-context-bot/.cursor/debug-283857.log"
_DEBUG_SESSION_ID = "283857"
_AGENT_DEBUG_LOG_PATH = "/Users/pelemenio/telegram-context-bot/.cursor/debug-aff1f6.log"
_AGENT_DEBUG_SESSION_ID = "aff1f6"


def _debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
) -> None:
    # #region agent log
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # #endregion


def _agent_diag(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
) -> None:
    payload = {
        "sessionId": _AGENT_DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _is_maxon_joke_request(text: str) -> bool:
    if not (text or "").strip():
        return False
    t = (text or "").casefold().replace("\u00a0", " ").replace("\u202f", " ")
    t = " ".join(t.split())
    return "шутка максона" in t


def _is_bot_alive_ping(question: str) -> bool:
    """Короткий «жив ли бот» без вызова LLM — не зависит от OpenAI."""
    q = (question or "").strip().lower().replace("ё", "е")
    q = " ".join(q.split())
    if len(q) > 120 or extract_urls(question):
        return False
    # «ты активен?», «ты работаешь?», «ты активен, работаешь?», «бот ок?»
    if re.fullmatch(
        r"(ты\s+)?(активен|работаешь|на\s+связи|жив(?:ой)?)"
        r"(?:\s*,\s*(активен|работаешь))?\s*\??",
        q,
    ):
        return True
    return bool(re.fullmatch(r"(бот\s+)?(в\s+строю|ок|alive|ping)\??", q))


def _openai_failure_reply(exc: BaseException) -> str | None:
    """Возвращает текст для пользователя для типичных ошибок API; иначе None."""
    err = f"{type(exc).__name__}: {exc}".lower()
    if "unsupported_country_region_territory" in err:
        # Region-restricted account should use local fallback answer path.
        return None
    if (
        "401" in err
        or "invalid_api_key" in err
        or "authentication" in err
        or "incorrect api key" in err
        or "permission denied" in err
    ):
        return (
            "Не удалось обратиться к OpenAI: похоже, неверный, просроченный или неактивный "
            "`OPENAI_API_KEY`.\nПроверь ключ в `.env` на сервере и лимиты на "
            "https://platform.openai.com/api-keys"
        )
    if "403" in err and ("model" in err or "access" in err or "country" in err):
        return (
            "OpenAI отклонил запрос (403): у ключа нет доступа к модели или региону.\n"
            "Проверь `OPENAI_MODEL` в `.env` и настройки аккаунта на platform.openai.com."
        )
    if (
        "model" in err
        and (
            "does not exist" in err
            or "not found" in err
            or "model_not_found" in err
            or "invalid model" in err
        )
    ):
        return (
            "Указанная модель OpenAI недоступна для этого ключа.\n"
            "Проверь `OPENAI_MODEL` в `.env` (например `gpt-4o-mini`)."
        )
    if "insufficient_quota" in err or "429" in err or "rate limit" in err:
        return None
    if "connection" in err or "timeout" in err or "connecterror" in err or "network" in err:
        return (
            "Не удалось связаться с серверами OpenAI (сеть / таймаут).\n"
            "Повтори запрос позже или проверь интернет на машине, где запущен бот."
        )
    return None


def _slugify_project_name(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]+", "-", (name or "").strip().lower())
    base = base.strip("-_")
    return base or "project"


def _parse_command_args(text: str | None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _response_mode_label(mode: str) -> str:
    if mode == "data_first":
        return "data_first (факты -> вывод)"
    if mode == "short_exec":
        return "short_exec (кратко для руководителя)"
    return "structured_marketing (ситуация -> гипотезы -> шаг)"


def _is_help_intent(question: str) -> bool:
    q = (question or "").strip().lower().replace("ё", "е")
    q = " ".join(q.split())
    if not q:
        return False
    help_markers = [
        "что ты умеешь",
        "что умеет бот",
        "что умеешь",
        "как пользоваться",
        "как тебя использовать",
        "какие функции",
        "какой функционал",
        "help",
        "start",
        "инструкция",
    ]
    return any(m in q for m in help_markers)


async def main() -> None:
    cfg = load_config()

    # Do not print secrets; just confirm they are present.
    import re
    import base64

    token_ok = bool(
        re.match(r"^[0-9]+:[A-Za-z0-9_-]+$", cfg.telegram_bot_token or "")
    )
    print(f"[startup] TELEGRAM_BOT_TOKEN present: {bool(cfg.telegram_bot_token)}")
    print(f"[startup] TELEGRAM_BOT_TOKEN format ok: {token_ok}")
    print(f"[startup] TELEGRAM_BOT_TOKEN length: {len(cfg.telegram_bot_token or '')}")
    print(f"[startup] OPENAI_API_KEY present: {bool(cfg.openai_api_key)}")
    print(
        f"[startup] Daily summary: {cfg.daily_hour:02d}:{cfg.daily_minute:02d} "
        f"{cfg.daily_timezone}"
    )
    if cfg.allowed_chat_ids:
        print(
            f"[startup] ALLOWED_CHAT_IDS is set ({len(cfg.allowed_chat_ids)} ids) — "
            "сводки только для этих чатов."
        )

    bot = Bot(token=cfg.telegram_bot_token)
    dp = Dispatcher()
    router = Router()
    dp.include_router(router)

    # If the bot previously used a webhook, Telegram will block long polling
    # (getUpdates) with a 409 Conflict. Clearing it makes polling work.
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        print("[startup] Webhook cleared (if it was set).")
    except Exception as e:
        # Non-fatal; polling may still work if no webhook is set.
        print(f"[startup] Webhook clear skipped/failed: {type(e).__name__}")

    db = Database(cfg.db_path)
    await db.connect()

    llm: LLMClient | None = None
    if cfg.openai_api_key:
        llm = LLMClient(
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
            suggestions_model=cfg.openai_model_suggestions,
            base_url=cfg.openai_base_url or None,
        )

    me = await bot.get_me()
    bot_id = me.id
    bot_username = (me.username or "").strip().lower()
    # #region agent log
    _agent_diag(
        run_id=f"boot-{int(time.time())}",
        hypothesis_id="H4",
        location="bot.py:main_after_get_me",
        message="bot runtime started",
        data={"bot_id": bot_id, "bot_username": bot_username},
    )
    # #endregion

    def is_bot_mentioned(message: Message) -> bool:
        if not bot_username:
            return False
        text = (message.text or message.caption or "").strip()
        if not text:
            return False
        # Simple & reliable detection for group messages: look for "@<username>".
        return f"@{bot_username}" in text.lower()

    def strip_bot_mention(text: str) -> str:
        if not bot_username:
            return text.strip()
        # Remove any occurrences of "@<username>" (case-insensitive).
        import re as _re

        return _re.sub(rf"@{_re.escape(bot_username)}\b", "", text, flags=_re.I).strip()

    @router.message(Command("start"))
    async def start_cmd(message: Message) -> None:
        # #region agent log
        _agent_diag(
            run_id=f"msg-{message.chat.id}-{message.message_id}",
            hypothesis_id="H1",
            location="bot.py:start_cmd",
            message="start command handler invoked",
            data={
                "chat_id": message.chat.id if message.chat else None,
                "raw_text": message.text or "",
            },
        )
        # #endregion
        await message.reply(build_help_text(bot_username))

    @router.message(Command("help"))
    async def help_cmd(message: Message) -> None:
        # #region agent log
        _agent_diag(
            run_id=f"msg-{message.chat.id}-{message.message_id}",
            hypothesis_id="H1",
            location="bot.py:help_cmd",
            message="help command handler invoked",
            data={
                "chat_id": message.chat.id if message.chat else None,
                "raw_text": message.text or "",
            },
        )
        # #endregion
        await message.reply(build_help_text(bot_username))

    async def answer_question(message: Message, question: str) -> None:
        question = (question or "").strip()
        debug_run_id = (
            f"ask-{message.chat.id}-{getattr(message, 'message_id', 'na')}-{int(time.time())}"
        )
        # #region agent log
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H1_input_or_branch",
            location="bot.py:answer_question_entry",
            message="answer_question entry",
            data={
                "chat_id": message.chat.id if message.chat else None,
                "from_user_id": message.from_user.id if message.from_user else None,
                "question_len": len(question),
                "is_bot_alive_ping": _is_bot_alive_ping(question),
                "is_help_intent": _is_help_intent(question),
                "has_text": bool(message.text),
                "has_caption": bool(getattr(message, "caption", None)),
            },
        )
        # #endregion
        if not question:
            # If user sent media (photo/etc.) via mention, still run analysis.
            media = extract_media_metadata(message)
            if media and media.get("type") == "photo":
                question = "Проанализируй это изображение и извлеки ключевые факты."
            elif media:
                question = "Проанализируй присланный материал и извлеки ключевые факты."
            else:
                await message.reply(
                    "Usage: `/ask <question>`", parse_mode="Markdown"
                )
                return

        if _is_maxon_joke_request(question):
            await message.reply(_MAXON_JOKE_REPLY)
            return

        if _is_bot_alive_ping(question):
            await message.reply(
                "Да, я на связи. Если «умные» ответы не приходят — проверь `OPENAI_API_KEY` "
                "и биллинг на https://platform.openai.com (логи процесса бота покажут точную ошибку)."
            )
            return

        if _is_help_intent(question):
            # #region agent log
            _agent_diag(
                run_id=f"msg-{message.chat.id}-{message.message_id}",
                hypothesis_id="H3",
                location="bot.py:answer_question_help_guard",
                message="help intent guard matched",
                data={"question": question[:200]},
            )
            # #endregion
            await message.reply(build_help_redirect(bot_username))
            return

        def _sanitize_llm_answer(candidate: str) -> str:
            # Only remove clearly forbidden phrases; formatting is left to the model.
            bad_phrases = ["из предоставленного контекста", "возможно", "вероятно"]
            low = (candidate or "").lower()
            if any(p in low for p in bad_phrases):
                return build_freeform_answer(question, selected_lines)
            return candidate

        session_id: int | None = None
        action_kb: InlineKeyboardMarkup | None = None
        phase = "before_try"

        try:
            phase = "send_thinking"
            await message.reply("Thinking…")
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H2_pre_llm_failure",
                location="bot.py:after_thinking_reply",
                message="thinking message sent",
                data={"chat_id": message.chat.id if message.chat else None},
            )
            # #endregion
            async def _typing_loop() -> None:
                # Telegram typing indicator for user feedback during long calls.
                while True:
                    try:
                        await bot.send_chat_action(message.chat.id, action="typing")
                    except Exception:
                        pass
                    await asyncio.sleep(4)

            typing_task = asyncio.create_task(_typing_loop())

            # If user included URLs, fetch+extract their text and append to context.
            urls = extract_urls(question)
            question_no_urls = question
            for u in urls:
                question_no_urls = question_no_urls.replace(u, "").strip()
            # Keep a minimal question even if only a link is provided.
            if not question_no_urls:
                question_no_urls = "Проанализируй источник по ссылке и извлеки ключевые факты."

            now = datetime.now().astimezone()
            min_ts = int((now - timedelta(days=7)).timestamp())
            recent_msgs = await db.get_recent_messages_for_chat(
                chat_id=message.chat.id, min_ts=min_ts, limit=300
            )
            active_project = await db.get_active_project(chat_id=message.chat.id)
            project_memory: dict[str, str] = {}
            project_lines: list[str] = []
            project_context_text: str | None = None
            if active_project is not None:
                project_memory = await db.get_project_memory(
                    chat_id=message.chat.id, project_id=active_project.id
                )
                project_lines = build_project_memory_lines(project_memory)
                if project_lines:
                    project_context_text = budgeted_join(
                        project_lines, max_chars=cfg.max_project_memory_chars
                    )

            # По умолчанию: если пользователь прислал ссылку, анализируем ТОЛЬКО содержимое ссылок.
            # Если он явно просит учесть “много инфы/весь чат/включая сообщения”, включаем контекст чата.
            ql = question_no_urls.lower()
            explicit_multi_context = any(
                k in ql
                for k in [
                    "много",
                    "все",
                    "всю",
                    "переписк",
                    "чат",
                    "включа",
                    "сводк",
                    "подытож",
                    "анализируй всё",
                    "ссылк",  # "ссылки"
                ]
            )
            strict_link_only = bool(urls) and not explicit_multi_context

            # Include current media in context. Otherwise the DB doesn't have this message yet,
            # and the bot won't have the file_id/image to analyze.
            current_media = extract_media_metadata(message)
            current_photo_file_ids: list[str] = []
            current_photo_marker: str | None = None
            if current_media and current_media.get("type") == "photo":
                fid = current_media.get("file_id")
                if fid:
                    current_photo_file_ids.append(str(fid))
                    current_photo_marker = f"[TELEGRAM_PHOTO_FILE_ID] {fid}"

            selected_lines: list[str] = []
            telegram_photo_file_ids: list[str] = list(current_photo_file_ids)
            if not urls or explicit_multi_context:
                ranked = rank_messages(question_no_urls, recent_msgs)
                top = ranked[: cfg.max_messages_for_ask]

                total_chars = 0
                for m in top:
                    line = message_to_excerpt(m)
                    extra = len(line) + 1
                    if total_chars + extra > cfg.max_context_chars:
                        break
                    selected_lines.append(line)
                    total_chars += extra
                    if (
                        m.media
                        and m.media.get("type") == "photo"
                        and m.media.get("file_id")
                    ):
                        telegram_photo_file_ids.append(str(m.media.get("file_id")))

            # Fetch links (on-demand) so all subsequent buttons use the enriched context.
            link_blocks: list[str] = []
            drive_image_urls: list[str] = []
            drive_image_data_urls: list[str] = []
            if urls:
                sem_links: int = max(1, cfg.max_links)
                sem = asyncio.Semaphore(sem_links)
                async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
                    link_texts: list[tuple[str, str]] = []

                    async def fetch_one(u: str) -> None:
                        async with sem:
                            eff_url, text = await fetch_url_text(
                                u,
                                client=client,
                                timeout_s=cfg.url_fetch_timeout_s,
                                max_chars=cfg.max_link_chars,
                            )
                            if text:
                                link_texts.append((eff_url, text))

                    await asyncio.gather(*(fetch_one(u) for u in urls[: cfg.max_links]))

                # Add extracted link content as extra context blocks.
                for eff_url, text in link_texts:
                        link_blocks.append(f"[LINK] {eff_url}: {text}")

                # If any of the URLs are Google Drive folders, also extract images.
                # We pass image URLs directly to the LLM.
                folder_images: list[str] = []
                for u in urls[: cfg.max_links]:
                    imgs = await fetch_google_drive_folder_image_urls(
                        u,
                        client=client,
                        timeout_s=cfg.url_fetch_timeout_s,
                        max_images=3,
                    )
                    folder_images.extend(imgs)
                # de-dupe
                seen_urls: set[str] = set()
                for img in folder_images:
                    if img not in seen_urls:
                        drive_image_urls.append(img)
                        seen_urls.add(img)

                # Convert Drive image URLs to base64 data URLs so the LLM doesn't need to fetch externally.
                for img_url in drive_image_urls[:3]:
                    try:
                        r_img = await client.get(img_url, timeout=cfg.url_fetch_timeout_s)
                        ct = (r_img.headers.get("content-type") or "").split(";")[0].strip().lower()
                        if not ct.startswith("image/"):
                            continue
                        b = r_img.content or b""
                        if not b:
                            continue
                        b64 = base64.b64encode(b).decode("ascii")
                        drive_image_data_urls.append(f"data:{ct};base64,{b64}")
                    except Exception:
                        continue

                # Put image markers into context so button-callback can reuse them.
                for img in drive_image_urls:
                    link_blocks.append(f"[IMAGE] {img}")

                if strict_link_only:
                    selected_lines = link_blocks
                else:
                    selected_lines.extend(link_blocks)

            if strict_link_only and not selected_lines:
                # Link mode, but we failed to extract content: ask user for export/share mode.
                selected_lines = [
                    f"[LINK] {urls[0]}: Не удалось извлечь текст/CSV из ссылки. Сделай экспорт (CSV/публичный просмотр) или дай ссылку на выгрузку доступную без логина."
                ]

            # If user sent a photo with mention, always include it in context too.
            if current_photo_marker:
                # Keep within max_context_chars budget.
                approx_total = sum(len(s) + 1 for s in selected_lines)
                if approx_total + len(current_photo_marker) <= cfg.max_context_chars:
                    selected_lines.append(current_photo_marker)

            if project_lines:
                selected_lines = prepend_project_context(
                    project_lines=project_lines,
                    context_lines=selected_lines,
                    max_chars=cfg.max_context_chars,
                )

            # Convert Telegram photo file_ids to base64 data URLs (for vision).
            telegram_image_data_urls: list[str] = []
            if telegram_photo_file_ids:
                # Limit amount for cost/latency.
                for fid in telegram_photo_file_ids[:2]:
                    try:
                        telegram_file = await bot.get_file(fid)
                        fp = getattr(telegram_file, "file_path", "") or ""
                        # Guess mime by extension.
                        ext = fp.split(".")[-1].lower() if "." in fp else ""
                        mime = "image/jpeg"
                        if ext in ["png"]:
                            mime = "image/png"
                        elif ext in ["jpg", "jpeg"]:
                            mime = "image/jpeg"
                        # Download bytes.
                        bio = await bot.download(fid)
                        b = bio.getvalue() if hasattr(bio, "getvalue") else b""
                        if b:
                            b64 = base64.b64encode(b).decode("ascii")
                            telegram_image_data_urls.append(
                                f"data:{mime};base64,{b64}"
                            )
                    except Exception:
                        continue

            # Important: update session question without URL noise.
            effective_question = question_no_urls
            if cfg.enable_auto_response_mode:
                if llm is not None:
                    response_mode = llm.classify_response_mode(question=effective_question)
                else:
                    response_mode = classify_response_mode(effective_question)
            else:
                response_mode = "structured_marketing"
            mode_prefix = f"Режим ответа: {_response_mode_label(response_mode)}\n\n"

            # If we have extracted images (Drive and/or Telegram), prefer multimodal analysis.
            has_images = (len(drive_image_data_urls) + len(telegram_image_data_urls)) > 0

            phase = "create_ask_session"
            session_id = await db.create_ask_session(
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                question=effective_question,
                selected_lines=selected_lines,
            )
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H3_db_or_context",
                location="bot.py:after_create_ask_session",
                message="ask session created",
                data={
                    "session_id": session_id,
                    "selected_lines_count": len(selected_lines),
                    "has_images": has_images,
                    "urls_count": len(urls),
                },
            )
            # #endregion
            action_kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="Выдать цифры",
                            callback_data=f"act:nums:{session_id}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            text="Предложения от ИИ",
                            callback_data=f"act:ai:{session_id}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            text="Расставить задачи",
                            callback_data=f"act:tasks:{session_id}",
                        )
                    ],
                ]
            )

            latest_summary_text = None
            if not strict_link_only:
                latest = await db.get_latest_daily_summary(chat_id=message.chat.id)
                latest_summary_text = latest[0] if latest else None

            if llm is None:
                # #region agent log
                _debug_log(
                    run_id=debug_run_id,
                    hypothesis_id="H4_llm_unavailable",
                    location="bot.py:llm_is_none_fallback",
                    message="llm is unavailable, using local fallback",
                    data={
                        "selected_lines_count": len(selected_lines),
                        "question_len": len(effective_question),
                    },
                )
                # #endregion
                out = mode_prefix + build_freeform_answer(effective_question, selected_lines)
                sent = await message.answer(
                    out, reply_markup=action_kb, disable_web_page_preview=True
                )
                await db.map_bot_message_to_session(
                    chat_id=message.chat.id,
                    bot_message_id=sent.message_id,
                    session_id=session_id,
                )
                return

            try:
                phase = "llm_call"
                llm_call_started_ms = int(time.time() * 1000)
                # #region agent log
                _debug_log(
                    run_id=debug_run_id,
                    hypothesis_id="H2_llm_latency_or_hang",
                    location="bot.py:before_llm_call",
                    message="starting llm request",
                    data={
                        "has_images": has_images,
                        "selected_lines_count": len(selected_lines),
                        "question_len": len(effective_question),
                    },
                )
                # #endregion
                if has_images:
                    answer = await llm.answer_question_with_images(
                        chat_id=message.chat.id,
                        question=effective_question,
                        latest_daily_summary=latest_summary_text,
                        context_messages=selected_lines,
                        image_urls=(drive_image_data_urls + telegram_image_data_urls),
                        project_context=project_context_text,
                        response_mode=response_mode,
                    )
                else:
                    answer = await llm.answer_question(
                        chat_id=message.chat.id,
                        question=effective_question,
                        latest_daily_summary=latest_summary_text,
                        context_messages=selected_lines,
                        project_context=project_context_text,
                        response_mode=response_mode,
                    )
                # #region agent log
                _debug_log(
                    run_id=debug_run_id,
                    hypothesis_id="H2_llm_latency_or_hang",
                    location="bot.py:after_llm_call",
                    message="llm request finished",
                    data={
                        "duration_ms": int(time.time() * 1000) - llm_call_started_ms,
                        "answer_len": len((answer.text or "").strip()),
                    },
                )
                # #endregion
            except Exception as e:
                err_low = str(e).lower()
                print(f"[answer_question] OpenAI error: {type(e).__name__}: {e}", flush=True)
                # #region agent log
                _debug_log(
                    run_id=debug_run_id,
                    hypothesis_id="H4_llm_error_path",
                    location="bot.py:llm_except",
                    message="llm exception caught",
                    data={
                        "exc_type": type(e).__name__,
                        "exc_str": str(e)[:300],
                        "err_low_snippet": err_low[:200],
                        "phase": phase,
                    },
                )
                # #endregion
                # Quota: fall back to rule-based answer instead of failing the chat.
                if (
                    "429" in err_low
                    or "insufficient_quota" in err_low
                    or "quota" in err_low
                    or "rate_limit" in err_low
                    or "unsupported_country_region_territory" in err_low
                    or ("request_forbidden" in err_low and "country" in err_low)
                ):
                    print(
                        "[answer_question] fallback_rule_based: llm region/quota restriction",
                        flush=True,
                    )
                    out = mode_prefix + build_freeform_answer(
                        effective_question, selected_lines
                    )
                    sent = await message.answer(
                        out, reply_markup=action_kb, disable_web_page_preview=True
                    )
                    await db.map_bot_message_to_session(
                        chat_id=message.chat.id,
                        bot_message_id=sent.message_id,
                        session_id=session_id,
                    )
                    return
                hint = _openai_failure_reply(e)
                if hint:
                    print("[answer_question] hint_reply_to_user: openai_failure_reply", flush=True)
                    sent = await message.answer(
                        mode_prefix + hint,
                        reply_markup=action_kb,
                        disable_web_page_preview=True,
                    )
                    await db.map_bot_message_to_session(
                        chat_id=message.chat.id,
                        bot_message_id=sent.message_id,
                        session_id=session_id,
                    )
                    return
                raise

            used_mode = answer.response_mode or response_mode
            out = _sanitize_llm_answer((answer.text or "").strip())
            out = f"Режим ответа: {_response_mode_label(used_mode)}\n\n{out}"
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H3_sanitize_forced_fallback",
                location="bot.py:after_sanitize_llm_answer",
                message="llm answer sanitized",
                data={
                    "raw_answer_len": len((answer.text or "").strip()),
                    "final_answer_len": len(out),
                    "used_fallback": out != (answer.text or "").strip(),
                },
            )
            # #endregion

            if len(out) > 3200:
                out = out[:3190] + "\n…"
            sent = await message.answer(
                out, reply_markup=action_kb, disable_web_page_preview=True
            )
            await db.map_bot_message_to_session(
                chat_id=message.chat.id,
                bot_message_id=sent.message_id,
                session_id=session_id,
            )
        except Exception as e:
            print(f"[answer_question] fatal: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H5_outer_fatal",
                location="bot.py:outer_except",
                message="fatal exception caught",
                data={
                    "exc_type": type(e).__name__,
                    "exc_str": str(e)[:300],
                    "action_kb_set": action_kb is not None,
                    "phase": phase,
                },
            )
            # #endregion
            out = "\n".join(
                [
                    "📊 Ошибка:",
                    "— Не удалось обработать запрос.",
                    "— Проверьте настройки и попробуйте ещё раз.",
                    "— Уточните вопрос, чтобы ответ был точнее.",
                    "⚠️ Вывод: Повторите запрос.",
                    f"— Технически: {type(e).__name__}",
                ]
            )
            if action_kb is not None:
                await message.answer(out, reply_markup=action_kb, disable_web_page_preview=True)
            else:
                await message.answer(out, disable_web_page_preview=True)
        finally:
            try:
                typing_task.cancel()
            except Exception:
                pass

    # Follow-up: if someone replies to a bot message, answer using the same session context.
    @router.message(
        lambda m: (
            m.chat is not None
            and m.reply_to_message is not None
            and m.reply_to_message.from_user is not None
            and m.reply_to_message.from_user.id == bot_id
            and (m.text or m.caption)
        )
    )
    async def on_reply_to_bot(message: Message) -> None:
        if message.chat is None or message.reply_to_message is None:
            return
        if not (message.text or message.caption):
            return

        bot_msg_id = message.reply_to_message.message_id
        session_id = await db.get_session_by_bot_message(
            chat_id=message.chat.id, bot_message_id=bot_msg_id
        )
        if session_id is None:
            return

        session = await db.get_ask_session_by_id(session_id=session_id)
        if session is None:
            return

        question = (message.text or message.caption or "").strip()
        # Strip /ask if present
        if question.lower().startswith("/ask"):
            parts = question.split(maxsplit=1)
            question = parts[1].strip() if len(parts) > 1 else ""
        # Strip mention if they included it
        question = strip_bot_mention(question)

        if not question:
            # If user replied with only media (no caption text), still analyze it.
            media = extract_media_metadata(message)
            if media and media.get("type") == "photo":
                question = "Проанализируй это изображение и извлеки ключевые факты."
            else:
                return

        if _is_help_intent(question):
            await message.answer(build_help_redirect(bot_username), disable_web_page_preview=True)
            return

        # Use session context; do not re-rank other chat messages.
        selected_lines = session[1]
        effective_question = question
        active_project = await db.get_active_project(chat_id=message.chat.id)
        project_context_text: str | None = None
        if active_project is not None:
            project_memory = await db.get_project_memory(
                chat_id=message.chat.id, project_id=active_project.id
            )
            project_lines = build_project_memory_lines(project_memory)
            if project_lines:
                project_context_text = budgeted_join(
                    project_lines, max_chars=cfg.max_project_memory_chars
                )
                selected_lines = prepend_project_context(
                    project_lines=project_lines,
                    context_lines=selected_lines,
                    max_chars=cfg.max_context_chars,
                )

        # Optional: if reply includes URLs, fetch and append.
        urls = extract_urls(question)
        if urls:
            question_no_urls = question
            for u in urls:
                question_no_urls = question_no_urls.replace(u, "").strip()
            effective_question = question_no_urls or question

            sem_links: int = max(1, cfg.max_links)
            sem = asyncio.Semaphore(sem_links)
            async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
                link_texts: list[tuple[str, str]] = []

                async def fetch_one(u: str) -> None:
                    async with sem:
                        eff_url, text = await fetch_url_text(
                            u,
                            client=client,
                            timeout_s=cfg.url_fetch_timeout_s,
                            max_chars=cfg.max_link_chars,
                        )
                        if text:
                            link_texts.append((eff_url, text))

                await asyncio.gather(*(fetch_one(u) for u in urls[: cfg.max_links]))
            for eff_url, text in link_texts:
                selected_lines.append(f"[LINK] {eff_url}: {text}")

        latest_summary_text = None
        if not extract_urls(question):
            latest = await db.get_latest_daily_summary(chat_id=message.chat.id)
            latest_summary_text = latest[0] if latest else None

        if llm is None:
            response_mode = (
                classify_response_mode(effective_question)
                if cfg.enable_auto_response_mode
                else "structured_marketing"
            )
            out = (
                f"Режим ответа: {_response_mode_label(response_mode)}\n\n"
                + build_freeform_answer(effective_question, selected_lines)
            )
            await message.answer(out, disable_web_page_preview=True)
            return

        async def _typing_loop() -> None:
            while True:
                try:
                    await bot.send_chat_action(message.chat.id, action="typing")
                except Exception:
                    pass
                await asyncio.sleep(4)

        typing_task = asyncio.create_task(_typing_loop())
        try:
            response_mode = (
                llm.classify_response_mode(question=effective_question)
                if cfg.enable_auto_response_mode
                else "structured_marketing"
            )
            # If this reply includes a Telegram photo, analyze it with vision.
            reply_media = extract_media_metadata(message)
            image_urls: list[str] = []
            if reply_media and reply_media.get("type") == "photo" and reply_media.get("file_id"):
                # Download photo and convert to data URL.
                try:
                    fid = str(reply_media.get("file_id"))
                    telegram_file = await bot.get_file(fid)
                    fp = getattr(telegram_file, "file_path", "") or ""
                    ext = fp.split(".")[-1].lower() if "." in fp else ""
                    mime = "image/jpeg"
                    if ext in ["png"]:
                        mime = "image/png"
                    elif ext in ["webp"]:
                        mime = "image/webp"
                    bio = await bot.download(fid)
                    b = bio.getvalue() if hasattr(bio, "getvalue") else b""
                    if b:
                        b64 = base64.b64encode(b).decode("ascii")
                        image_urls.append(f"data:{mime};base64,{b64}")
                except Exception:
                    image_urls = []

            if image_urls:
                answer = await llm.answer_question_with_images(
                    chat_id=message.chat.id,
                    question=effective_question,
                    latest_daily_summary=latest_summary_text,
                    context_messages=selected_lines,
                    image_urls=image_urls[:1],
                    project_context=project_context_text,
                    response_mode=response_mode,
                )
            else:
                answer = await llm.answer_question(
                    chat_id=message.chat.id,
                    question=effective_question,
                    latest_daily_summary=latest_summary_text,
                    context_messages=selected_lines,
                    project_context=project_context_text,
                    response_mode=response_mode,
                )
            candidate = (answer.text or "").strip()
            # Local format sanity check; fallback if it looks unsafe.
            low = (candidate or "").lower()
            if "из предоставленного контекста" in low or "возможно" in low or "вероятно" in low:
                out = build_freeform_answer(effective_question, selected_lines)
            else:
                out = candidate
            used_mode = answer.response_mode or response_mode
            out = f"Режим ответа: {_response_mode_label(used_mode)}\n\n{out}"
            if len(out) > 3200:
                out = out[:3190] + "\n…"
            await message.answer(out, disable_web_page_preview=True)
        except Exception:
            response_mode = (
                classify_response_mode(effective_question)
                if cfg.enable_auto_response_mode
                else "structured_marketing"
            )
            out = (
                f"Режим ответа: {_response_mode_label(response_mode)}\n\n"
                + build_freeform_answer(effective_question, selected_lines)
            )
            await message.answer(out, disable_web_page_preview=True)
        finally:
            try:
                typing_task.cancel()
            except Exception:
                pass

    @router.message(Command("project_new"))
    async def project_new(message: Message) -> None:
        if message.chat is None:
            return
        args = _parse_command_args(message.text)
        if not args:
            await message.reply(
                "Формат: /project_new <название проекта>\n"
                "Пример: /project_new Контекст РСЯ Q2"
            )
            return
        project_name = args.strip()
        if len(project_name) > 80:
            await message.reply("Название слишком длинное. Лимит: 80 символов.")
            return

        base_slug = _slugify_project_name(project_name)
        slug = base_slug
        suffix = 2
        while True:
            existing = await db.get_project_by_ref(chat_id=message.chat.id, project_ref=slug)
            if existing is None:
                break
            slug = f"{base_slug}-{suffix}"
            suffix += 1

        project_id = await db.create_project(chat_id=message.chat.id, slug=slug, name=project_name)
        await db.set_active_project(chat_id=message.chat.id, project_id=project_id)
        await message.reply(
            f"Проект создан и выбран активным.\n"
            f"id: {project_id}\nslug: {slug}\nname: {project_name}"
        )

    @router.message(Command("project_list"))
    async def project_list(message: Message) -> None:
        if message.chat is None:
            return
        projects = await db.list_projects(chat_id=message.chat.id)
        if not projects:
            await message.reply(
                "Пока нет проектов. Создай первый:\n/project_new <название проекта>"
            )
            return
        active_project = await db.get_active_project(chat_id=message.chat.id)
        lines = ["Проекты в этом чате:"]
        for p in projects:
            marker = " (active)" if active_project and active_project.id == p.id else ""
            lines.append(f"— {p.id} | {p.slug} | {p.name}{marker}")
        await message.reply("\n".join(lines))

    @router.message(Command("project_use"))
    async def project_use(message: Message) -> None:
        if message.chat is None:
            return
        ref = _parse_command_args(message.text)
        if not ref:
            await message.reply(
                "Формат: /project_use <id|slug>\n"
                "Пример: /project_use 3\nили /project_use context-rsya-q2"
            )
            return
        project = await db.get_project_by_ref(chat_id=message.chat.id, project_ref=ref)
        if project is None:
            await message.reply("Проект не найден в этом чате.")
            return
        ok = await db.set_active_project(chat_id=message.chat.id, project_id=project.id)
        if not ok:
            await message.reply("Не удалось выбрать проект. Попробуй ещё раз.")
            return
        await message.reply(f"Активный проект: {project.name} ({project.slug})")

    @router.message(Command("project_show"))
    async def project_show(message: Message) -> None:
        if message.chat is None:
            return
        active_project = await db.get_active_project(chat_id=message.chat.id)
        if active_project is None:
            await message.reply(
                "Активный проект не выбран.\n"
                "Используй /project_use <id|slug> или создай /project_new <название>."
            )
            return

        memory = await db.get_project_memory(
            chat_id=message.chat.id, project_id=active_project.id
        )
        lines = [
            "Активный проект:",
            f"id: {active_project.id}",
            f"slug: {active_project.slug}",
            f"name: {active_project.name}",
            "",
            "Карточка проекта:",
        ]

        added_sections = 0
        for section in _PROJECT_SECTIONS_ORDER:
            content = (memory.get(section) or "").strip()
            if not content:
                continue
            lines.append(f"— {section}: {content}")
            added_sections += 1

        # Also show custom sections if they were ever added directly in DB.
        for section, content in memory.items():
            if section in _PROJECT_SECTIONS:
                continue
            c = (content or "").strip()
            if not c:
                continue
            lines.append(f"— {section}: {c}")
            added_sections += 1

        if added_sections == 0:
            lines.append("Пока пусто. Заполни секции через /project_set.")

        await message.reply("\n".join(lines), disable_web_page_preview=True)

    @router.message(Command("project_set"))
    async def project_set(message: Message) -> None:
        if message.chat is None:
            return
        raw = _parse_command_args(message.text)
        if not raw:
            await message.reply(
                "Формат: /project_set <section> <content>\n"
                "Разделы: brief, kpi, constraints, audience, hypotheses, notes"
            )
            return

        pieces = raw.split(maxsplit=1)
        section = pieces[0].strip().lower()
        content = pieces[1].strip() if len(pieces) > 1 else ""
        if section not in _PROJECT_SECTIONS:
            await message.reply(
                "Неизвестный раздел.\nРазрешено: brief, kpi, constraints, audience, hypotheses, notes"
            )
            return
        if not content:
            await message.reply("Пустое значение. Добавь текст после названия раздела.")
            return
        if len(content) > 1200:
            await message.reply("Слишком длинный текст. Лимит: 1200 символов.")
            return

        active_project = await db.get_active_project(chat_id=message.chat.id)
        if active_project is None:
            await message.reply(
                "Сначала выбери проект.\n"
                "1) /project_new <название>\n2) или /project_use <id|slug>"
            )
            return
        ok = await db.upsert_project_memory(
            chat_id=message.chat.id,
            project_id=active_project.id,
            section=section,
            content=content,
        )
        if not ok:
            await message.reply("Не удалось сохранить секцию памяти проекта.")
            return
        await message.reply(
            f"Сохранено в проект `{active_project.slug}`:\n"
            f"— {section}: {content[:160]}{'…' if len(content) > 160 else ''}",
        )

    # Store messages from people into SQLite.
    @router.message()
    async def store_incoming(message: Message) -> None:
        if message.chat is None or message.from_user is None:
            return

        if message.from_user.id == bot_id:
            return

        # Avoid storing bot commands themselves.
        if message.text and message.text.strip().startswith("/"):
            # #region agent log
            _agent_diag(
                run_id=f"msg-{message.chat.id}-{message.message_id}",
                hypothesis_id="H1",
                location="bot.py:store_incoming_command_ignored",
                message="command message reached generic handler and ignored for storage",
                data={"raw_text": message.text.strip()},
            )
            # #endregion
            return

        text = message.text or None
        caption = getattr(message, "caption", None) or None
        media = extract_media_metadata(message)

        if not (text or caption or media):
            return

        # Ответы только по @упоминанию, чтобы бот не мешал коллегам в чате.
        if is_bot_mentioned(message):
            raw = message.text or message.caption or ""
            stripped = strip_bot_mention(raw)
            t = (stripped or "").strip().lower()

            if _is_maxon_joke_request(stripped):
                await message.reply(_MAXON_JOKE_REPLY)
            else:
                # Allow either "@bot /ask <q>" or "@bot <q>" formats.
                if t.startswith("/ask"):
                    parts = stripped.split(maxsplit=1)
                    question = parts[1].strip() if len(parts) > 1 else ""
                else:
                    question = stripped
                await answer_question(message, question)

        created_at_ts = int(message.date.timestamp())

        stored = StoredMessage(
            chat_id=message.chat.id,
            message_id=message.message_id,
            from_user_id=message.from_user.id,
            from_username=message.from_user.username,
            created_at_ts=created_at_ts,
            text=text,
            caption=caption,
            media=media,
        )
        await db.insert_message(stored)

    # Context-aware Q&A.
    @router.message(Command("ask"))
    async def ask(message: Message) -> None:
        if message.chat is None:
            return
        # Require @mention as well for /ask.
        # #region agent log
        _agent_diag(
            run_id=f"msg-{message.chat.id}-{message.message_id}",
            hypothesis_id="H2",
            location="bot.py:ask_handler_entry",
            message="ask command handler invoked",
            data={
                "raw_text": message.text or "",
                "is_mentioned": is_bot_mentioned(message),
            },
        )
        # #endregion
        if not is_bot_mentioned(message):
            # #region agent log
            _agent_diag(
                run_id=f"msg-{message.chat.id}-{message.message_id}",
                hypothesis_id="H2",
                location="bot.py:ask_handler_reject_no_mention",
                message="ask rejected because bot mention missing",
                data={"raw_text": message.text or ""},
            )
            # #endregion
            return

        raw = (message.text or "").strip()
        parts = raw.split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else ""
        await answer_question(message, question)

    @router.callback_query(lambda c: c.data and c.data.startswith("act:"))
    async def on_action(callback: CallbackQuery) -> None:
        # Always answer callback query to avoid "Loading..." forever.
        answered = False
        try:
            if callback.message is None or callback.from_user is None:
                return

            data = callback.data or ""
            parts = data.split(":", 2)
            if len(parts) != 3:
                await callback.answer("Некорректная команда.", show_alert=False)
                answered = True
                return

            _, kind, session_id_raw = parts
            try:
                session_id = int(session_id_raw)
            except Exception:
                await callback.answer("Некорректная сессия.", show_alert=False)
                answered = True
                return

            session = await db.get_ask_session(
                session_id=session_id,
                chat_id=callback.message.chat.id,
                user_id=callback.from_user.id,
            )
            if session is None:
                await callback.answer("Нет доступа к этим данным.", show_alert=False)
                answered = True
                return

            question, selected_lines = session

            if kind == "nums":
                # Prefer deterministic KPI extraction from link CSV/text.
                det = build_numbers_from_link_context(question, selected_lines)
                if det:
                    out = det
                else:
                    if llm is None:
                        out = build_numbers_answer(question, selected_lines)
                    else:
                        try:
                            res = await asyncio.wait_for(
                                llm.extract_key_numbers(
                                    question=question,
                                    context_messages=selected_lines,
                                ),
                                timeout=25,
                            )
                            out = res.text
                        except Exception:
                            out = build_numbers_answer(question, selected_lines)
                # Post-verify: output numbers must exist in selected_lines.
                try:
                    import re as _re

                    out_low = out or ""
                    # Exclude media/file markers from the "number exists" check.
                    ctx_text = "\n".join(
                        s
                        for s in selected_lines
                        if (not s.startswith("[IMAGE] "))
                        and ("[TELEGRAM_PHOTO_FILE_ID]" not in s)
                    )

                    def norm_digits(s: str) -> str:
                        # Keep digits only; remove separators.
                        return _re.sub(r"\D+", "", s or "")

                    ctx_numbers = set()
                    for m in _re.finditer(r"\d[\d\s.,]*\d|\d+", ctx_text):
                        n = norm_digits(m.group(0))
                        if n and len(n) >= 3:
                            ctx_numbers.add(n)

                    out_numbers = []
                    ok = True
                    for m in _re.finditer(r"\d[\d\s.,]*\d|\d+", out_low):
                        n = norm_digits(m.group(0))
                        if n and len(n) >= 3:
                            out_numbers.append(n)
                            if n not in ctx_numbers:
                                ok = False
                                break

                    if not ok:
                        # Retry with stricter prompt by re-running extraction.
                        res2 = await llm.extract_key_numbers(
                            question=question
                            + "\nВАЖНО: нельзя выводить числа, которых нет точь-в-точь в таблице.",
                            context_messages=selected_lines,
                        )
                        out = res2.text
                except Exception:
                    pass
            elif kind == "ai":
                if llm is None:
                    out = build_marketing_suggestions_fallback(question, selected_lines)
                else:
                    # If selected_lines include extracted images, use multimodal suggestions.
                    image_urls: list[str] = []
                    telegram_photo_file_ids: list[str] = []
                    for l in selected_lines:
                        if l.startswith("[IMAGE] "):
                            image_urls.append(l[len("[IMAGE] ") :].strip())
                        if "[TELEGRAM_PHOTO_FILE_ID]" in l:
                            token = "[TELEGRAM_PHOTO_FILE_ID]"
                            after = l.split(token, 1)[1].strip()
                            fid = after.split()[0] if after else ""
                            if fid:
                                telegram_photo_file_ids.append(fid)
                    try:
                        if image_urls:
                            # Convert to data URLs (if they are remote URLs).
                            # If they are already data URLs, reuse as-is.
                            data_urls: list[str] = []
                            async with httpx.AsyncClient(
                                headers={"User-Agent": "Mozilla/5.0"}
                            ) as client2:
                                for u in image_urls[:3]:
                                    if u.startswith("data:image/"):
                                        data_urls.append(u)
                                        continue
                                    try:
                                        r_img = await client2.get(
                                            u, timeout=cfg.url_fetch_timeout_s
                                        )
                                        ct = (
                                            (r_img.headers.get("content-type") or "")
                                            .split(";")[0]
                                            .strip()
                                            .lower()
                                        )
                                        if ct.startswith("image/") and r_img.content:
                                            b64 = base64.b64encode(r_img.content).decode(
                                                "ascii"
                                            )
                                            data_urls.append(
                                                f"data:{ct};base64,{b64}"
                                            )
                                    except Exception:
                                        continue
                            # Convert Telegram photos to data URLs.
                            for fid in telegram_photo_file_ids[:2]:
                                try:
                                    tfile = await bot.get_file(fid)
                                    fp = getattr(tfile, "file_path", "") or ""
                                    ext = fp.split(".")[-1].lower() if "." in fp else ""
                                    mime = "image/jpeg"
                                    if ext in ["png"]:
                                        mime = "image/png"
                                    elif ext in ["webp"]:
                                        mime = "image/webp"
                                    bio = await bot.download(fid)
                                    b = bio.getvalue() if hasattr(bio, "getvalue") else b""
                                    if b:
                                        b64 = base64.b64encode(b).decode("ascii")
                                        data_urls.append(f"data:{mime};base64,{b64}")
                                except Exception:
                                    continue
                            res = await asyncio.wait_for(
                                llm.marketing_suggestions_with_images(
                                    question=question,
                                    context_messages=selected_lines,
                                    image_urls=data_urls[:3],
                                ),
                                timeout=30,
                            )
                        elif telegram_photo_file_ids:
                            # Only Telegram photos, no Drive images.
                            data_urls: list[str] = []
                            for fid in telegram_photo_file_ids[:2]:
                                try:
                                    tfile = await bot.get_file(fid)
                                    fp = getattr(tfile, "file_path", "") or ""
                                    ext = fp.split(".")[-1].lower() if "." in fp else ""
                                    mime = "image/jpeg"
                                    if ext in ["png"]:
                                        mime = "image/png"
                                    elif ext in ["webp"]:
                                        mime = "image/webp"
                                    bio = await bot.download(fid)
                                    b = bio.getvalue() if hasattr(bio, "getvalue") else b""
                                    if b:
                                        b64 = base64.b64encode(b).decode("ascii")
                                        data_urls.append(
                                            f"data:{mime};base64,{b64}"
                                        )
                                except Exception:
                                    continue
                            res = await asyncio.wait_for(
                                llm.marketing_suggestions_with_images(
                                    question=question,
                                    context_messages=selected_lines,
                                    image_urls=data_urls[:3],
                                ),
                                timeout=30,
                            )
                        else:
                            res = await asyncio.wait_for(
                                llm.marketing_suggestions(
                                    question=question,
                                    context_messages=selected_lines,
                                ),
                                timeout=25,
                            )
                        out = res.text
                    except Exception:
                        out = build_marketing_suggestions_fallback(question, selected_lines)
            elif kind == "tasks":
                now = datetime.now().astimezone()
                min_ts = int((now - timedelta(days=7)).timestamp())
                recent_msgs = await db.get_recent_messages_for_chat(
                    chat_id=callback.message.chat.id, min_ts=min_ts, limit=250
                )
                recent_excerpts = [message_to_excerpt(m) for m in recent_msgs]
                out = build_tasks_answer(question, recent_excerpts)
            else:
                await callback.answer("Неизвестное действие.", show_alert=False)
                answered = True
                return

            if len(out) > 3500:
                out = out[:3490] + "\n…"

            await callback.answer()
            answered = True
            await callback.message.answer(out, disable_web_page_preview=True)
        except Exception:
            try:
                if callback:
                    await callback.answer("Ошибка обработки. Попробуйте ещё раз.", show_alert=False)
            except Exception:
                pass

    # Background daily summary task.
    params = DailyRunParams(
        daily_hour=cfg.daily_hour,
        daily_minute=cfg.daily_minute,
        timezone_name=cfg.daily_timezone,
        max_context_chars=cfg.max_context_chars,
        messages_limit_for_summary=800,
        lookback_hours=24,
    )

    allowed = cfg.allowed_chat_ids or []
    daily_task = asyncio.create_task(
        daily_summary_loop(
            bot=bot,
            db=db,
            llm=llm,
            allowed_chat_ids=allowed,
            params=params,
        )
    )

    try:
        await dp.start_polling(bot)
    finally:
        daily_task.cancel()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())

