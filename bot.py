from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from config import load_config
from daily_loop import DailyRunParams, daily_summary_loop
from db import Database, StoredMessage
from llm import LLMClient
from message_extract import extract_media_metadata
from retrieval import message_to_excerpt, rank_messages
from formatter import (
    build_numbers_answer,
    build_marketing_suggestions_fallback,
    build_tasks_answer,
    build_fallback_answer,
    build_freeform_answer,
)
from links import extract_urls, fetch_url_text, fetch_google_drive_folder_image_urls
import httpx
from table_kpis import build_numbers_from_link_context


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
        )

    me = await bot.get_me()
    bot_id = me.id
    bot_username = (me.username or "").strip().lower()

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

    async def answer_question(message: Message, question: str) -> None:
        question = (question or "").strip()
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

        def _sanitize_llm_answer(candidate: str) -> str:
            # Only remove clearly forbidden phrases; formatting is left to the model.
            bad_phrases = ["из предоставленного контекста", "возможно", "вероятно"]
            low = (candidate or "").lower()
            if any(p in low for p in bad_phrases):
                return build_freeform_answer(question, selected_lines)
            return candidate

        session_id: int | None = None
        action_kb: InlineKeyboardMarkup | None = None

        try:
            await message.reply("Thinking…")
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

            # If we have extracted images (Drive and/or Telegram), prefer multimodal analysis.
            has_images = (len(drive_image_data_urls) + len(telegram_image_data_urls)) > 0

            session_id = await db.create_ask_session(
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                question=effective_question,
                selected_lines=selected_lines,
            )
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
                out = build_freeform_answer(effective_question, selected_lines)
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
                if has_images:
                    answer = await llm.answer_question_with_images(
                        question=effective_question,
                        latest_daily_summary=latest_summary_text,
                        context_messages=selected_lines,
                        image_urls=(drive_image_data_urls + telegram_image_data_urls),
                    )
                else:
                    answer = await llm.answer_question(
                        chat_id=message.chat.id,
                        question=effective_question,
                        latest_daily_summary=latest_summary_text,
                        context_messages=selected_lines,
                    )
            except Exception as e:
                # If OpenAI quota is exhausted, fall back to excerpts
                # instead of returning an error to the chat.
                msg = str(e).lower()
                if "429" in msg or "insufficient_quota" in msg or "quota" in msg:
                    out = build_freeform_answer(effective_question, selected_lines)
                    sent = await message.answer(
                        out, reply_markup=action_kb, disable_web_page_preview=True
                    )
                    await db.map_bot_message_to_session(
                        chat_id=message.chat.id,
                        bot_message_id=sent.message_id,
                        session_id=session_id,
                    )
                    return
                raise

            out = _sanitize_llm_answer((answer.text or "").strip())

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
            out = "\n".join(
                [
                    "📊 Ошибка:",
                    "— Не удалось обработать запрос.",
                    "— Проверьте настройки и попробуйте ещё раз.",
                    "— Уточните вопрос, чтобы ответ был точнее.",
                    "⚠️ Вывод: Повторите запрос.",
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

        # Use session context; do not re-rank other chat messages.
        selected_lines = session[1]
        effective_question = question

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
            out = build_freeform_answer(effective_question, selected_lines)
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
                )
            else:
                answer = await llm.answer_question(
                    chat_id=message.chat.id,
                    question=effective_question,
                    latest_daily_summary=latest_summary_text,
                    context_messages=selected_lines,
                )
            candidate = (answer.text or "").strip()
            # Local format sanity check; fallback if it looks unsafe.
            low = (candidate or "").lower()
            if "из предоставленного контекста" in low or "возможно" in low or "вероятно" in low:
                out = build_freeform_answer(effective_question, selected_lines)
            else:
                out = candidate
            if len(out) > 3200:
                out = out[:3190] + "\n…"
            await message.answer(out, disable_web_page_preview=True)
        except Exception:
            out = build_freeform_answer(effective_question, selected_lines)
            await message.answer(out, disable_web_page_preview=True)
        finally:
            try:
                typing_task.cancel()
            except Exception:
                pass

    # Store messages from people into SQLite.
    @router.message()
    async def store_incoming(message: Message) -> None:
        if message.chat is None or message.from_user is None:
            return

        if message.from_user.id == bot_id:
            return

        # Avoid storing bot commands themselves.
        if message.text and message.text.strip().startswith("/"):
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
        if not is_bot_mentioned(message):
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

