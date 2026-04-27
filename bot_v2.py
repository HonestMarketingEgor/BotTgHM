from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import shutil
import time

import httpx
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import BotCommand, FSInputFile, Message

from config import load_config
from db import Database, StoredMessage
from formatter import (
    build_analysis_fallback,
    build_assistant_fallback,
    build_help_text,
    build_help_redirect,
)
from llm import LLMClient
from links import extract_urls, fetch_url_text
from message_extract import extract_media_metadata
from retrieval import message_to_excerpt, rank_messages
from vk_match_service import run_vk_match

HELP_MODE = "help_mode"
ASSISTANT_MODE = "assistant_mode"
ANALYSIS_MODE = "analysis_mode"

VALID_CHAT_MODES = {ASSISTANT_MODE, ANALYSIS_MODE}
VK_STAGE_WAIT_A = "wait_file_a"
VK_STAGE_WAIT_B = "wait_files_b"
VK_MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024
VK_SESSION_TTL_SECONDS = 2 * 3600


@dataclass
class ChatRuntimeState:
    mode: str


@dataclass
class VkMatchSession:
    stage: str
    created_at_ts: int
    work_dir: Path
    file_a_path: Path | None
    files_b_paths: list[Path]


def _is_help_intent(text: str) -> bool:
    q = (text or "").strip().lower().replace("ё", "е")
    markers = [
        "что ты умеешь",
        "что ты можешь",
        "что умеет бот",
        "что может бот",
        "что он может",
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


def _needs_chat_context(text: str) -> bool:
    q = (text or "").strip().lower().replace("ё", "е")
    markers = [
        "по чату",
        "в чате",
        "по переписке",
        "из переписки",
        "по сообщениям",
        "из контекста чата",
        "что обсуждали",
        "что было в чате",
        "сводка чата",
    ]
    return any(m in q for m in markers)


def _is_current_chat_name_intent(text: str) -> bool:
    q = (text or "").strip().lower().replace("ё", "е")
    patterns = [
        "как называется этот чат",
        "название этого чата",
        "назови этот чат",
        "какое название чата",
        "что это за чат",
    ]
    return any(p in q for p in patterns)


def _parse_cross_chat_intent(text: str) -> tuple[str, str] | None:
    q = (text or "").strip()
    low = q.lower()
    if "в чате " not in low and "по чату " not in low:
        return None

    m = re.search(
        r"(?:в|по)\s+чате\s+[\"'«]?([^\"'»?,.!]+)[\"'»]?",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    chat_name = (m.group(1) or "").strip()
    if not chat_name:
        return None

    rest = (q[: m.start()] + " " + q[m.end() :]).strip()
    rest = re.sub(r"\s+", " ", rest)
    for phrase in ["что обсуждали", "что обсуждалось", "расскажи", "подскажи"]:
        rest = re.sub(phrase, "", rest, flags=re.IGNORECASE).strip()
    topic = rest.strip(" ?!.,")
    return chat_name, topic


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

    try:
        await bot.set_my_commands(
            [
                BotCommand(command="help", description="Показать возможности бота"),
                BotCommand(command="ask", description="Анализ чата: /ask <вопрос>"),
                BotCommand(command="daily_summary", description="Сводка за 24ч по текущему чату"),
                BotCommand(command="summary", description="Короткий алиас сводки"),
                BotCommand(command="mode", description="Режим: assistant | analysis"),
                BotCommand(command="chat_info", description="Текущий chat_id и название"),
                BotCommand(command="vkmatch", description="Сопоставление лидов A vs B"),
                BotCommand(command="reset", description="Сброс режима чата"),
            ]
        )
    except Exception as e:
        print(f"[startup] set_my_commands skipped: {type(e).__name__}: {e}")

    runtime_state_by_chat: dict[int, ChatRuntimeState] = {}
    vk_sessions: dict[tuple[int, int], VkMatchSession] = {}
    vk_media_buffers: dict[tuple[int, int, str], list[Message]] = {}
    vk_media_tasks: dict[tuple[int, int, str], asyncio.Task] = {}
    default_mode = _normalize_mode(cfg.default_mode) or ASSISTANT_MODE

    def _chat_state(chat_id: int) -> ChatRuntimeState:
        st = runtime_state_by_chat.get(chat_id)
        if st is None:
            st = ChatRuntimeState(mode=default_mode)
            runtime_state_by_chat[chat_id] = st
        return st

    def _vk_key(message: Message) -> tuple[int, int] | None:
        if message.chat is None or message.from_user is None:
            return None
        return (message.chat.id, message.from_user.id)

    def _cleanup_vk_session_files(session: VkMatchSession) -> None:
        try:
            shutil.rmtree(session.work_dir, ignore_errors=True)
        except Exception:
            pass

    def _drop_vk_session(key: tuple[int, int]) -> None:
        session = vk_sessions.pop(key, None)
        if session is not None:
            _cleanup_vk_session_files(session)

    async def _download_document_to_dir(message: Message, prefix: str, max_size: int) -> tuple[Path, str]:
        if message.document is None:
            raise ValueError("Ожидался файл-документ.")
        doc = message.document
        if doc.file_size and doc.file_size > max_size:
            raise ValueError(
                f"Файл слишком большой ({doc.file_size} байт). "
                f"Максимум: {max_size} байт."
            )
        key = _vk_key(message)
        if key is None:
            raise ValueError("Не удалось определить сессию.")
        session = vk_sessions.get(key)
        if session is None:
            raise ValueError("Сессия /VkMatch не найдена. Запустите команду заново.")

        original_name = (doc.file_name or f"{prefix}_{doc.file_id}").strip()
        ext = Path(original_name).suffix.lower()
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem) or prefix
        filename = f"{prefix}_{safe_base}_{int(time.time()*1000)}{ext}"
        target = session.work_dir / filename

        bio = await bot.download(doc.file_id)
        payload = bio.getvalue() if hasattr(bio, "getvalue") else b""
        target.write_bytes(payload)
        return target, ext

    async def _run_vk_pipeline_and_send(message: Message, session: VkMatchSession) -> None:
        await message.reply("Файлы получил. Начал обработку, это может занять до пары минут.")
        output_path = session.work_dir / f"vk_match_result_{int(time.time())}.xlsx"
        try:
            stats = await asyncio.wait_for(
                asyncio.to_thread(run_vk_match, session.file_a_path, session.files_b_paths, output_path),
                timeout=900,
            )
            await message.reply_document(
                FSInputFile(str(output_path)),
                caption=(
                    "Готово.\n"
                    f"Лидов в A: {stats.get('rows_in_a', 0)}\n"
                    f"Сопоставлено: {stats.get('matched_rows', 0)}\n"
                    f"Не сопоставлено: {stats.get('unmatched_rows', 0)}\n"
                    f"Top combinations: {stats.get('combination_top', 0)}"
                ),
            )
        except asyncio.TimeoutError:
            await message.reply("Обработка заняла слишком много времени и была остановлена. Попробуйте с меньшими файлами.")
        except Exception as e:
            await message.reply(f"Не удалось обработать файлы: {e}")

    async def _handle_vk_b_documents(message: Message, documents: list[Message]) -> None:
        key = _vk_key(message)
        if key is None:
            return
        session = vk_sessions.get(key)
        if session is None or session.stage != VK_STAGE_WAIT_B:
            return
        if not documents:
            await message.reply("Не получил файлы B. Пришлите одним сообщением один или несколько документов.")
            return

        allowed_b = {".csv", ".xls", ".xlsx"}
        added_files: list[Path] = []
        for doc_message in documents:
            if doc_message.document is None:
                continue
            downloaded, ext = await _download_document_to_dir(doc_message, "B", VK_MAX_FILE_SIZE_BYTES)
            if ext not in allowed_b:
                downloaded.unlink(missing_ok=True)
                await message.reply(
                    "Файл B имеет неподдерживаемый формат. Допустимо: .csv, .xls, .xlsx"
                )
                return
            added_files.append(downloaded)

        if not added_files:
            await message.reply("Не удалось получить файлы B. Пришлите документы еще раз.")
            return

        session.files_b_paths.extend(added_files)
        await _run_vk_pipeline_and_send(message, session)
        _drop_vk_session(key)

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

    async def register_chat_presence(message: Message) -> None:
        if message.chat is None or message.from_user is None:
            return
        title = (
            (message.chat.title or "").strip()
            or (message.chat.full_name or "").strip()
            or str(message.chat.id)
        )
        await db.upsert_chat_registry(
            chat_id=message.chat.id,
            chat_title=title,
            chat_type=message.chat.type,
        )
        await db.mark_chat_member_seen(
            chat_id=message.chat.id,
            user_id=message.from_user.id,
        )

    async def _answer_cross_chat_query(
        *,
        message: Message,
        source_chat_query: str,
        topic: str,
    ) -> bool:
        resolved = await db.find_chat_by_title_fuzzy(
            title_query=source_chat_query,
            min_score=cfg.fuzzy_chat_min_score,
            max_candidates=cfg.fuzzy_chat_max_candidates,
        )
        if resolved is None:
            await message.reply(
                f"Не нашел чат с названием, похожим на «{source_chat_query}»."
            )
            return True
        source_chat_id, source_chat_title, _score = resolved

        allowed = await db.has_cross_chat_access(
            user_id=message.from_user.id,
            current_chat_id=message.chat.id,
            target_chat_id=source_chat_id,
        )
        if not allowed:
            await message.reply(
                "Не могу показать данные этого чата: доступ не подтвержден для этого пользователя."
            )
            return True

        now_ts = int(time.time())
        retention_min_ts = now_ts - cfg.retention_days * 24 * 3600
        topic_msgs = await db.get_messages_for_chat_topic(
            chat_id=source_chat_id,
            topic_query=topic,
            min_ts=retention_min_ts,
            limit=cfg.max_messages_for_ask,
        )
        if not topic_msgs:
            has_older = await db.has_messages_older_than(
                chat_id=source_chat_id, older_than_ts=retention_min_ts
            )
            if has_older:
                await message.reply(
                    f"По чату «{source_chat_title}» нет доступного контекста по теме: "
                    f"бот уже не хранит оперативные данные старше {cfg.retention_days} дней."
                )
            else:
                await message.reply(
                    f"По чату «{source_chat_title}» не нашел сообщений по теме «{topic or 'запрос'}»."
                )
            return True

        context_lines: list[str] = []
        total_chars = 0
        for msg in topic_msgs:
            line = message_to_excerpt(msg)
            extra = len(line) + 1
            if total_chars + extra > cfg.max_context_chars:
                break
            context_lines.append(line)
            total_chars += extra

        if llm is None:
            out = build_analysis_fallback(topic or source_chat_query, context_lines)
            await message.reply(
                f"По чату: {source_chat_title}\n\n{out}",
                disable_web_page_preview=True,
            )
            return True

        try:
            result = await llm.answer(
                mode=ANALYSIS_MODE,
                question=topic or f"Что обсуждали в чате {source_chat_title}",
                context_messages=context_lines,
                source_chat_title=source_chat_title,
                memory_scope=f"retention_{cfg.retention_days}_days",
            )
            text = (result.text or "").strip() or "Не удалось сформировать ответ."
        except Exception as e:
            hint = _openai_failure_reply(e)
            if hint:
                await message.reply(hint, disable_web_page_preview=True)
                return True
            text = build_analysis_fallback(topic or source_chat_query, context_lines)
        await message.reply(
            f"По чату: {source_chat_title}\n\n{text}",
            disable_web_page_preview=True,
        )
        return True

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
            await message.reply(build_help_text(bot_username))
            return

        if _is_current_chat_name_intent(q):
            chat_title = (
                (message.chat.title or "").strip()
                or (message.chat.full_name or "").strip()
                or str(message.chat.id)
            )
            await message.reply(
                f"Текущий чат называется: «{chat_title}».\n"
                f"chat_id: {message.chat.id}"
            )
            return

        cross_chat_req = _parse_cross_chat_intent(q)
        if cross_chat_req is not None:
            source_chat_query, topic = cross_chat_req
            handled = await _answer_cross_chat_query(
                message=message,
                source_chat_query=source_chat_query,
                topic=topic,
            )
            if handled:
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
        urls = extract_urls(q)
        effective_question = q
        if urls:
            for u in urls:
                effective_question = effective_question.replace(u, " ")
            effective_question = re.sub(r"\s+", " ", effective_question).strip()
            if not effective_question:
                effective_question = "Проанализируй источник по ссылке и извлеки ключевые факты."

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
                context_lines.append(f"[LINK] {eff_url}: {text}")

        use_chat_context = (
            mode == ANALYSIS_MODE
            and cfg.enable_chat_analysis
            and (forced_mode == ANALYSIS_MODE or _needs_chat_context(q))
        )
        if use_chat_context:
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
                out = build_analysis_fallback(effective_question, context_lines)
            else:
                out = build_assistant_fallback(effective_question)
            await message.reply(out, disable_web_page_preview=True)
            return

        try:
            result = await llm.answer(
                mode=mode,
                question=effective_question,
                context_messages=context_lines or None,
            )
            text = (result.text or "").strip() or "Не удалось сформировать ответ."
        except Exception as e:
            hint = _openai_failure_reply(e)
            if hint:
                await message.reply(hint, disable_web_page_preview=True)
                return
            if mode == ANALYSIS_MODE:
                text = build_analysis_fallback(effective_question, context_lines)
            else:
                text = build_assistant_fallback(effective_question)
        await message.reply(text, disable_web_page_preview=True)

    async def run_daily_summary_for_chat(chat_id: int) -> str:
        tz = cfg.daily_timezone
        end_dt = datetime.now().astimezone()
        start_dt = end_dt - timedelta(hours=24)
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        date_key = end_dt.strftime("%Y-%m-%d")

        msgs = await db.get_messages_for_chat_in_range(
            chat_id=chat_id,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=800,
        )
        if not msgs:
            return "За последние 24 часа в этом чате нет данных для сводки."

        excerpts = [message_to_excerpt(m) for m in msgs]
        selected: list[str] = []
        total_chars = 0
        for line in excerpts:
            extra = len(line) + 1
            if total_chars + extra > cfg.max_context_chars:
                break
            selected.append(line)
            total_chars += extra

        if not selected:
            return "Недостаточно данных для построения сводки."

        if llm is not None:
            try:
                daily = await llm.daily_summary(
                    chat_id=chat_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    message_bullets=selected,
                )
                summary_text = (daily.text or "").strip()
                if not summary_text:
                    summary_text = "Сводка не сформирована: модель вернула пустой ответ."
            except Exception:
                fallback = selected[:10]
                summary_text = "Сводка (упрощённая, без LLM):\n\n" + "\n".join(
                    [f"— {line}" for line in fallback]
                )
        else:
            fallback = selected[:10]
            summary_text = "Сводка (упрощённая, без LLM):\n\n" + "\n".join(
                [f"— {line}" for line in fallback]
            )

        header = f"📅 Сводка по запросу за {date_key} (период ~24 ч)\n\n"
        out_text = (header + summary_text).strip()
        await db.insert_daily_summary(
            chat_id=chat_id,
            date_key=date_key,
            start_ts=start_ts,
            end_ts=end_ts,
            summary_text=out_text,
        )
        return out_text

    async def send_text_chunks(message: Message, text: str) -> None:
        chunk_size = 3900
        payload = text or ""
        while payload:
            part = payload[:chunk_size]
            payload = payload[chunk_size:]
            await message.reply(part, disable_web_page_preview=True)

    @router.message(Command("start"))
    async def on_start(message: Message) -> None:
        await register_chat_presence(message)
        await message.reply(build_help_text(bot_username))

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        await register_chat_presence(message)
        await message.reply(build_help_text(bot_username))

    @router.message(Command("mode"))
    async def on_mode(message: Message) -> None:
        if message.chat is None:
            return
        await register_chat_presence(message)
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
        await register_chat_presence(message)
        runtime_state_by_chat[message.chat.id] = ChatRuntimeState(mode=default_mode)
        await message.reply(f"Сброс выполнен. Режим по умолчанию: {default_mode}")

    @router.message(Command("ask"))
    async def on_ask(message: Message) -> None:
        if message.chat is None:
            return
        await register_chat_presence(message)
        q = _parse_command_args(message.text)
        if not q:
            await message.reply("Формат: /ask <вопрос>")
            return
        await answer_user_query(message=message, question=q, forced_mode=ANALYSIS_MODE)

    @router.message(Command("daily_summary"))
    @router.message(Command("summary"))
    async def on_daily_summary(message: Message) -> None:
        if message.chat is None:
            return
        await register_chat_presence(message)
        await message.reply("Формирую сводку за последние 24 часа...")
        out = await run_daily_summary_for_chat(message.chat.id)
        await send_text_chunks(message, out)

    @router.message(Command("chat_info"))
    async def on_chat_info(message: Message) -> None:
        if message.chat is None:
            return
        await register_chat_presence(message)
        title = (message.chat.title or "").strip() or "(без названия)"
        await message.reply(
            f"chat_id: {message.chat.id}\n"
            f"chat_title: {title}\n"
            f"chat_type: {message.chat.type}"
        )

    @router.message(Command("VkMatch"))
    @router.message(Command("vkmatch"))
    async def on_vk_match(message: Message) -> None:
        if message.chat is None or message.from_user is None:
            return
        await register_chat_presence(message)
        key = (message.chat.id, message.from_user.id)
        old = vk_sessions.get(key)
        if old is not None:
            _cleanup_vk_session_files(old)
        work_dir = Path("./data/vk_match_tmp") / f"{message.chat.id}_{message.from_user.id}_{int(time.time())}"
        work_dir.mkdir(parents=True, exist_ok=True)
        vk_sessions[key] = VkMatchSession(
            stage=VK_STAGE_WAIT_A,
            created_at_ts=int(time.time()),
            work_dir=work_dir,
            file_a_path=None,
            files_b_paths=[],
        )
        await message.reply(
            "Режим VkMatch запущен.\n"
            "Шаг 1/2: пришлите файл A (csv/xls/xlsx/pdf)."
        )

    @router.message()
    async def on_text(message: Message) -> None:
        if message.chat is None or message.from_user is None:
            return
        if message.from_user.id == bot_id:
            return

        vk_key = (message.chat.id, message.from_user.id)
        vk_session = vk_sessions.get(vk_key)
        if vk_session is not None and message.document is not None:
            await register_chat_presence(message)
            if vk_session.stage == VK_STAGE_WAIT_A:
                allowed_a = {".csv", ".xls", ".xlsx", ".pdf"}
                downloaded, ext = await _download_document_to_dir(message, "A", VK_MAX_FILE_SIZE_BYTES)
                if ext not in allowed_a:
                    downloaded.unlink(missing_ok=True)
                    await message.reply("Файл A должен быть формата: .csv, .xls, .xlsx или .pdf")
                    return
                vk_session.file_a_path = downloaded
                vk_session.stage = VK_STAGE_WAIT_B
                await message.reply(
                    "Файл A принят.\n"
                    "Шаг 2/2: пришлите одним сообщением файлы B (один или несколько, форматы .csv/.xls/.xlsx)."
                )
                return

            if vk_session.stage == VK_STAGE_WAIT_B:
                media_group_id = message.media_group_id
                if media_group_id:
                    buffer_key = (message.chat.id, message.from_user.id, media_group_id)
                    vk_media_buffers.setdefault(buffer_key, []).append(message)
                    existing_task = vk_media_tasks.get(buffer_key)
                    if existing_task is not None and not existing_task.done():
                        existing_task.cancel()

                    async def _flush_media_group() -> None:
                        try:
                            await asyncio.sleep(1.2)
                            items = vk_media_buffers.pop(buffer_key, [])
                            vk_media_tasks.pop(buffer_key, None)
                            await _handle_vk_b_documents(message, items)
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            await message.reply(f"Ошибка обработки файлов B: {e}")

                    vk_media_tasks[buffer_key] = asyncio.create_task(_flush_media_group())
                    return

                await _handle_vk_b_documents(message, [message])
                return

        if message.text and message.text.strip().startswith("/"):
            return

        text = message.text or None
        caption = getattr(message, "caption", None) or None
        media = extract_media_metadata(message)
        if text or caption or media:
            await register_chat_presence(message)
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

    async def retention_cleanup_loop() -> None:
        while True:
            try:
                now_ts = int(time.time())
                older_than_ts = now_ts - cfg.retention_days * 24 * 3600
                await db.cleanup_old_messages(older_than_ts=older_than_ts)
                expired_keys = [
                    key
                    for key, sess in vk_sessions.items()
                    if now_ts - sess.created_at_ts > VK_SESSION_TTL_SECONDS
                ]
                for key in expired_keys:
                    _drop_vk_session(key)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(max(3600, cfg.retention_cleanup_hours * 3600))

    retention_task = asyncio.create_task(retention_cleanup_loop())

    try:
        await dp.start_polling(bot)
    finally:
        retention_task.cancel()
        await db.close()

