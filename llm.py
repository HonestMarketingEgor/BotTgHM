from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI

from prompts import (
    ASK_MODE_DATA_FIRST,
    ASK_MODE_SHORT_EXEC,
    ASK_MODE_STRUCTURED_MARKETING,
    ASK_SYSTEM,
    DAILY_SUMMARY_SYSTEM,
    MARKETING_SUGGESTIONS_SYSTEM,
    PROJECT_CONTEXT_SYSTEM,
)
from prompts import NUMBERS_SYSTEM

_DEBUG_LOG_PATH = "/Users/pelemenio/telegram-context-bot/.cursor/debug-283857.log"
_DEBUG_SESSION_ID = "283857"


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


@dataclass(frozen=True)
class LLMResult:
    text: str
    response_mode: str | None = None


def classify_response_mode(question: str) -> str:
    q = (question or "").strip().lower()
    data_markers = [
        "kpi",
        "ctr",
        "cpc",
        "cpl",
        "cpm",
        "roi",
        "roas",
        "бюджет",
        "расход",
        "цифр",
        "таблиц",
        "метрик",
        "лид",
        "конверс",
        "выруч",
        "ддр",
    ]
    exec_markers = [
        "руковод",
        "кратко",
        "summary",
        "самое важное",
        "итог",
        "приоритет",
        "что делать сегодня",
        "для директора",
        "для owner",
    ]
    structured_markers = [
        "гипотез",
        "причин",
        "почему",
        "план",
        "стратег",
        "что тестировать",
        "как улучш",
        "следующий шаг",
    ]
    if any(m in q for m in data_markers):
        return "data_first"
    if any(m in q for m in exec_markers):
        return "short_exec"
    if any(m in q for m in structured_markers):
        return "structured_marketing"
    return "structured_marketing"


def _mode_instruction(mode: str) -> str:
    if mode == "data_first":
        return ASK_MODE_DATA_FIRST
    if mode == "short_exec":
        return ASK_MODE_SHORT_EXEC
    return ASK_MODE_STRUCTURED_MARKETING


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        suggestions_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._suggestions_model = suggestions_model or model

    def classify_response_mode(self, *, question: str) -> str:
        return classify_response_mode(question)

    async def daily_summary(
        self,
        *,
        chat_id: int,
        start_ts: int,
        end_ts: int,
        message_bullets: Sequence[str],
    ) -> LLMResult:
        user_content = "\n".join(message_bullets)
        prompt = f"""Time window: {start_ts}..{end_ts}

Messages:
{user_content}
"""

        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": DAILY_SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip())

    async def answer_question(
        self,
        *,
        chat_id: int,
        question: str,
        latest_daily_summary: str | None,
        context_messages: Sequence[str],
        project_context: str | None = None,
        response_mode: str | None = None,
    ) -> LLMResult:
        selected_mode = response_mode or classify_response_mode(question)
        context_block = ""
        if project_context:
            context_block += f"Project context:\n{project_context}\n\n"
        if latest_daily_summary:
            context_block += f"Latest daily summary:\n{latest_daily_summary}\n\n"

        context_block += "Recent relevant messages:\n" + "\n".join(context_messages)
        format_instruction = _mode_instruction(selected_mode)

        prompt = f"""Question:
{question}

Context:
{context_block}

Response format:
{format_instruction}
"""

        run_id = f"llm-ask-{chat_id}-{int(time.time())}"
        started_ms = int(time.time() * 1000)
        # #region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H2_llm_latency_or_hang",
            location="llm.py:answer_question_before_openai",
            message="calling openai chat.completions",
            data={
                "question_len": len(question),
                "context_messages_count": len(context_messages),
                "has_daily_summary": bool(latest_daily_summary),
                "has_project_context": bool(project_context),
                "response_mode": selected_mode,
                "model": self._model,
            },
        )
        # #endregion
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ASK_SYSTEM + "\n\n" + PROJECT_CONTEXT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        # #region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H2_llm_latency_or_hang",
            location="llm.py:answer_question_after_openai",
            message="openai response received",
            data={
                "duration_ms": int(time.time() * 1000) - started_ms,
                "choices_count": len(getattr(resp, "choices", []) or []),
            },
        )
        # #endregion
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip(), response_mode=selected_mode)

    async def marketing_suggestions(
        self,
        *,
        question: str,
        context_messages: Sequence[str],
    ) -> LLMResult:
        """
        Кнопка «Предложения от ИИ».
        """
        context_block = "\n".join(context_messages)
        prompt = f"""Question:
{question}

Chat context:
{context_block}
"""

        resp = await self._client.chat.completions.create(
            model=self._suggestions_model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": MARKETING_SUGGESTIONS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip())

    async def answer_question_with_images(
        self,
        *,
        chat_id: int,
        question: str,
        latest_daily_summary: str | None,
        context_messages: Sequence[str],
        image_urls: Sequence[str],
        project_context: str | None = None,
        response_mode: str | None = None,
    ) -> LLMResult:
        """
        Multimodal variant: adds images via image_url parts.
        """
        selected_mode = response_mode or classify_response_mode(question)
        context_block = ""
        if project_context:
            context_block += f"Project context:\n{project_context}\n\n"
        if latest_daily_summary:
            context_block += f"Latest daily summary:\n{latest_daily_summary}\n\n"
        context_block += "Recent relevant messages:\n" + "\n".join(context_messages)
        format_instruction = _mode_instruction(selected_mode)

        prompt = f"""Question:
{question}

Context:
{context_block}

Analyze the provided images as the source of the creative details.
Extract what matters for answering the question. Do not invent numbers.

Response format:
{format_instruction}
"""

        # Build OpenAI multimodal message with image_url parts.
        user_content: list[dict] = [{"type": "text", "text": prompt}]
        for u in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": u}})

        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ASK_SYSTEM + "\n\n" + PROJECT_CONTEXT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip(), response_mode=selected_mode)

    async def marketing_suggestions_with_images(
        self,
        *,
        question: str,
        context_messages: Sequence[str],
        image_urls: Sequence[str],
    ) -> LLMResult:
        context_block = "\n".join(context_messages)
        prompt = f"""Question:
{question}

Chat context:
{context_block}

Use the images as the primary source for creative/assets details.
Provide improvements and next steps. Do not invent facts.
"""
        user_content: list[dict] = [{"type": "text", "text": prompt}]
        for u in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": u}})

        resp = await self._client.chat.completions.create(
            model=self._suggestions_model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": MARKETING_SUGGESTIONS_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip())

    async def extract_key_numbers(
        self,
        *,
        question: str,
        context_messages: Sequence[str],
    ) -> LLMResult:
        context_block = "\n".join(context_messages)
        prompt = f"""Question:
{question}

Messages:
{context_block}
"""

        resp = await self._client.chat.completions.create(
            model=self._suggestions_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": NUMBERS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        # Keep only the allowed lines (first header + up to 4 lines after it).
        cleaned = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
        return LLMResult(text=cleaned)

