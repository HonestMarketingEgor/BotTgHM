from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI

from prompts import ASK_SYSTEM, DAILY_SUMMARY_SYSTEM, MARKETING_SUGGESTIONS_SYSTEM
from prompts import NUMBERS_SYSTEM


@dataclass(frozen=True)
class LLMResult:
    text: str


class LLMClient:
    def __init__(
        self, *, api_key: str, model: str, suggestions_model: str | None = None
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._suggestions_model = suggestions_model or model

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
    ) -> LLMResult:
        context_block = ""
        if latest_daily_summary:
            context_block += f"Latest daily summary:\n{latest_daily_summary}\n\n"

        context_block += "Recent relevant messages:\n" + "\n".join(context_messages)

        prompt = f"""Question:
{question}

Context:
{context_block}
"""

        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ASK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip())

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
        question: str,
        latest_daily_summary: str | None,
        context_messages: Sequence[str],
        image_urls: Sequence[str],
    ) -> LLMResult:
        """
        Multimodal variant: adds images via image_url parts.
        """
        context_block = ""
        if latest_daily_summary:
            context_block += f"Latest daily summary:\n{latest_daily_summary}\n\n"
        context_block += "Recent relevant messages:\n" + "\n".join(context_messages)

        prompt = f"""Question:
{question}

Context:
{context_block}

Analyze the provided images as the source of the creative details.
Extract what matters for answering the question. Do not invent numbers.
"""

        # Build OpenAI multimodal message with image_url parts.
        user_content: list[dict] = [{"type": "text", "text": prompt}]
        for u in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": u}})

        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ASK_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip())

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

