from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI

from prompts import ANALYSIS_SYSTEM, ASSISTANT_SYSTEM, DAILY_SUMMARY_SYSTEM, HELP_SYSTEM


@dataclass(frozen=True)
class LLMResult:
    text: str


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

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

    async def answer(
        self,
        *,
        mode: str,
        question: str,
        context_messages: Sequence[str] | None = None,
    ) -> LLMResult:
        if mode == "help_mode":
            system = HELP_SYSTEM
        elif mode == "analysis_mode":
            system = ANALYSIS_SYSTEM
        else:
            system = ASSISTANT_SYSTEM

        prompt = f"Question:\n{question.strip()}\n"
        if context_messages:
            prompt += "\nContext:\n" + "\n".join(context_messages)

        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text.strip())

