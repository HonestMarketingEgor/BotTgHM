from __future__ import annotations

import re
from typing import Sequence


def _extract_content(excerpt_line: str) -> str:
    # excerpt_line is like: "YYYY-mm-dd HH:MM @user: text..."
    # We keep everything after the first ": " to focus on the message content.
    if ": " in excerpt_line:
        return excerpt_line.split(": ", 1)[1]
    return excerpt_line


def _detect_topic(question: str, context_messages: Sequence[str]) -> str:
    q = (question or "").lower()
    ctx = " ".join(context_messages).lower()

    if "бюджет" in q or "бюджет" in ctx:
        return "Бюджет"
    if "релиз" in q or "релиз" in ctx or "release" in q or "release" in ctx:
        return "Релиз"
    if "план" in q or "план" in ctx:
        return "План"
    if "срок" in q or "дедлайн" in q or "дата" in ctx:
        return "Сроки"
    return "Обсуждение"


def _collect_numbers(context_messages: Sequence[str]) -> list[str]:
    ctx = " ".join(context_messages)
    # Keep common numeric patterns: 2, 2.5, 2,5, 2026, etc.
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", ctx)
    return nums


def _fallback_answer(question: str, context_messages: Sequence[str]) -> str:
    topic = _detect_topic(question, context_messages)
    ctx_raw = " ".join(context_messages).lower()
    ctx = " ".join(_extract_content(x) for x in context_messages).lower()

    numbers = _collect_numbers(context_messages)

    fact1 = f"Обсуждают ключевые решения по теме «{topic}»."

    fact2 = ""
    if "слив" in ctx_raw or "сливаем" in ctx_raw or "слива" in ctx_raw:
        fact2 = "Есть сигнал про неэффективные траты («сливаем деньги»)."
    elif "бюджет" in ctx_raw or "расход" in ctx_raw or "трата" in ctx_raw:
        fact2 = "Обсуждают структуру/распределение расходов в рамках бюджета."
    elif "релиз" in ctx_raw or "release" in ctx_raw:
        fact2 = "Есть договоренности по срокам/содержанию релиза."
    else:
        fact2 = "В контексте присутствуют доводы и требования к решению."

    if numbers:
        fact3 = f"Упомянуты числа: {', '.join(numbers[:3])}."
    else:
        fact3 = "Не хватает конкретных цифр/дат — нужны уточнения."

    conclusion = "Собери недостающие цифры (суммы/даты) и зафиксируй критерии успеха, чтобы принять решение."

    return "\n".join(
        [
            f"📊 {topic}:",
            f"— {fact1}",
            f"— {fact2}",
            f"— {fact3}",
            f"⚠️ Вывод: {conclusion}",
        ]
    )


def build_fallback_answer(question: str, context_messages: Sequence[str]) -> str:
    # Ensure 5-7 lines; current template is exactly 5 lines.
    return _fallback_answer(question, context_messages)


def _parse_excerpt(excerpt_line: str) -> tuple[str, str, str]:
    """
    Returns (time_HHMM, author, content).
    excerpt_line format: "YYYY-mm-dd HH:MM @user: content"
    """
    if " @" in excerpt_line:
        dt_part, rest = excerpt_line.split(" @", 1)
    else:
        return ("", "", excerpt_line)

    dt_time = dt_part.strip().split(" ")
    time_hhmm = dt_time[-1] if dt_time else ""

    if ": " in rest:
        author, content = rest.split(": ", 1)
    else:
        author, content = rest, ""

    return (time_hhmm, author.strip(), (content or "").strip())


def _normalize_spaces(s: str) -> str:
    return (s or "").replace("\u00A0", " ").replace("\n", " ").strip()


def _extract_amounts_from_text(text: str) -> list[tuple[float, str]]:
    """
    Returns list of (value_as_number, display_label).
    Supports: "20 000", "20k", "100к", "$500", "500 $" etc (heuristic).
    """
    t = _normalize_spaces(text.lower())
    out: list[tuple[float, str]] = []

    # Patterns for currency symbols and thousand suffix "к"/"тыс".
    # Capture: number part (may contain spaces as thousand separators), optional decimals, optional suffix.
    pattern = re.compile(
        r"(?P<cur>[$€£])?\s*"
        r"(?P<num>\d{1,3}(?:[ ]\d{3})+|\d+)"
        r"(?:[.,](?P<dec>\d+))?\s*"
        r"(?P<suffix>к|тыс)?"
    )
    for m in pattern.finditer(t):
        num_raw = m.group("num").replace(" ", "").replace("\u00A0", "")
        try:
            base = int(num_raw)
        except Exception:
            continue
        dec = m.group("dec")
        val = float(base)
        if dec:
            val = val + float(f"0.{dec}")

        suffix = m.group("suffix")
        if suffix in ("к", "тыс"):
            val = val * 1000.0

        cur = m.group("cur")
        label = ""
        if cur:
            label = cur
        elif suffix in ("к", "тыс"):
            label = "тыс."
        out.append((val, label))
    return out


def _collect_lines_with_keywords(context_messages: Sequence[str], keywords: Sequence[str]) -> list[str]:
    keys = [k.lower() for k in keywords]
    out: list[str] = []
    for line in context_messages:
        content = _extract_content(line).lower()
        if any(k in content for k in keys):
            out.append(line)
    return out


def build_numbers_answer(question: str, context_messages: Sequence[str]) -> str:
    def fmt_num(v: float) -> str:
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return str(round(v, 2)).rstrip("0").rstrip(".")

    def detect_unit(s: str) -> str | None:
        ll = s.lower()
        if "₽" in ll or "руб" in ll:
            return "₽"
        if "$" in ll:
            return "$"
        if "€" in ll:
            return "€"
        return None

    def extract_first_amount_near_keyword(line: str, keyword: str) -> float | None:
        # Extracts the first numeric value after keyword.
        lower = line.lower()
        idx = lower.find(keyword)
        if idx < 0:
            return None
        tail = lower[idx:]
        m = re.search(rf"{re.escape(keyword)}[^0-9]*([0-9]+(?:[.,][0-9]+)?)", tail)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    # Extract CPL / CPL квал / ROI from lines that explicitly mention them.
    cpl_val: float | None = None
    cpl_qual_val: float | None = None
    roi_val: float | None = None
    cpl_unit: str | None = None
    roi_unit: str | None = None

    for line in context_messages:
        content = _extract_content(line)
        ll = content.lower()
        if "cpl" in ll:
            unit = detect_unit(content)
            if "квал" in ll or "qual" in ll:
                if cpl_qual_val is None:
                    v = extract_first_amount_near_keyword(ll, "cpl")
                    if v is not None:
                        cpl_qual_val = v
                        cpl_unit = unit
            else:
                if cpl_val is None:
                    v = extract_first_amount_near_keyword(ll, "cpl")
                    if v is not None:
                        cpl_val = v
                        cpl_unit = unit
        if "roi" in ll or "рroi" in ll:
            if roi_val is None:
                unit = detect_unit(content)
                m = re.search(r"(?:roi|рroi)[^0-9]*([0-9]+(?:[.,][0-9]+)?)", ll)
                if m:
                    try:
                        roi_val = float(m.group(1).replace(",", "."))
                        roi_unit = unit
                    except Exception:
                        pass

    # Extract budget amounts only from “budget-like” lines (avoid treating CPL/ROI lines as budget).
    budget_keywords = ["бюджет", "бюджеты", "расход", "трата", "стоимость", "спенд", "слив", "тест"]
    budget_values: list[float] = []
    budget_unit: str | None = None

    for line in context_messages:
        content = _extract_content(line)
        ll = content.lower()
        if any(k in ll for k in budget_keywords) and "cpl" not in ll and "roi" not in ll:
            unit = detect_unit(content)
            if budget_unit is None:
                budget_unit = unit
            for v, _ in _extract_amounts_from_text(content):
                budget_values.append(v)

    # Percent tokens (optional if present)
    percent_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\s*%\b", " ".join(_extract_content(l) for l in context_messages))

    # Compose minimal output: up to 4 metric lines.
    metrics: list[str] = []

    if cpl_qual_val is not None:
        unit_part = f" {cpl_unit}".rstrip() if cpl_unit else ""
        metrics.append(f"KPI по CPL квал: {fmt_num(cpl_qual_val)}{unit_part}")
    elif cpl_val is not None:
        unit_part = f" {cpl_unit}".rstrip() if cpl_unit else ""
        metrics.append(f"KPI по CPL: {fmt_num(cpl_val)}{unit_part}")

    if roi_val is not None and len(metrics) < 4:
        unit_part = f" {roi_unit}".rstrip() if roi_unit else ""
        metrics.append(f"KPI по ROI: {fmt_num(roi_val)}{unit_part}")

    if budget_values and len(metrics) < 4:
        uniq = sorted(set(round(v, 6) for v in budget_values))
        # Keep the shortest meaningful: first 2 values.
        uniq = uniq[:2]
        unit_part = f" {budget_unit}".rstrip() if budget_unit else ""
        if len(uniq) == 1:
            metrics.append(f"Бюджет: {fmt_num(uniq[0])}{unit_part}")
        else:
            metrics.append(f"Бюджеты: {fmt_num(uniq[0])}{unit_part} и {fmt_num(uniq[1])}{unit_part}")

    if percent_tokens and len(metrics) < 4:
        metrics.append(f"Проценты: {percent_tokens[0]}")

    # Cap to at most 4 metric lines (no computations).
    metrics = metrics[:4]
    if not metrics:
        return "📊 Цифры: нет явных KPI (CPL/ROI) и бюджетных сумм в выбранном контексте."
    if len(metrics) == 1:
        return metrics[0]
    return "📊 Цифры:\n" + "\n".join(metrics)


def build_deep_analysis(question: str, context_messages: Sequence[str]) -> str:
    # Heuristic “analyst” response in strict 5-line format.
    ctx = " ".join(_extract_content(l) for l in context_messages).lower()
    topic = _detect_topic(question, context_messages)

    has_ineff = ("слив" in ctx) or ("неэффектив" in ctx) or ("неэфф" in ctx)
    has_cpl = "cpl" in ctx
    has_roi = "roi" in ctx
    has_budget = "бюджет" in ctx or "расход" in ctx or "трата" in ctx
    has_control_gap = (
        "нет контроля" in ctx
        or "без контроля" in ctx
        or "нет метрик" in ctx
        or "без метрик" in ctx
        or "не хватает" in ctx
        or "нет данных" in ctx
        or "без данных" in ctx
    )
    responsibility_missing = (
        "нет ответственного" in ctx
        or "ответственного нет" in ctx
        or "нет владельца" in ctx
        or "без ответственного" in ctx
    )

    fact1 = "Обсуждают бюджет/расход как управленческую тему." if has_budget else "Тема бюджета присутствует частично."
    fact2 = "Есть сигнал о риске неэффективных трат (упоминание «сливаем деньги»/неэффективности)." if has_ineff else "Сигнала о проблемных тратах не видно в выбранных фрагментах."
    if has_control_gap and responsibility_missing:
        fact3 = "Риски: нет контроля метрик и нет ответственного за бюджет."
    elif has_control_gap:
        fact3 = "Риск: решения принимаются без контроля метрик/драйверов."
    elif responsibility_missing:
        fact3 = "Риск: нет ответственного за бюджет, решения не доводятся до результата."
    else:
        if has_cpl and has_roi:
            fact3 = "Есть метрики (CPL/ROI), можно связывать действия с результатом."
        elif has_cpl:
            fact3 = "Есть CPL-ориентиры; ROI/доп. метрики в данных ограничены."
        else:
            fact3 = "Недостаточно метрик для управленческих решений."

    conclusion = "Зафиксируйте набор метрик (CPL/ROI) и владельца бюджета, чтобы решения были измеримыми."

    return "\n".join(
        [
            f"📊 По бюджету:",
            f"— {fact1}",
            f"— {fact2}",
            f"— {fact3}",
            f"⚠️ Вывод: {conclusion}",
        ]
    )


def build_messages_answer(question: str, context_messages: Sequence[str]) -> str:
    # Show only facts: author + time + exact snippet (truncated).
    ctx = " ".join(_extract_content(l) for l in context_messages).lower()
    want_budget = ("бюджет" in question.lower()) or ("бюджет" in ctx) or ("cpl" in ctx) or ("roi" in ctx) or ("слив" in ctx)

    keywords = ["бюджет", "расход", "трата", "cpl", "roi", "слив", "сливаем", "тест"]
    filtered = []
    for line in context_messages:
        content = _extract_content(line).lower()
        if want_budget:
            if any(k in content for k in keywords):
                filtered.append(line)
        else:
            filtered.append(line)

    # Keep it short: 1 label + up to 5 items = <= 6 lines.
    filtered = filtered[:5]
    lines: list[str] = ["📩 Сообщения:"]
    for line in filtered:
        # Special handling for link blocks we inject as context.
        if line.startswith("[LINK] "):
            # "[LINK] <url>: <text>"
            rest = line[len("[LINK] ") :]
            url = rest.split(":", 1)[0].strip()
            snippet = rest.split(":", 1)[1].strip() if ":" in rest else ""
            snippet = _normalize_spaces(snippet)
            if len(snippet) > 150:
                snippet = snippet[:147] + "…"
            lines.append(f"Ссылка — {url}: {snippet}")
            continue

        time_hhmm, author, content = _parse_excerpt(line)
        content = _normalize_spaces(content)
        if len(content) > 150:
            content = content[:147] + "…"
        if time_hhmm and author:
            lines.append(f"{author} — {time_hhmm}: {content}")
        elif content:
            lines.append(content)
    return "\n".join(lines)


def build_freeform_answer(question: str, context_messages: Sequence[str]) -> str:
    """
    Short, non-template fallback for when the LLM output is unusable or unavailable.
    Uses the provided context lines (already extracted from chat/link).
    """
    q = (question or "").lower()
    # Use a tiny keyword set to pick relevant lines.
    keywords = []
    for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", q):
        if token not in keywords:
            keywords.append(token)

    def score_line(line: str) -> int:
        content = _extract_content(line).lower()
        if not keywords:
            return 1
        return sum(1 for k in keywords[:12] if k in content)

    picked = sorted(context_messages, key=score_line, reverse=True)[:6]
    if not picked:
        return "Нет данных для ответа."

    # If link blocks are present, prefer those.
    link_lines = [l for l in picked if l.startswith("[LINK] ")]
    if link_lines:
        picked = link_lines[:4]

    lines: list[str] = ["Ключевое по запросу:"]
    for l in picked:
        lines.append("— " + _normalize_spaces(_extract_content(l))[:220])
    return "\n".join(lines[:7])


def build_marketing_suggestions_fallback(question: str, context_messages: Sequence[str]) -> str:
    """
    Быстрый fallback, если OpenAI недоступен.
    Дает 5-7 строк с конкретными следующими шагами без воды.
    """
    q = (question or "").lower()
    ctx = " ".join(_extract_content(l) for l in context_messages).lower()

    has_budget = "бюджет" in q or "расход" in q or "трата" in q or "бюджет" in ctx or "cpl" in ctx or "roi" in ctx
    has_cpl = "cpl" in ctx
    has_roi = "roi" in ctx
    has_ineff = "слив" in ctx or "неэффектив" in ctx or "неэфф" in ctx
    has_test = "тест" in q or "тест" in ctx or "эксперим" in ctx
    has_launch = "запуск" in q or "запуск" in ctx or "заплан" in ctx or "релиз" in ctx

    fact1 = "Зафиксируй цель теста через метрику: CPL/ROI."
    if has_budget:
        fact1 = "Привяжи бюджет к целевой метрике: CPL и/или ROI."

    fact2 = "Собери 2-3 гипотезы: аудитория + креатив + оффер."
    if has_launch or has_test:
        fact2 = "Спланируй запуск/тест: когорты, длительность, критерий остановки."

    fact3 = "Проверь трекинг: конверсии, CPL квал, воронка от клика до результата."
    if not (has_cpl or has_roi):
        fact3 = "Если CPL/ROI не сходятся, сначала выровняй воронку и определения конверсий."

    fact4 = "Риск: нет контроля → решения без метрик и ответственного."
    if not has_ineff:
        fact4 = "Снизь риски: еженедельный контроль метрик и бюджетных решений."

    out_lines = [
        "🧠 Решения для улучшения:",
        f"— {fact1}",
        f"— {fact2}",
        f"— {fact3}",
        f"— {fact4}",
        "⚠️ Риск: нет контроля метрик → решения без измерения. Снизить: зафиксируй KPI и критерий остановки.",
    ]
    return "\n".join(out_lines[:7])


def build_tasks_answer(question: str, context_messages: Sequence[str]) -> str:
    """
    Расстановка задач только на тех, кто реально упомянут/присутствует в контексте.
    """
    # Authors are user handles without '@' from _parse_excerpt.
    authors: set[str] = set()
    for line in context_messages:
        _, author, _ = _parse_excerpt(line)
        if author:
            authors.add(author.lower())

    # Mapping from handle -> short role/action owner.
    owners: dict[str, str] = {
        "dasssshay": "Даша",
        "isaevnikita": "Никита",
        "blackwoot": "Сергей",
        "dre1ws": "Андрей",
        "alex_hristich": "Александр",
        "olesya_targ": "Олеся",
    }

    def present(handle: str) -> bool:
        return handle.lower() in authors

    q = (question or "").lower()
    ctx = " ".join(_extract_content(l) for l in context_messages).lower()

    mentions_budget = "бюджет" in q or "cpl" in q or "roi" in q or "расход" in q or "трата" in q or "слив" in ctx
    mentions_launch = "запуск" in q or "релиз" in q or "тест" in q or "тест" in ctx

    tasks: list[str] = ["🎯 Расставить задачи:"]

    # PM/strategy
    if present("dasssshay"):
        tasks.append(
            "- @dasssshay: свести KPI в таблицу, организовать коммуникацию по задачам и контроль бюджета/спенд."
        )
    if present("isaevnikita"):
        tasks.append(
            "- @IsaevNikita: утвердить стратегию эксперимента/релиза, связать цели с KPI и определить план/критерии остановки."
        )

    # Targetologists
    # Decide likely platforms by keywords in question/context.
    platform_hint = ctx + " " + q
    wants_fb = any(x in platform_hint for x in ["facebook", "instagram", "linkedin", "tiktok", "fb", "insta", "linkedin", "tiktok", "ссылка"])
    wants_yandex = any(x in platform_hint for x in ["yandex", "google", "google ads", "яндекс"])
    wants_vk = any(x in platform_hint for x in ["vk", "vkontakte", "vk ads"])

    if any(x in platform_hint for x in ["facebook", "instagram", "tiktok", "лиц", "linkedin"]):
        wants_fb = True

    if present("blackwoot") and (mentions_budget or mentions_launch or wants_fb):
        tasks.append(
            "- @Blackwoot: запустить тест по своему каналу, выдать гипотезы креатива/аудитории и контролировать CPL/ROI."
        )
    if present("dre1ws") and (mentions_budget or mentions_launch or wants_fb):
        tasks.append(
            "- @dre1ws: оптимизировать кампании по CPL/качеству, предложить 2-3 варианта оффера/аудиторий."
        )

    if present("alex_hristich") and (wants_yandex or mentions_budget or mentions_launch):
        tasks.append(
            "- @Alex_Hristich: настроить связку Яндекс/Google под KPI, проверить семантику и минимизировать CPL."
        )
    if present("olesya_targ") and (wants_vk or mentions_budget or mentions_launch):
        tasks.append(
            "- @Olesya_targ: проработать VK ADS под KPI, запустить тест форматов и обеспечить сквозное измерение."
        )

    # If only one owner is present, still keep it useful.
    if len(tasks) == 1:
        tasks.append("- В чате нет распознаваемых ролей. Добавь людей с @username, чтобы бот мог назначать задачи.")

    return "\n".join(tasks[:7])


def build_help_text(bot_username: str | None = None) -> str:
    mention_hint = f"@{bot_username}" if bot_username else "@bot"
    return "\n".join(
        [
            "Я ИИ-помощник: универсальный ассистент + аналитик для рекламного агентства.",
            "",
            "Как задать вопрос:",
            f"— в групповом чате: напиши {mention_hint} и вопрос",
            "— в группе команды работают без @упоминания: /start, /help, /ask, /mode, /reset, /chat_info",
            f"— формат: /ask <вопрос> или {mention_hint} <вопрос>",
            "",
            "Что умею (универсально):",
            "— объяснить, как пользоваться ботом и с чего начать",
            "— структурировать запрос и предложить план действий",
            "— дать короткий ответ на бытовые и рабочие вопросы",
            "— корректно назвать текущий чат по его заголовку",
            "",
            "Что умею для агентства:",
            "— анализировать сообщения чата по явному запросу",
            "— предлагать варианты действий и гипотезы",
            "— подсвечивать, каких данных не хватает для решения",
            "— искать контекст по другому чату (если пользователь состоит в обоих чатах)",
            "",
            "Память и хранение:",
            "— оперативные сообщения хранятся до 6 месяцев",
            "— если данные старше 6 месяцев, бот прямо скажет об этом",
            "",
            "Режимы:",
            "— /mode assistant  (универсальный помощник)",
            "— /mode analysis   (анализ по чату)",
            "— /reset           (сброс к режиму по умолчанию)",
            "— /chat_info       (название и id текущего чата)",
        ]
    )


def build_help_redirect(bot_username: str | None = None) -> str:
    mention_hint = f"@{bot_username}" if bot_username else "@bot"
    return "\n".join(
        [
            "Этот запрос про возможности бота.",
            "Чтобы получить полную инструкцию, используй /help.",
            f"Коротко: в группе напиши {mention_hint} и вопрос, например:",
            f"— {mention_hint} проанализируй KPI и предложи план на неделю",
        ]
    )


def build_assistant_fallback(question: str) -> str:
    return "\n".join(
        [
            "Сейчас отвечу как помощник в коротком формате.",
            f"Запрос: {question[:180]}",
            "Если нужен анализ переписки, используй /mode analysis или начни с «проанализируй ...».",
        ]
    )


def build_analysis_fallback(question: str, context_messages: Sequence[str]) -> str:
    if not context_messages:
        return (
            "Для анализа не хватает контекста из чата. "
            "Сформулируй задачу точнее или добавь данные в чат."
        )
    return build_freeform_answer(question, context_messages)

