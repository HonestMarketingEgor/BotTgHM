from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TableKpis:
    year: int | None
    leads: int | None
    spent: float | None
    cpl: float | None
    ctr: float | None
    cv: float | None
    cpl_qual: float | None


def _detect_year(text: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", text or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _norm_number(s: str) -> float | None:
    if s is None:
        return None
    raw = str(s).strip()
    if not raw:
        return None

    # Remove thousands separators spaces.
    raw = raw.replace("\u00a0", " ")
    raw = raw.replace(" ", "")

    # If comma used as decimal separator.
    if "," in raw and "." not in raw:
        raw = raw.replace(",", ".")

    # Remove trailing percent sign.
    raw = raw.replace("%", "")

    try:
        return float(raw)
    except Exception:
        return None


def _detect_delimiter(header_line: str) -> str:
    # Heuristic: choose delimiter that appears more often.
    comma = header_line.count(",")
    semi = header_line.count(";")
    tab = header_line.count("\t")
    if tab > comma and tab > semi:
        return "\t"
    if semi > comma:
        return ";"
    return ","


def _find_header_index(headers: list[str], patterns: Iterable[str]) -> int | None:
    low = [h.lower() for h in headers]
    for i, h in enumerate(low):
        for p in patterns:
            if re.search(p, h, flags=re.IGNORECASE):
                return i
    return None


def _parse_rows_from_csv_text(csv_text: str) -> list[list[str]]:
    # Take a portion that looks like CSV.
    lines = [ln for ln in (csv_text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    # Find a likely header line by detecting known column words.
    header_idx = None
    for i, ln in enumerate(lines[:200]):
        if re.search(r"(лид|потрач|цена|ctr|cpl|сv|cv|показы|переход)", ln, flags=re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0

    header_line = lines[header_idx]
    delim = _detect_delimiter(header_line)

    reader = csv.reader(lines[header_idx:], delimiter=delim)
    rows = []
    for row in reader:
        if not row:
            continue
        # Keep row length consistent enough.
        if len(row) < 2:
            continue
        rows.append(row)
    return rows


def compute_kpis_from_csv(csv_text: str, *, question: str) -> TableKpis:
    year = _detect_year(question)

    rows = _parse_rows_from_csv_text(csv_text)
    if not rows:
        return TableKpis(
            year=year,
            leads=None,
            spent=None,
            cpl=None,
            ctr=None,
            cv=None,
            cpl_qual=None,
        )

    headers = [h.strip() for h in rows[0]]
    data_rows = rows[1:]

    idx_date = _find_header_index(headers, [r"дата"])
    idx_leads = _find_header_index(headers, [r"лид"])
    idx_spent = _find_header_index(headers, [r"потрач", r"стоим", r"cost"])
    idx_cpl = _find_header_index(headers, [r"cpl", r"цена\s*лида", r"цена\s*лид"])
    idx_ctr = _find_header_index(headers, [r"ctr"])
    idx_clicks = _find_header_index(headers, [r"переход"])
    idx_impr = _find_header_index(headers, [r"показы|impress"])
    idx_cv = _find_header_index(headers, [r"\bcv\b", r"cv%", r"cvr|конверс"])
    idx_cpl_qual = _find_header_index(headers, [r"cpl\s*квал", r"cpl\s*qual"])

    total_leads = 0
    total_spent = 0.0
    total_clicks = 0.0
    total_impr = 0.0
    any_rows = 0
    cpl_qual_vals: list[float] = []
    cv_vals: list[float] = []

    for r in data_rows:
        if len(r) < len(headers):
            # tolerate short rows; skip if header parsing seems broken.
            continue

        if year is not None and idx_date is not None:
            date_raw = (r[idx_date] or "").strip()
            # Expect dd.mm.yyyy or yyyy-mm-dd.
            y = None
            m1 = re.search(r"\b(19|20)\d{2}\b", date_raw)
            if m1:
                y = int(m1.group(0))
            if y is not None and y != year:
                continue

        any_rows += 1

        if idx_leads is not None:
            v = _norm_number(r[idx_leads])
            if v is not None:
                total_leads += int(round(v))

        if idx_spent is not None:
            v = _norm_number(r[idx_spent])
            if v is not None:
                total_spent += v

        if idx_clicks is not None:
            v = _norm_number(r[idx_clicks])
            if v is not None:
                total_clicks += v

        if idx_impr is not None:
            v = _norm_number(r[idx_impr])
            if v is not None:
                total_impr += v

        if idx_cpl_qual is not None:
            v = _norm_number(r[idx_cpl_qual])
            if v is not None:
                cpl_qual_vals.append(v)

        if idx_cv is not None:
            v = _norm_number(r[idx_cv])
            if v is not None:
                cv_vals.append(v)

    spent = total_spent if any_rows > 0 else None
    leads = total_leads if any_rows > 0 else None

    cpl = None
    if spent is not None and leads is not None and leads > 0:
        cpl = spent / leads

    ctr = None
    if total_impr > 0 and idx_clicks is not None and idx_impr is not None:
        ctr = (total_clicks / total_impr) * 100.0

    cv = None
    if cv_vals:
        # Prefer mean for CV/CVR; it's usually already a percentage per row.
        cv = sum(cv_vals) / len(cv_vals)

    cpl_qual = None
    if cpl_qual_vals:
        cpl_qual = sum(cpl_qual_vals) / len(cpl_qual_vals)

    return TableKpis(
        year=year,
        leads=leads,
        spent=spent,
        cpl=cpl,
        ctr=ctr,
        cv=cv,
        cpl_qual=cpl_qual,
    )


def format_kpis_for_numbers(kpis: TableKpis) -> str:
    def fmt_int(v: float) -> str:
        return str(int(round(v)))

    def fmt_money(v: float) -> str:
        # Keep user-friendly formatting: 2 decimals only if needed.
        if abs(v - int(v)) < 1e-9:
            return f"{int(v)} ₽"
        return f"{v:.2f}".replace(".", ",") + " ₽"

    def fmt_percent(v: float) -> str:
        if abs(v - int(v)) < 1e-9:
            return f"{int(v)}%"
        return f"{v:.2f}".replace(".", ",") + "%"

    lines: list[str] = []

    if kpis.leads is not None:
        lines.append(f"Лиды: {kpis.leads}")
    if kpis.cpl is not None:
        lines.append(f"KPI по CPL: {fmt_money(kpis.cpl)}")
    if kpis.cpl_qual is not None:
        lines.append(f"KPI по CPL квал: {fmt_money(kpis.cpl_qual)}")
    if kpis.ctr is not None:
        lines.append(f"CTR: {fmt_percent(kpis.ctr)}")
    if kpis.cv is not None:
        lines.append(f"CV: {fmt_percent(kpis.cv)}")
    if kpis.spent is not None:
        lines.append(f"Потрачено: {fmt_money(kpis.spent)}")

    # Keep max 4 metric lines.
    # If question included CTR/CPL explicitly, those will typically appear earlier.
    return "\n".join(lines[:4])


def build_numbers_from_link_context(question: str, context_messages: list[str]) -> str | None:
    # Try to parse any link blocks as CSV and compute KPI deterministically.
    link_csv_candidates: list[str] = []
    for line in context_messages:
        if line.startswith("[LINK] "):
            # "[LINK] <url>: <text>"
            parts = line.split(":", 1)
            if len(parts) == 2:
                link_text = parts[1].strip()
                # CSV-like: contains commas/semicolons and header words.
                if re.search(r"(Дата|дата|Лиды|Показы|CTR|CPL|Потрачено)", link_text, flags=re.IGNORECASE):
                    link_csv_candidates.append(link_text)

    if not link_csv_candidates:
        return None

    # Use the first parseable one.
    for csv_text in link_csv_candidates:
        kpis = compute_kpis_from_csv(csv_text, question=question)
        if kpis.leads is not None or kpis.cpl is not None or kpis.ctr is not None:
            out = format_kpis_for_numbers(kpis)
            if out.strip():
                return out
    return None

