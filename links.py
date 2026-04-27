from __future__ import annotations

import re
from typing import Iterable

import httpx
from bs4 import BeautifulSoup


_URL_RE = re.compile(
    r"(?P<url>https?://[^\s]+|www\.[^\s]+)",
    flags=re.IGNORECASE,
)
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_BROWSER_HEADERS = {
    "User-Agent": _BROWSER_USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls: list[str] = []
    for m in _URL_RE.finditer(text):
        u = m.group("url").strip().strip(").,;")
        if u.lower().startswith("www."):
            u = "https://" + u
        # Strip trailing punctuation and markdown punctuation.
        u = u.strip()
        urls.append(u)
    # de-dupe preserving order
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    s = _clean_text(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _maybe_google_sheet_csv_url(url: str) -> str | None:
    # Example:
    # https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return None
    sheet_id = m.group(1)
    # Public sheets usually work with export. For private sheets, this fails (401/403).
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def _maybe_google_drive_download_url(url: str) -> str | None:
    # Example:
    # https://drive.google.com/file/d/<ID>/view
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return None
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _maybe_google_drive_folder_id(url: str) -> str | None:
    # Examples:
    # https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
    # https://drive.google.com/drive/folders/<FOLDER_ID>
    m = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        return None
    return m.group(1)


def _build_google_drive_file_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


async def fetch_google_drive_folder_image_urls(
    url: str,
    *,
    client: httpx.AsyncClient,
    timeout_s: int,
    max_images: int,
) -> list[str]:
    """
    Extract image file IDs from a public/accessible Google Drive folder page.
    Works best for folders that are publicly accessible without additional auth.
    """
    folder_id = _maybe_google_drive_folder_id(url)
    if not folder_id:
        return []

    # Fetch folder HTML.
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        resp = await client.get(folder_url, timeout=timeout_s, follow_redirects=True)
    except Exception:
        return []

    html = resp.text or ""

    # Collect candidate file IDs using multiple patterns.
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)/view",
        r"/file/d/([a-zA-Z0-9_-]+)/preview",
        r"uc\?export=download&id=([a-zA-Z0-9_-]+)",
        r"export=download&id=([a-zA-Z0-9_-]+)",
    ]
    candidates: list[str] = []
    for p in patterns:
        for m in re.finditer(p, html):
            fid = m.group(1)
            if fid and fid not in candidates:
                candidates.append(fid)

    # If we didn't find candidates, nothing more to do.
    if not candidates:
        return []

    # Validate candidates by checking content-type (cheap for only a few images).
    image_urls: list[str] = []
    seen: set[str] = set()

    async def is_image_file(file_id: str) -> bool:
        download_url = _build_google_drive_file_download_url(file_id)
        try:
            r = await client.get(download_url, timeout=timeout_s, follow_redirects=True)
            ct = (r.headers.get("content-type") or "").lower()
            return ct.startswith("image/")
        except Exception:
            return False

    for fid in candidates:
        if fid in seen:
            continue
        seen.add(fid)
        if await is_image_file(fid):
            image_urls.append(_build_google_drive_file_download_url(fid))
            if len(image_urls) >= max_images:
                break

    return image_urls


async def fetch_url_text(
    url: str,
    *,
    client: httpx.AsyncClient,
    timeout_s: int,
    max_chars: int,
) -> tuple[str, str | None]:
    """
    Returns: (effective_url, extracted_text_or_none)
    """
    effective_url = url
    sheet_csv = _maybe_google_sheet_csv_url(url)
    if sheet_csv:
        effective_url = sheet_csv

    drive_download = _maybe_google_drive_download_url(url)
    if drive_download:
        effective_url = drive_download

    attempts = [
        {},
        _BROWSER_HEADERS,
        {**_BROWSER_HEADERS, "Referer": "https://www.google.com/"},
    ]
    resp: httpx.Response | None = None
    for extra_headers in attempts:
        try:
            resp = await client.get(
                effective_url,
                timeout=timeout_s,
                follow_redirects=True,
                headers=extra_headers or None,
            )
        except Exception:
            resp = None
            continue

        # If upstream returns explicit error page, try one more browser-like attempt.
        if resp.status_code >= 400:
            if resp.status_code in {403, 429, 503}:
                continue
            return effective_url, None
        break

    if resp is None or resp.status_code >= 400:
        return effective_url, None

    ct = (resp.headers.get("content-type") or "").lower()
    if "text/csv" in ct or "application/csv" in ct:
        text = resp.text
        # CSV can be huge; for analytics tables we want both the header
        # (column meanings) and the beginning/end (often early periods vs later periods).
        lines = text.splitlines()
        lines = [ln for ln in lines if ln.strip()]
        if len(lines) <= 2:
            return effective_url, _truncate(text, max_chars)

        header = lines[0]
        head = lines[1 : 201]
        tail = lines[-200:] if len(lines) > 201 else []
        omitted = max(0, len(lines) - (1 + len(head) + len(tail)))

        if omitted > 0 and tail and tail != head:
            combined = "\n".join([header] + head + [f"... [omitted {omitted} lines] ..."] + tail)
        else:
            combined = "\n".join([header] + head + tail)

        return effective_url, _truncate(combined, max_chars)

    if "text/html" in ct or "application/xhtml+xml" in ct or "<html" in resp.text[:200].lower():
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        low = text.lower()
        # Protect LLM from anti-bot placeholder pages.
        if (
            "403 forbidden" in low
            or "access denied" in low
            or "доступ запрещ" in low
        ) and len(text) < 1000:
            return effective_url, None
        return effective_url, _truncate(text, max_chars)

    # If it looks like plain text.
    if "text/plain" in ct:
        return effective_url, _truncate(resp.text, max_chars)

    # Unsupported content types (pdf, docx, etc.) -> return None.
    return effective_url, None

