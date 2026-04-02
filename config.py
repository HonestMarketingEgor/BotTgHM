import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


def _clean_env_value(value: str) -> str:
    """
    Trim whitespace and remove accidental surrounding quotes.
    This makes .env editing less fragile (e.g., TOKEN="abc:def").
    """
    v = value.strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1].strip()
    return v


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str
    db_path: str
    openai_model: str
    openai_model_suggestions: str

    daily_hour: int
    daily_minute: int
    # IANA timezone name, e.g. Europe/Moscow — used for daily summary wall-clock time.
    daily_timezone: str

    max_messages_for_ask: int
    max_context_chars: int

    allowed_chat_ids: list[int]

    url_fetch_timeout_s: int
    max_link_chars: int
    max_links: int
    max_project_memory_chars: int
    enable_auto_response_mode: bool
    default_mode: str
    enable_chat_analysis: bool


def load_config() -> Config:
    # Load variables from .env (if present) so local setup is simple.
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path)

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not telegram_bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    telegram_bot_token = _clean_env_value(telegram_bot_token)

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_api_key = _clean_env_value(openai_api_key)
    openai_base_url = _clean_env_value(os.getenv("OPENAI_BASE_URL", ""))

    db_path = os.getenv("DB_PATH", "./data/bot.db").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    openai_model_suggestions = os.getenv("OPENAI_MODEL_SUGGESTIONS", openai_model).strip()

    daily_hour = _env_int("DAILY_HOUR", 19)
    daily_minute = _env_int("DAILY_MINUTE", 0)
    daily_timezone = _clean_env_value(os.getenv("DAILY_TIMEZONE", "Europe/Moscow"))
    try:
        ZoneInfo(daily_timezone)
    except Exception as e:
        raise RuntimeError(
            f"Invalid DAILY_TIMEZONE={daily_timezone!r} (use IANA name e.g. Europe/Moscow)"
        ) from e

    # Defaults are chosen to keep prompt size (and OpenAI cost) bounded.
    max_messages_for_ask = _env_int("MAX_MESSAGES_FOR_ASK", 25)
    max_context_chars = _env_int("MAX_CONTEXT_CHARS", 6000)

    allowed_raw = os.getenv("ALLOWED_CHAT_IDS", "").strip()
    allowed_chat_ids: list[int] = []
    if allowed_raw:
        allowed_chat_ids = [int(x.strip()) for x in allowed_raw.split(",") if x.strip()]

    return Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        db_path=db_path,
        openai_model=openai_model,
        openai_model_suggestions=openai_model_suggestions,
        daily_hour=daily_hour,
        daily_minute=daily_minute,
        daily_timezone=daily_timezone,
        max_messages_for_ask=max_messages_for_ask,
        max_context_chars=max_context_chars,
        allowed_chat_ids=allowed_chat_ids,
        url_fetch_timeout_s=_env_int("URL_FETCH_TIMEOUT_S", 20),
        max_link_chars=_env_int("MAX_LINK_CHARS", 30000),
        max_links=_env_int("MAX_LINKS", 3),
        max_project_memory_chars=_env_int("MAX_PROJECT_MEMORY_CHARS", 1800),
        enable_auto_response_mode=_env_bool("ENABLE_AUTO_RESPONSE_MODE", True),
        default_mode=_clean_env_value(os.getenv("DEFAULT_MODE", "assistant_mode")),
        enable_chat_analysis=_env_bool("ENABLE_CHAT_ANALYSIS", True),
    )

