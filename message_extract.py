from __future__ import annotations

from typing import Any


def _maybe_int(x: Any) -> int | None:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def extract_media_metadata(message: Any) -> dict[str, Any] | None:
    """
    Extract basic media metadata so we can still "analyze" media presence + captions
    without doing OCR/transcription.
    """
    # aiogram Message exposes different attributes depending on content type.
    if getattr(message, "photo", None):
        photos = list(message.photo)
        # Pick the largest by file_size if present; otherwise by area.
        best = None
        best_score = -1
        for p in photos:
            size = _maybe_int(getattr(p, "file_size", None))
            w = _maybe_int(getattr(p, "width", None)) or 0
            h = _maybe_int(getattr(p, "height", None)) or 0
            score = (size or 0) * 10_000 + (w * h)
            if score > best_score:
                best_score = score
                best = p

        if best is None:
            return {"type": "photo"}
        return {
            "type": "photo",
            "file_id": best.file_id,
            "file_unique_id": best.file_unique_id,
            "file_size": _maybe_int(best.file_size),
            "width": _maybe_int(best.width),
            "height": _maybe_int(best.height),
        }

    if getattr(message, "video", None):
        v = message.video
        return {
            "type": "video",
            "file_id": v.file_id,
            "file_unique_id": v.file_unique_id,
            "file_size": _maybe_int(getattr(v, "file_size", None)),
            "mime_type": getattr(v, "mime_type", None),
            "width": _maybe_int(getattr(v, "width", None)),
            "height": _maybe_int(getattr(v, "height", None)),
        }

    if getattr(message, "document", None):
        d = message.document
        return {
            "type": "document",
            "file_id": d.file_id,
            "file_unique_id": d.file_unique_id,
            "file_size": _maybe_int(getattr(d, "file_size", None)),
            "mime_type": getattr(d, "mime_type", None),
            "name": getattr(d, "file_name", None),
        }

    if getattr(message, "audio", None):
        a = message.audio
        return {
            "type": "audio",
            "file_id": a.file_id,
            "file_unique_id": a.file_unique_id,
            "file_size": _maybe_int(getattr(a, "file_size", None)),
            "mime_type": getattr(a, "mime_type", None),
            "title": getattr(a, "title", None),
        }

    if getattr(message, "voice", None):
        v = message.voice
        return {
            "type": "voice",
            "file_id": v.file_id,
            "file_unique_id": v.file_unique_id,
            "file_size": _maybe_int(getattr(v, "file_size", None)),
            "duration": _maybe_int(getattr(v, "duration", None)),
        }

    if getattr(message, "sticker", None):
        s = message.sticker
        return {
            "type": "sticker",
            "file_id": s.file_id,
            "file_unique_id": s.file_unique_id,
            "file_size": _maybe_int(getattr(s, "file_size", None)),
            "emoji": getattr(s, "emoji", None),
        }

    # Other types exist (location/contact/etc). We keep it simple for now.
    return None

