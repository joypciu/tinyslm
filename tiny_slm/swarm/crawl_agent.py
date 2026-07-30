"""Crawl sub-agent: bounded HTTP fetch + main-text extraction."""

from __future__ import annotations

import html as html_lib
import re
import urllib.error
import urllib.request
from typing import List, Optional, Set
from urllib.parse import urlparse

from tiny_slm.search import SearchHit
from tiny_slm.swarm.vector_store import Chunk

_USER_AGENT = "TinySLM-Swarm/1.0 (+local research agent; educational)"
_MAX_BYTES = 500_000
_TIMEOUT = 8.0


def _allowed_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False
    low = url.lower()
    if any(low.endswith(ext) for ext in (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".exe")):
        return False
    return True


def _strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url_text(url: str, timeout: float = _TIMEOUT) -> Optional[str]:
    if not _allowed_url(url):
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ctype and "text" not in ctype and ctype:
                return None
            data = resp.read(_MAX_BYTES + 1)
            if len(data) > _MAX_BYTES:
                data = data[:_MAX_BYTES]
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype)
            if m:
                charset = m.group(1)
            raw = data.decode(charset, errors="replace")
            return _strip_html(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None
    except Exception:
        return None


def _chunk_text(text: str, title: str, url: str, subgoal: str, size: int = 420) -> List[Chunk]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) < 80:
        return []
    out: List[Chunk] = []
    step = max(200, size - 80)
    for i in range(0, min(len(text), size * 6), step):
        piece = text[i : i + size].strip()
        if len(piece) < 60:
            continue
        out.append(Chunk(text=piece, url=url, title=title, subgoal=subgoal))
        if len(out) >= 8:
            break
    return out


def crawl_hits(
    hits: List[SearchHit],
    subgoal: str,
    *,
    max_pages: int = 3,
    seen: Optional[Set[str]] = None,
) -> List[Chunk]:
    """Crawl up to max_pages unique URLs from search hits; always keep snippet chunks."""
    seen = seen if seen is not None else set()
    chunks: List[Chunk] = []
    pages = 0
    for h in hits:
        title = h.title or ""
        body = h.body or ""
        url = (h.href or "").strip()
        # Snippet chunk even without crawl
        snip = f"{title}. {body}".strip()
        if len(snip) >= 40:
            chunks.append(Chunk(text=snip[:500], url=url, title=title, subgoal=subgoal))
        if pages >= max_pages:
            continue
        if not url or url in seen or not _allowed_url(url):
            continue
        seen.add(url)
        text = fetch_url_text(url)
        pages += 1
        if text:
            chunks.extend(_chunk_text(text, title=title, url=url, subgoal=subgoal))
    return chunks
