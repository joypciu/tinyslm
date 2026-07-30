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

_CHROME_RE = re.compile(
    r"(?i)\b("
    r"sign\s*in|sign\s*up|log\s*in|subscribe|newsletter|cookie|privacy policy|"
    r"terms of (service|use)|sitemap|open in app|get app|write a comment|"
    r"related articles?|trending|refcards?|video library|edit profile|"
    r"manage email|how to post|submission guidelines|follow us|share this|"
    r"accept (all )?cookies|advertisement|sponsored|"
    r"save this for your next|min read|table of contents|skip to content"
    r")\b"
)


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
    if any(
        low.endswith(ext)
        for ext in (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".exe")
    ):
        return False
    # Low-signal hosts for grounded research
    host = (p.netloc or "").lower()
    if any(
        bad in host
        for bad in (
            "youtube.com",
            "youtu.be",
            "tiktok.com",
            "pinterest.com",
            "facebook.com",
            "instagram.com",
        )
    ):
        return False
    return True


def _extract_main_html(raw: str) -> str:
    """Prefer <article>/<main> blocks when present."""
    for pat in (
        r"(?is)<article\b[^>]*>(.*?)</article>",
        r"(?is)<main\b[^>]*>(.*?)</main>",
        r"(?is)<div[^>]+role=[\"']main[\"'][^>]*>(.*?)</div>",
    ):
        m = re.search(pat, raw)
        if m and len(m.group(1)) > 400:
            return m.group(1)
    return raw


def _strip_html(raw: str) -> str:
    raw = _extract_main_html(raw)
    text = re.sub(r"(?is)<(script|style|noscript|svg|iframe|nav|footer|header).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    lines = []
    for line in re.split(r"[\n\r]+", text):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if len(line) < 40:
            continue
        if _CHROME_RE.search(line):
            continue
        # Drop link-farm lines (too many short tokens / pipes)
        if line.count("|") >= 3 or line.count("•") >= 4:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_plain(text: str) -> str:
    lines = []
    for line in re.split(r"[\n\r]+", text or ""):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 40:
            continue
        if _CHROME_RE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


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
    text = _clean_plain(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        return []
    out: List[Chunk] = []
    step = max(200, size - 80)
    for i in range(0, min(len(text), size * 8), step):
        piece = text[i : i + size].strip()
        if len(piece) < 80:
            continue
        if _CHROME_RE.search(piece):
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
        snip = _clean_plain(f"{title}. {body}")
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
