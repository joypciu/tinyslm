"""DuckDuckGo web search helper (no API key)."""

from __future__ import annotations

import re
from typing import List


def _get_ddgs():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError as e:
            raise ImportError("Install search backend: pip install ddgs") from e


def clean_search_query(user_message: str) -> str:
    """Strip chat fluff so DuckDuckGo gets a usable query."""
    q = (user_message or "").strip()
    q = re.sub(
        r"^(please\s+)?(search\s+(the\s+web\s+for\s+|for\s+)?|look\s+up\s+|google\s+|find\s+)",
        "",
        q,
        flags=re.I,
    ).strip()
    return q or user_message.strip()


def search_web(query: str, max_results: int = 3) -> str:
    """Return a short plain-text digest of DuckDuckGo results."""
    query = clean_search_query(query)
    if not query:
        return ""

    DDGS = _get_ddgs()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except TypeError:
        # Some versions don't use context manager
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            return f"(search failed: {exc})"
    except Exception as exc:
        return f"(search failed: {exc})"

    if not results:
        return "(no search results)"

    lines: List[str] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        href = (r.get("href") or r.get("link") or "").strip()
        lines.append(f"{i}. {title}\n{body}\n{href}")
    return "\n\n".join(lines)


def needs_search(user_message: str) -> bool:
    """Heuristic: current events / factual lookup cues."""
    msg = user_message.lower()
    triggers = [
        "search",
        "look up",
        "google",
        "duckduckgo",
        "what is the latest",
        "who is",
        "when did",
        "news",
        "current",
        "today",
        "2024",
        "2025",
        "2026",
        "price of",
        "weather",
    ]
    return any(t in msg for t in triggers)


def answer_from_search(digest: str, max_chars: int = 280) -> str:
    """Build a short grounded reply from a DuckDuckGo digest (no LM rewrite)."""
    text = (digest or "").strip()
    if not text or text.startswith("(search failed") or text.startswith("(no search"):
        return ""
    # Prefer first result body line after "1. title"
    m = re.search(r"^1\.\s*(.+)$\n(.+)$", text, re.M)
    if m:
        title = m.group(1).strip()
        body = m.group(2).strip()
        # skip URL-only body
        if body.startswith("http"):
            body = ""
        if body:
            snip = body.split(". ")[0].strip().rstrip(".") + "."
            out = f"From the web: {snip}"
            if title and title.lower() not in snip.lower():
                out = f"From the web ({title}): {snip}"
            return out[:max_chars]
        if title:
            return f"From the web: {title}."[:max_chars]
    # Fallback: first non-empty non-url line
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("http") or re.match(r"^\d+\.\s*$", line):
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        if len(line) >= 12:
            return ("From the web: " + line.rstrip(".") + ".")[:max_chars]
    return ""
