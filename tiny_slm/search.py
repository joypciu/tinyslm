"""DuckDuckGo web search helper (no API key) + extractive answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


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
        r"^(please\s+)?(can you\s+|could you\s+)?(search\s+(the\s+web\s+for\s+|for\s+)?"
        r"|look\s+up\s+|google\s+|find\s+(out\s+)?|tell me\s+|explain\s+)",
        "",
        q,
        flags=re.I,
    ).strip()
    q = re.sub(r"\b(please|thanks|thank you|briefly|simply)\b", " ", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" ?!.")
    # "who is X" / "what is X" keep the subject
    m = re.match(r"^(who is|what is|what's|whats)\s+(.+)$", q, flags=re.I)
    if m:
        q = m.group(2).strip()
    return q or (user_message or "").strip()


@dataclass
class SearchHit:
    title: str
    body: str
    href: str


def _hits_from_raw(results) -> List[SearchHit]:
    hits: List[SearchHit] = []
    for r in results or []:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        href = (r.get("href") or r.get("link") or "").strip()
        if title or body:
            hits.append(SearchHit(title=title, body=body, href=href))
    return hits


def search_web_hits(query: str, max_results: int = 5) -> List[SearchHit]:
    """Fetch structured DuckDuckGo hits (news endpoint when the ask is newsy)."""
    query = clean_search_query(query)
    if not query:
        return []
    DDGS = _get_ddgs()
    want_news = any(
        w in query.lower()
        for w in ("news", "latest", "breaking", "today", "headline")
    )
    try:
        try:
            with DDGS() as ddgs:
                results = []
                if want_news and hasattr(ddgs, "news"):
                    try:
                        results = list(ddgs.news(query, max_results=max_results))
                    except Exception:
                        results = []
                if not results:
                    results = list(ddgs.text(query, max_results=max_results))
        except TypeError:
            ddgs = DDGS()
            results = []
            if want_news and hasattr(ddgs, "news"):
                try:
                    results = list(ddgs.news(query, max_results=max_results))
                except Exception:
                    results = []
            if not results:
                results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []
    return _hits_from_raw(results)


def format_hits(hits: List[SearchHit], max_chars: int = 900) -> str:
    if not hits:
        return "(no search results)"
    lines: List[str] = []
    size = 0
    for i, h in enumerate(hits, 1):
        block = f"{i}. {h.title}\n{h.body}\n{h.href}".strip()
        if size + len(block) > max_chars and lines:
            break
        lines.append(block)
        size += len(block)
    return "\n\n".join(lines)


def search_web(query: str, max_results: int = 4) -> str:
    """Return a short plain-text digest of DuckDuckGo results."""
    query = clean_search_query(query)
    if not query:
        return ""
    try:
        hits = search_web_hits(query, max_results=max_results)
        # One retry with a simpler query if the first pass is empty
        if not hits and " " in query:
            short = " ".join(query.split()[:6])
            if short != query:
                hits = search_web_hits(short, max_results=max_results)
    except Exception as exc:
        return f"(search failed: {exc})"
    if not hits:
        return "(no search results)"
    return format_hits(hits)


def needs_search(user_message: str) -> bool:
    """Heuristic: live lookup / current events / open factual asks."""
    msg = (user_message or "").lower().strip()
    if not msg:
        return False
    # Skip pure chat / memory / math-ish
    if any(
        w in msg
        for w in (
            "using memory",
            "remember this",
            "hello",
            "hi!",
            "thanks",
            "i'm bored",
            "how are you",
        )
    ):
        if not any(t in msg for t in ("search", "look up", "news", "latest")):
            return False

    triggers = [
        "search",
        "look up",
        "google",
        "duckduckgo",
        "what is the latest",
        "latest news",
        "current events",
        "breaking news",
        "when did",
        "news about",
        "price of",
        "weather in",
        "weather today",
        "as of 2024",
        "as of 2025",
        "as of 2026",
        "in 2024",
        "in 2025",
        "in 2026",
        "according to",
        "on the internet",
        "from the web",
    ]
    if any(t in msg for t in triggers):
        return True
    if re.search(r"\bwho is\b", msg) and not re.search(
        r"\bwho is (you|this|that|it|tinyslm)\b", msg
    ):
        return True
    # Open knowledge asks the tiny model usually fails on
    if re.search(
        r"\b(explain|why do|why does|why is|what causes|how does|what happens)\b",
        msg,
    ):
        return True
    # "how do" for world knowledge, but not coding ("how do I write a function")
    if re.search(r"\bhow do\b", msg) and not re.search(
        r"\b(python|code|function|program|script|loop|variable)\b", msg
    ):
        return True
    if re.search(r"\bwhat is\b", msg) and len(msg) > 12:
        # FAQ may catch common ones first; remaining "what is" → web
        return True
    return False


def _best_sentences(text: str, limit: int = 2) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        # Strip DDG relative-time prefixes: "5 hours ago · ..."
        p = re.sub(
            r"^\d+\s+(minutes?|hours?|days?|weeks?)\s+ago\s*[·|\-:]?\s*",
            "",
            p,
            flags=re.I,
        ).strip()
        if len(p) < 25:
            continue
        if p.lower().startswith("http"):
            continue
        # Prefer informative / fact-dense sentences
        score = 0
        if re.search(r"\b(is|are|was|were|means|refers|caused|because|elected|born)\b", p, re.I):
            score += 2
        if re.search(r"\d{4}|\d+%|\$\d+", p):
            score += 3
        elif re.search(r"\d", p):
            score += 1
        if re.search(r"\b(according to|reported|announced|founded)\b", p, re.I):
            score += 1
        if len(p) > 200:
            p = p[:197].rstrip() + "..."
        out.append((score, p))
    out.sort(key=lambda x: (-x[0], -len(x[1])))
    return [p for _, p in out[:limit]]


def answer_from_search(
    digest: str,
    max_chars: int = 420,
    query: str = "",
) -> str:
    """Build a grounded multi-snippet reply from a DuckDuckGo digest."""
    text = (digest or "").strip()
    if not text or text.startswith("(search failed") or text.startswith("(no search"):
        return ""

    # Parse numbered hits
    blocks = re.split(r"\n\s*\n", text)
    snippets: List[str] = []
    titles: List[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        title = re.sub(r"^\d+\.\s*", "", lines[0]).strip()
        body_lines = [ln for ln in lines[1:] if not ln.startswith("http")]
        body = " ".join(body_lines).strip()
        if title:
            titles.append(title)
        for sent in _best_sentences(body, limit=2):
            # "Also known as Ada Lovelace, was an English..." -> "Ada Lovelace was..."
            m = re.match(
                r"^(?:also )?known as\s+(.+?),\s*(was|is|were)\s+(.+)$",
                sent,
                re.I,
            )
            if m:
                sent = f"{m.group(1).strip()} {m.group(2)} {m.group(3)}"
            elif re.match(r"^(born|elected)\b", sent, re.I) and title:
                subj = title.split("-")[0].strip()
                if subj.lower() not in sent.lower():
                    sent = f"{subj} {sent[0].lower() + sent[1:]}"
            snippets.append(sent)
        if not body and title and len(title) > 20:
            snippets.append(title.rstrip(".") + ".")

    # Dedupe near-identical sentences
    uniq: List[str] = []
    for s in snippets:
        key = re.sub(r"[^a-z0-9]+", "", s.lower())[:48]
        if any(key and key in re.sub(r"[^a-z0-9]+", "", u.lower()) for u in uniq):
            continue
        uniq.append(s)
        if len(uniq) >= 4:
            break

    if not uniq and titles:
        return f"From the web: {titles[0]}."[:max_chars]
    if not uniq:
        return ""

    # Prefer sentences that overlap query keywords / proper names
    q_terms = {
        t
        for t in re.findall(r"[a-z0-9]+", clean_search_query(query or "").lower())
        if len(t) > 2 and t not in {"what", "who", "when", "where", "does", "about", "latest", "news"}
    }
    if q_terms:
        uniq.sort(
            key=lambda s: (
                -sum(1 for t in q_terms if t in s.lower()),
                -len(s),
            )
        )
        # If top sentence ignores the query, try pairing with a title that matches
        top = uniq[0].lower()
        if sum(1 for t in q_terms if t in top) == 0:
            for t in titles:
                if any(term in t.lower() for term in q_terms) and len(t) > 15:
                    uniq.insert(0, t.rstrip(".") + ".")
                    break

    out = f"From the web: {uniq[0]}"
    if len(uniq) > 1 and uniq[1].lower() != uniq[0].lower():
        out += " Also: " + uniq[1]
    return out[:max_chars]


def should_prefer_web_answer(user: str) -> bool:
    """True when an extractive web answer should beat a neural draft."""
    u = (user or "").lower()
    return needs_search(u) or any(
        w in u for w in ("search", "look up", "latest", "news", "who is", "when did", "explain")
    )
