"""SLM agent helper: research league teams via search tools → JSON records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tiny_slm.search import search_web_hits

# Known Saudi soccer competitions the agent should cover.
_SAUDI_LEAGUE_QUERIES: List[Tuple[str, str]] = [
    ("Saudi Arabia - Saudi Pro League", "Saudi Pro League teams clubs Al-Hilal Al-Nassr Al-Ahli"),
    ("Saudi Arabia - Saudi Pro League", "list of Saudi Pro League clubs"),
    (
        "Saudi Arabia - Saudi First Division League",
        "Saudi First Division League Yelo League clubs list",
    ),
    (
        "Saudi Arabia - Saudi Second Division League",
        "Saudi Second Division League clubs teams list",
    ),
    (
        "Saudi Arabia - Saudi Women's Premier League",
        "Saudi Women's Premier League clubs teams list",
    ),
]

_STOP = {
    "the",
    "and",
    "for",
    "club",
    "fc",
    "sc",
    "saudi",
    "arabia",
    "league",
    "pro",
    "division",
    "premier",
    "first",
    "second",
    "women",
    "women's",
    "list",
    "teams",
    "team",
    "all",
    "clubs",
    "football",
    "soccer",
    "season",
    "table",
    "standings",
    "wikipedia",
    "official",
    "website",
}

_SAUDI_CITIES = (
    "Riyadh",
    "Jeddah",
    "Dammam",
    "Khobar",
    "Dhahran",
    "Mecca",
    "Makkah",
    "Medina",
    "Madinah",
    "Abha",
    "Tabuk",
    "Hail",
    "Buraydah",
    "Buraidah",
    "Khamis Mushait",
    "Najran",
    "Jazan",
    "Jizan",
    "Yanbu",
    "Al Bahah",
    "Al-Bahah",
    "Taif",
    "Al Taif",
    "Qatif",
    "Hofuf",
    "Al Hofuf",
    "Unaizah",
    "Sakakah",
    "Arar",
    "Bisha",
    "Neom",
    "Al Majmaah",
    "Al-Kharj",
    "Al Kharj",
)

_PLAYER_STOP = {
    "coach",
    "manager",
    "squad",
    "roster",
    "player",
    "players",
    "goalkeeper",
    "defender",
    "midfielder",
    "forward",
    "striker",
    "captain",
    "saudi",
    "arabia",
    "football",
    "soccer",
    "club",
    "team",
    "league",
    "season",
    "transfer",
    "wikipedia",
    "official",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "things",
    "know",
    "about",
    "yahoo",
    "sports",
    "get",
    "fivb",
    "men's",
    "mens",
    "volleyball",
    "world",
    "manchester",
    "united",
    "cup",
    "news",
    "live",
    "watch",
    "highlights",
    "fixture",
    "fixtures",
    "table",
    "standings",
    "transfermarkt",
    "sofascore",
    "flashscore",
    "premier",
    "division",
    "pro",
    "logo",
    "png",
    "home",
    "away",
    "match",
    "matches",
    "goals",
    "assists",
    "national",
    "international",
    "former",
    "current",
    "best",
    "top",
    "stars",
    "lineup",
    "kit",
    "jersey",
    "libyan",
    "english",
    "spanish",
    "french",
    "german",
    "italian",
    "city",
    "park",
    "policy",
    "privacy",
    "terms",
    "creators",
    "advertise",
    "developers",
    "demand",
    "positions",
    "sep",
    "atual",
    "primeira",
    "divisão",
    "divisao",
    "most",
    "hat",
    "tricks",
    "awwal",
    "real",
    "madrid",
    "barcelona",
    "arsenal",
    "chelsea",
    "liverpool",
    "bayern",
    "psg",
    "juventus",
    "milan",
    "inter",
    "tottenham",
    "leicester",
    "bi",
    "key",
    "positions",
    "eine",
    "option",
    "santos",
    "aveiro",
    "funchal",
    "altura",
    "portugal",
    "posición",
    "posicion",
    "derecho",
    "agente",
    "bristol",
    "today",
    "for",
    "option",
    "spain",
    "france",
    "brazil",
    "argentina",
    "germany",
    "italy",
    "england",
    "croatia",
    "serbia",
    "egypt",
    "morocco",
    "agency",
    "agent",
    "height",
    "weight",
    "born",
    "age",
    "left",
    "right",
    "winger",
    "market",
    "latest",
    "results",
    "egypt",
    "today",
    "option",
    "neom",
    "salary",
    "list",
    "sporting",
    "lisbon",
    "ranking",
    "history",
    "continental",
    "kit",
    "arm",
    "safety",
    "how",
    "transfers",
    "transfer",
    "update",
    "updates",
    "profile",
    "biography",
}

_ORG_WORDS = {
    "city",
    "united",
    "madrid",
    "barcelona",
    "arsenal",
    "chelsea",
    "liverpool",
    "bayern",
    "park",
    "stadium",
    "arena",
    "policy",
    "privacy",
    "terms",
    "sports",
    "fc",
    "sc",
    "club",
    "cup",
    "league",
}


def looks_league_collect(user: str) -> bool:
    u = (user or "").lower()
    if not any(w in u for w in ("saudi", "ksa")):
        return False
    if not any(w in u for w in ("league", "leagues", "team", "teams", "club", "clubs")):
        return False
    return any(
        w in u
        for w in (
            "collect",
            "gather",
            "research",
            "search",
            "find",
            "list",
            "json",
            "save",
            "information",
            "infos",
            "structure",
            "players",
            "player",
            "roster",
            "squad",
            "city",
            "logo",
            "enrich",
            "fill",
            "complete",
        )
    )


def wants_team_details(user: str) -> bool:
    u = (user or "").lower()
    return any(
        w in u
        for w in (
            "player",
            "players",
            "roster",
            "squad",
            "city",
            "logo",
            "enrich",
            "fill",
            "complete",
            "mascot",
            "details",
            "information",
        )
    )


def _abbrev(name: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", name) if w.lower() not in _STOP]
    if not words:
        words = re.findall(r"[A-Za-z0-9]+", name)[:3]
    if len(words) == 1:
        return words[0][:3].upper()
    letters = "".join(w[0] for w in words[:3])
    return (letters or name[:3]).upper()[:5]


def _team_id(name: str, league: str) -> str:
    raw = f"{name}|{league}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16].upper()


def _player_id(team_id: str, player_name: str) -> str:
    raw = f"{team_id}|{player_name}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16].upper()


def _nickname(name: str) -> str:
    n = re.sub(r"\b(FC|SC|Club)\b", "", name, flags=re.I)
    n = re.sub(r"\s+", " ", n).strip(" -")
    return n or name


def _normalize_club(name: str) -> str:
    n = re.sub(r"\s+", " ", (name or "").strip())
    n = re.split(r"\s*[—–\-]\s+", n, maxsplit=1)[0].strip()
    n = re.sub(r"\blive in the USA\b", "", n, flags=re.I).strip()
    n = re.sub(
        r"\b(live|news|score|fixture|match|today|vs|versus|table|wiki|all in action)\b.*$",
        "",
        n,
        flags=re.I,
    ).strip(" -|,.")
    bare = {
        "hilal": "Al-Hilal",
        "nassr": "Al-Nassr",
        "ahli": "Al-Ahli",
        "ittihad": "Al-Ittihad",
        "shabab": "Al-Shabab",
        "ettifaq": "Al-Ettifaq",
        "fateh": "Al-Fateh",
        "fayha": "Al-Fayha",
        "khaleej": "Al-Khaleej",
        "raed": "Al-Raed",
        "wehda": "Al-Wehda",
        "taawoun": "Al-Taawoun",
        "okhdood": "Al-Okhdood",
        "qadsiah": "Al-Qadsiah",
        "riyadh": "Al-Riyadh",
        "hazem": "Al-Hazem",
        "najma": "Al-Najma",
        "neom": "Neom SC",
        "neomsc": "Neom SC",
    }
    key = re.sub(r"[^a-z0-9]+", "", n.lower())
    if key in bare:
        return bare[key]
    if re.match(r"^Al\s+", n):
        n = re.sub(r"^Al\s+", "Al-", n)
    return n


def _is_plausible_club(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 42:
        return False
    if " " in name and len(name.split()) > 4:
        return False
    low = name.lower()
    if any(
        bad in low
        for bad in (
            "league",
            "division",
            "premier",
            "wikipedia",
            "transfer",
            "live in",
            "saudi arabia",
            "standings",
            "fixture",
            "all in",
            "action",
        )
    ):
        return False
    return bool(re.search(r"[A-Za-z]", name))


def _candidate_names(text: str) -> List[str]:
    """Pull club-like names from search titles/snippets."""
    found: List[str] = []
    patterns = [
        r"\bAl[-\s][A-Z][a-zA-Z]{2,}(?:\s+(?:FC|SC|SFC|Club))?\b",
        r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\s+(?:FC|SC|SFC)\b",
        r"\b(?:Neom|Okaz|Damac|Ettifaq|Fateh|Fayha|Khaleej|Okhdood|Qadsiah|Raed|Riyadh|Shabab|Wehda|Ittihad|Nassr|Hilal|Ahli|Taawoun|Hazem|Najma|Kholood|Orubah|Jabalain|Tai)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            found.append(m.group(0))
    for chunk in re.split(r"[,;|•·]|\band\b", text):
        chunk = chunk.strip()
        if re.match(r"^Al[-\s]", chunk) and 4 <= len(chunk) <= 40:
            found.append(chunk)

    out: List[str] = []
    seen = set()
    for n in found:
        n2 = _normalize_club(n)
        if not _is_plausible_club(n2):
            continue
        base = re.sub(r"\s+(FC|SC|SFC|Club)$", "", n2, flags=re.I).strip()
        key = re.sub(r"[^a-z0-9]+", "", base.lower())
        if key.startswith("al") and len(key) > 4:
            key = key[2:]
        if key in seen or len(key) < 4:
            continue
        seen.add(key)
        out.append(base)
    return out


def _extract_city(text: str, club: str = "") -> Optional[str]:
    # Prefer "based in / from / club of <city>"
    m = re.search(
        r"\b(?:based in|from|club of|city of)\s+("
        + "|".join(re.escape(c) for c in sorted(_SAUDI_CITIES, key=len, reverse=True))
        + r")\b",
        text,
        flags=re.I,
    )
    if m:
        city = m.group(1)
        for c in _SAUDI_CITIES:
            if c.lower() == city.lower():
                return c
        return city.title()
    # Prefer city near the club name mention
    club_pat = re.escape(club) if club else None
    if club_pat:
        for c in sorted(_SAUDI_CITIES, key=len, reverse=True):
            if re.search(
                rf".{{0,60}}{club_pat}.{{0,60}}\b{re.escape(c)}\b|\b{re.escape(c)}\b.{{0,60}}{club_pat}",
                text,
                flags=re.I | re.S,
            ):
                return c
    for c in _SAUDI_CITIES:
        if re.search(rf"\b{re.escape(c)}\b", text, flags=re.I):
            # Skip Neom unless club is Neom
            if c.lower() == "neom" and "neom" not in (club or "").lower():
                continue
            return c
    return None


def _extract_logo_url(hits, club: str = "") -> Optional[str]:
    club_keys = [
        k
        for k in re.findall(r"[a-z0-9]{3,}", (club or "").lower())
        if k not in {"the", "club", "saudi", "al"}
    ]
    scored: List[Tuple[int, str]] = []
    for h in hits:
        for url in re.findall(r"https?://[^\s\"'<>]+", f"{h.href}\n{h.body}\n{h.title}"):
            url = url.rstrip(".,);]")
            low = url.lower()
            score = 0
            if any(ext in low for ext in (".png", ".svg", ".jpg", ".jpeg", ".webp")):
                score += 8
            else:
                if not any(
                    x in low
                    for x in ("/images/", "/logo", "/badge", "/crest", "upload.wikimedia")
                ):
                    continue
                score += 2
            if "logo" in low:
                score += 6
            if "badge" in low or "crest" in low:
                score += 5
            if "upload.wikimedia.org" in low:
                score += 7
            if club_keys and any(k in low for k in club_keys):
                score += 10
            else:
                # Require club token for non-wiki or reject
                if "upload.wikimedia.org" not in low and not any(k in low for k in club_keys):
                    continue
                score -= 3
            if any(
                bad in low
                for bad in (
                    "pngdownload.io",
                    "/png-image/",
                    "shutterstock",
                    "neckarrems",
                    "mapfre",
                    "kit_left",
                    "kit_right",
                    "kit_body",
                )
            ):
                score -= 8
            if score > 0:
                scored.append((score, url))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_score, best_url = scored[0]
    # Must mention club or be a clear commons logo with club token in nearby isn't available — require club key
    low = best_url.lower()
    if club_keys and not any(k in low for k in club_keys):
        return None
    return best_url


def _is_plausible_player(name: str, club: str) -> bool:
    name = re.sub(r"\s+", " ", (name or "").strip(" .,;"))
    if not name or len(name) < 5 or len(name) > 36:
        return False
    parts = name.split()
    if len(parts) < 2 or len(parts) > 3:
        return False
    for p in parts:
        if len(p) > 12:
            return False
        if "-" in p and not re.match(r"^[A-Z][a-z]+-[A-Z][a-z]+$", p):
            return False
        if re.match(r"^[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']{1,16}$", p):
            continue
        if re.match(r"^[A-Z][a-z]+-[A-Z][a-z]+$", p):
            continue
        if p in {"Al", "El", "De", "Da", "Di", "Van", "Von", "Ben", "Ibn"}:
            continue
        return False
    # Reject glued junk like Bristoloday
    if any(re.search(r"(today|news|sport|market|option|results)$", p.lower()) for p in parts):
        return False
    low_parts = {p.lower().strip(".'") for p in parts}
    if low_parts & _PLAYER_STOP:
        return False
    if low_parts & _ORG_WORDS:
        return False
    low = name.lower()
    club_key = re.sub(r"[^a-z]+", "", club.lower())
    name_key = re.sub(r"[^a-z]+", "", low)
    if club_key and club_key in name_key:
        return False
    # Also reject when bare club stem appears (Al-Nassr → nassr)
    stem = club_key[2:] if club_key.startswith("al") and len(club_key) > 4 else club_key
    if stem and len(stem) >= 4 and stem in name_key:
        return False
    if any(stem and stem == p for p in low_parts):
        return False
    if any(
        bad in low
        for bad in (
            "saudi pro",
            "premier",
            "division",
            "football club",
            "sports club",
            "transfermarkt",
            "sofascore",
            "flashscore",
            "things to",
            "yahoo",
            "volleyball",
            "manchester",
            "libyan cup",
            "real madrid",
            "hat-trick",
            "privacy",
        )
    ):
        return False
    return True


def _extract_players(text: str, club: str, limit: int = 22) -> List[str]:
    found: List[Tuple[int, str]] = []
    for m in re.finditer(
        r"\b\d{1,2}[\).:\-]\s*([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']+){1,2})",
        text,
    ):
        found.append((5, m.group(1)))
    for m in re.finditer(
        r"\b(?:signed|joined|forward|striker|midfielder|defender|goalkeeper|winger|captain)\s+"
        r"([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']+){1,2})",
        text,
        flags=re.I,
    ):
        found.append((4, m.group(1)))
    # Transfermarkt-ish "Name Age Position"
    for m in re.finditer(
        r"\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']{2,14}\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']{2,14})\s+\d{1,2}\b",
        text,
    ):
        found.append((6, m.group(1)))
    for m in re.finditer(
        r"\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']{2,14}(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ']{2,14}){1,2})\b",
        text,
    ):
        # Low priority general names — only keep if later validated and we still need more
        found.append((1, m.group(1)))

    found.sort(key=lambda x: -x[0])
    out: List[str] = []
    seen = set()
    # Prefer high-confidence extractions first; only fill with general names if sparse
    high = [(s, n) for s, n in found if s >= 4]
    low = [(s, n) for s, n in found if s < 4]
    ordered = high + low
    for score, n in ordered:
        n = re.sub(r"\s+", " ", n).strip()
        if not _is_plausible_player(n, club):
            continue
        key = re.sub(r"[^a-z]+", "", n.lower())
        if key in seen or len(key) < 7:
            continue
        # If we already have enough high-confidence names, skip low-score junk
        if score < 4 and len(out) >= max(4, limit // 3):
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= limit:
            break
    return out


def _make_record(name: str, league: str) -> Dict:
    return {
        "id": _team_id(name, league),
        "normalized_name": name,
        "abbreviation": _abbrev(name),
        "city": None,
        "mascot": None,
        "nickname": _nickname(name),
        "league": league,
        "sport": "Soccer",
        "logo_url": None,
        "players": [],
        "player_count": 0,
    }


def _wiki_get_json(url: str) -> Optional[dict]:
    import urllib.request

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TinySLM/1.0 (local research agent; educational)"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _wiki_resolve_title(club: str) -> Optional[str]:
    """Resolve a club name to an English Wikipedia title via opensearch."""
    import urllib.parse

    queries = [
        f"{club} Saudi football club",
        f"{club} FC Saudi Arabia",
        f"{club} SFC",
        club,
    ]
    for q in queries:
        url = (
            "https://en.wikipedia.org/w/api.php?action=opensearch&limit=5&namespace=0&format=json&search="
            + urllib.parse.quote(q)
        )
        data = _wiki_get_json(url)
        if not data or len(data) < 2:
            continue
        titles = data[1] or []
        club_l = club.lower().replace("-", " ")
        stem = re.sub(r"^al[\s\-]+", "", club_l)
        for t in titles:
            tl = t.lower()
            if "disambiguation" in tl:
                continue
            if stem and stem in tl.replace("-", " ").replace("_", " "):
                if any(
                    k in tl
                    for k in (
                        "saudi",
                        "riyadh",
                        "jeddah",
                        "dammam",
                        "fc",
                        "sfc",
                        "club",
                        "football",
                    )
                ) or stem == tl.replace("-", " ").replace("_", " "):
                    return t
        for t in titles:
            tl = t.lower().replace("_", " ").replace("-", " ")
            if "disambiguation" in tl:
                continue
            if stem and stem in tl:
                return t
    return None


def _wiki_clean_player_name(raw: str) -> str:
    n = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", raw or "")
    n = re.sub(r"\{\{[^}]+\}\}", "", n)
    n = re.sub(r"<[^>]+>", "", n)
    n = re.sub(r"'{2,}", "", n)
    n = re.sub(r"\s+", " ", n).strip(" |")
    return n


def _wiki_fetch_club_details(club: str, max_players: int = 22) -> Dict:
    """Primary enrichment source: Wikipedia pageimages + squad wikitext."""
    import urllib.parse

    out: Dict = {"city": None, "logo_url": None, "players": [], "nickname": None}
    title = _wiki_resolve_title(club)
    if not title:
        return out

    qtitle = urllib.parse.quote(title.replace(" ", "_"))
    meta = _wiki_get_json(
        "https://en.wikipedia.org/w/api.php?action=query&redirects=1&format=json"
        f"&prop=pageimages|extracts&exintro=1&explaintext=1&pithumbsize=500&titles={qtitle}"
    )
    if meta:
        pages = (meta.get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), {}) if pages else {}
        thumb = ((page.get("thumbnail") or {}).get("source")) if page else None
        if thumb:
            out["logo_url"] = thumb
        extract = page.get("extract") or ""
        city = _extract_city(extract, club=club)
        if city:
            out["city"] = city

    parsed = _wiki_get_json(
        "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&redirects=1"
        f"&page={qtitle}"
    )
    if not parsed or "parse" not in parsed:
        return out
    wt = parsed["parse"]["wikitext"]["*"]

    nick_m = re.search(r"\|\s*nickname\s*=\s*([^\n{]+)", wt, flags=re.I)
    if nick_m:
        nick = _wiki_clean_player_name(nick_m.group(1))
        if nick and len(nick) < 40:
            out["nickname"] = nick

    for key in ("location", "ground", "stadium"):
        gm = re.search(rf"\|\s*{key}\s*=\s*([^\n]+)", wt, flags=re.I)
        if gm and not out["city"]:
            city = _extract_city(_wiki_clean_player_name(gm.group(1)), club=club)
            if city:
                out["city"] = city

    raw_names = re.findall(r"\{\{fs player[^}]*\|name=([^|}]+)", wt, flags=re.I)
    if not raw_names:
        raw_names = re.findall(
            r"\{\{football squad player[^}]*\|name=([^|}]+)", wt, flags=re.I
        )
    players: List[str] = []
    seen = set()
    for raw in raw_names:
        pname = _wiki_clean_player_name(raw)
        if len(pname.split()) < 2 or len(pname) > 48:
            continue
        # Drop footnote markers etc.
        pname = re.sub(r"\[.*?\]", "", pname).strip()
        if not pname:
            continue
        key = re.sub(r"[^a-z]+", "", pname.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        players.append(pname)
        if len(players) >= max_players:
            break
    out["players"] = players
    return out


def enrich_team(team: Dict, max_players: int = 22) -> Tuple[Dict, str]:
    """SLM research tools: Wikipedia primary + DDG fallback for city/logo/players."""
    name = team.get("normalized_name") or ""
    notes: List[str] = []
    team["players"] = []
    team["player_count"] = 0
    team["logo_url"] = None

    wiki = _wiki_fetch_club_details(name, max_players=max_players)
    notes.append(
        f"[tool:wikipedia] {name} → city={wiki.get('city')!r} "
        f"logo={'yes' if wiki.get('logo_url') else 'no'} "
        f"players={len(wiki.get('players') or [])}"
    )
    if wiki.get("city"):
        team["city"] = wiki["city"]
    if wiki.get("logo_url"):
        team["logo_url"] = wiki["logo_url"]
    if wiki.get("nickname"):
        team["nickname"] = wiki["nickname"]

    player_names = list(wiki.get("players") or [])

    # DDG fallback / supplement when wiki is thin
    if not team.get("city") or not team.get("logo_url") or len(player_names) < 5:
        q_city = f"{name} Saudi football club city based in"
        hits_city = search_web_hits(q_city, max_results=5)
        notes.append(f"[tool:search] {q_city} → {len(hits_city)} hits")
        blob_city = "\n".join(f"{h.title}\n{h.body}\n{h.href}" for h in hits_city)
        if not team.get("city"):
            city = _extract_city(blob_city, club=name)
            if city:
                team["city"] = city
        if not team.get("logo_url"):
            logo = _extract_logo_url(hits_city, club=name)
            if logo:
                team["logo_url"] = logo

        q_squad = f"{name} Transfermarkt squad roster players"
        hits_squad = search_web_hits(q_squad, max_results=8)
        notes.append(f"[tool:search] {q_squad} → {len(hits_squad)} hits")
        blob_squad = "\n".join(f"{h.title}\n{h.body}" for h in hits_squad)
        extra = _extract_players(blob_squad, club=name, limit=max_players)
        for pname in extra:
            if pname not in player_names:
                player_names.append(pname)
            if len(player_names) >= max_players:
                break

    players = []
    for pname in player_names[:max_players]:
        players.append(
            {
                "id": _player_id(team["id"], pname),
                "normalized_name": pname,
                "position": None,
                "nationality": None,
                "number": None,
            }
        )
    team["players"] = players
    team["player_count"] = len(players)
    notes.append(
        f"[tool:extract] {name}: city={team.get('city')!r} "
        f"logo={'yes' if team.get('logo_url') else 'no'} players={len(players)}"
    )
    return team, "\n".join(notes)


def enrich_teams(teams: List[Dict], max_players: int = 22) -> Tuple[List[Dict], str]:
    notes: List[str] = []
    out: List[Dict] = []
    for t in teams:
        enriched, n = enrich_team(t, max_players=max_players)
        out.append(enriched)
        notes.append(n)
    notes.append(f"[tool:reason] enriched {len(out)} teams with city/logo/players")
    return out, "\n".join(notes)


def collect_saudi_league_teams(
    max_per_league: int = 24,
    enrich: bool = True,
    max_players: int = 22,
) -> Tuple[List[Dict], str]:
    """Agent search loop: query each Saudi league, extract clubs, optionally enrich."""
    notes: List[str] = []
    teams: List[Dict] = []
    seen = set()

    for league_label, query in _SAUDI_LEAGUE_QUERIES:
        hits = search_web_hits(query, max_results=8)
        notes.append(f"[tool:search] {query} → {len(hits)} hits")
        blob = "\n".join(f"{h.title}\n{h.body}" for h in hits)
        names = _candidate_names(blob)
        if len(names) < 4:
            alt = search_web_hits(query.replace(" list", ""), max_results=6)
            blob2 = "\n".join(f"{h.title}\n{h.body}" for h in alt)
            names = _candidate_names(blob + "\n" + blob2)
        added = 0
        for name in names:
            key = re.sub(r"[^a-z0-9]+", "", name.lower())
            full = f"{key}|{league_label}"
            if full in seen:
                continue
            seen.add(full)
            teams.append(_make_record(name, league_label))
            added += 1
            if added >= max_per_league:
                break
        notes.append(f"[tool:extract] {league_label}: {added} teams")

    if enrich and teams:
        teams, enrich_notes = enrich_teams(teams, max_players=max_players)
        notes.append(enrich_notes)

    notes.append(f"[tool:reason] assembled {len(teams)} team records as JSON")
    return teams, "\n".join(notes)


def format_teams_json(teams: List[Dict]) -> str:
    return json.dumps(teams, indent=4, ensure_ascii=False)


def answer_league_collect(user: str) -> Optional[str]:
    if not looks_league_collect(user):
        return None
    u = (user or "").lower()
    # Prefer enriching an existing saved file when user complains about empty players/city/logo
    existing_path = Path("saudi_league_teams.json")
    if existing_path.is_file() and any(
        w in u for w in ("player", "city", "logo", "empty", "enrich", "fill", "complete", "missing")
    ):
        try:
            teams = json.loads(existing_path.read_text(encoding="utf-8"))
        except Exception:
            teams = []
        if isinstance(teams, list) and teams:
            teams, _notes = enrich_teams(teams, max_players=18)
            return format_teams_json(teams)

    teams, _notes = collect_saudi_league_teams(enrich=True, max_players=16)
    if not teams:
        return None
    return format_teams_json(teams)
