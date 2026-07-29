"""Byte-Pair Encoding tokenizer trained from scratch (no pretrained vocab)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<user>",
    "<assistant>",
    "<search>",
    "</search>",
    "<memory>",
    "</memory>",
    "<agent>",
    "</agent>",
]



class TinyTokenizer:
    def __init__(
        self,
        merges: Optional[List[tuple]] = None,
        vocab: Optional[dict] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = special_tokens or list(SPECIAL_TOKENS)
        self.merges: List[tuple] = [tuple(m) for m in (merges or [])]
        self.vocab: dict = dict(vocab) if vocab else {}
        self.id_to_token: dict = {}
        if self.vocab:
            self._rebuild_reverse()

    def _rebuild_reverse(self) -> None:
        self.id_to_token = {int(i): t for t, i in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def _get_stats(self, words: List[List[str]]) -> Counter:
        pairs: Counter = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_pair(self, words: List[List[str]], pair: tuple) -> List[List[str]]:
        a, b = pair
        merged = a + b
        out: List[List[str]] = []
        for word in words:
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            out.append(new_word)
        return out

    def train(self, texts: Iterable[str], vocab_size: int = 4096, max_chars: int = 80_000) -> None:
        # Start from characters (UTF-8 safe): each unique char is a base token
        corpus = "\n".join(texts)
        if len(corpus) > max_chars:
            # Sample evenly for faster BPE (full corpus still used later for LM training)
            step = max(1, len(corpus) // max_chars)
            corpus = corpus[::step][:max_chars]

        chars = sorted(set(corpus))
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # Word-level split then character sequences for BPE
        raw_words = re.findall(r"\S+|\n", corpus)
        # Cap number of word pieces processed per merge pass
        if len(raw_words) > 12_000:
            raw_words = raw_words[:12_000]
        words = [list(w) for w in raw_words]

        self.merges = []
        target = max(vocab_size, len(self.vocab))
        while len(self.vocab) < target:
            stats = self._get_stats(words)
            if not stats:
                break
            pair = stats.most_common(1)[0][0]
            if stats[pair] < 2:
                break
            words = self._merge_pair(words, pair)
            merged = pair[0] + pair[1]
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
            self.merges.append(pair)

        self._rebuild_reverse()

    def _bpe(self, token: str) -> List[str]:
        if not token:
            return []
        if token in self.vocab:
            return [token]
        word = list(token)
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            # Apply lowest-rank (earliest) merge present
            candidate = None
            best = None
            for p in pairs:
                r = merge_rank.get(p)
                if r is not None and (best is None or r < best):
                    best = r
                    candidate = p
            if candidate is None:
                break
            a, b = candidate
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        # Preserve special tokens as atomic units
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        parts = re.split(pattern, text)
        for part in parts:
            if not part:
                continue
            if part in self.vocab and part in self.special_tokens:
                ids.append(self.vocab[part])
                continue
            for piece in re.findall(r"\S+|\n|\s", part):
                if piece in self.vocab and len(piece) > 1 and piece not in self.special_tokens:
                    # Prefer whole-token if present (rare)
                    ids.append(self.vocab[piece])
                    continue
                for sym in self._bpe(piece):
                    ids.append(self.vocab.get(sym, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        toks = []
        special = set(self.special_tokens)
        for i in ids:
            tok = self.id_to_token.get(int(i), "<unk>")
            if skip_special and tok in special:
                continue
            toks.append(tok)
        return "".join(toks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            merges=[tuple(m) for m in data["merges"]],
            vocab=data["vocab"],
            special_tokens=data["special_tokens"],
        )
