"""Inference: chat + 2M-token memory + agentic tool loop."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from tiny_slm.agent import looks_agentic, run_agent_tools
from tiny_slm.memory import LongContextMemory
from tiny_slm.model import TinySLM
from tiny_slm.search import clean_search_query, needs_search, search_web
from tiny_slm.tokenizer import TinyTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "tinyslm.pt"
DEFAULT_TOK = ROOT / "checkpoints" / "tokenizer.json"


def _clean_for_history(reply: str) -> str:
    if "[model]" in reply:
        reply = reply.split("[model]", 1)[-1]
    reply = re.sub(r"^\[web\].*", "", reply, flags=re.S).strip()
    reply = re.sub(r"^\[agent\].*?\[model\]", "", reply, flags=re.S).strip()
    return reply[:280].strip()


class TinyChat:
    def __init__(
        self,
        ckpt_path: Path = DEFAULT_CKPT,
        tok_path: Path = DEFAULT_TOK,
        device: str = "cpu",
        auto_search: bool = True,
        memory_tokens: int = 2_000_000,
    ):
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {ckpt_path}. Train first: python train.py"
            )
        if not tok_path.exists():
            raise FileNotFoundError(f"Missing tokenizer {tok_path}")

        self.device = torch.device(device)
        self.tokenizer = TinyTokenizer.load(tok_path)
        self.model, self.step = TinySLM.load_checkpoint(str(ckpt_path), map_location=str(self.device))
        self.model.to(self.device)
        self.model.eval()
        self.auto_search = auto_search
        self.history: List[Tuple[str, str]] = []
        self.memory = LongContextMemory(max_tokens=memory_tokens)

    def reset(self) -> None:
        self.history.clear()
        # Keep long memory across resets unless caller clears it
        # Use clear_memory() for a full wipe.

    def clear_memory(self) -> None:
        self.memory.clear()

    def ingest(self, text: str, source: str = "doc") -> dict:
        added = self.memory.add_text(text, source=source)
        return {"added_tokens": added, **self.memory.stats()}

    def _build_prompt(
        self,
        user: str,
        tool_block: str = "",
        memory_block: str = "",
    ) -> str:
        parts = ["<bos>"]
        for u, a in self.history[-2:]:
            parts.append(f"<user>{u[:120]}<eos><assistant>{a[:160]}<eos>")
        ctx_bits = []
        if memory_block:
            ctx_bits.append(f"[memory]\n{memory_block[:700]}\n")
        if tool_block:
            ctx_bits.append(f"[agent]\n{tool_block[:700]}\n")
        prefix = "".join(ctx_bits)
        if prefix:
            parts.append(f"<user>{prefix}\nQuestion: {user}<eos><assistant>")
        else:
            parts.append(f"<user>{user}<eos><assistant>")
        return "".join(parts)

    @torch.inference_mode()
    def _generate(
        self,
        ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> list[int]:
        model = self.model
        block = model.config.block_size
        eos = self.tokenizer.eos_id
        device = self.device

        if len(ids) >= block:
            ids = ids[-(block - 1) :]

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _, past = model(idx, use_cache=True)
        out_ids: list[int] = []
        generated = list(ids)

        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :].clone()
            if repetition_penalty != 1.0 and generated:
                recent = set(generated[-64:])
                for tid in recent:
                    if logits_last[0, tid] > 0:
                        logits_last[0, tid] /= repetition_penalty
                    else:
                        logits_last[0, tid] *= repetition_penalty

            logits_last = logits_last / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            tid = int(next_id.item())
            out_ids.append(tid)
            generated.append(tid)
            if tid == eos:
                break
            if past is not None and past[0][0].size(2) >= block:
                past = None
                ctx = torch.tensor([generated[-block:]], dtype=torch.long, device=device)
                logits, _, past = model(ctx, use_cache=True)
            else:
                logits, _, past = model(next_id, past_kvs=past, use_cache=True)
        return out_ids

    def generate_reply(
        self,
        user: str,
        max_new_tokens: int = 96,
        temperature: float = 0.42,
        top_k: int = 28,
        force_search: bool = False,
        force_agent: bool = False,
        repetition_penalty: float = 1.18,
    ) -> Tuple[str, Optional[str]]:
        tool_block = ""
        memory_block = ""
        search_digest: Optional[str] = None
        agent_meta = None

        # Retrieve only when memory has content
        if self.memory.chunks:
            memory_block = self.memory.retrieve(user, top_k=4, max_chars=800)

        if force_agent or looks_agentic(user):
            tool_block, agent_meta = run_agent_tools(
                user,
                memory_retrieve=lambda q: self.memory.retrieve(q, top_k=4, max_chars=600),
                auto_search=self.auto_search or force_search,
            )
            # Tiny models synthesize better from a short "notes → answer" prompt
            synth = (
                f"Notes:\n{tool_block[:700]}\n"
                f"Answer the user clearly in short steps if needed.\n"
                f"User ask: {user}"
            )
            prompt = self._build_prompt(synth, tool_block="", memory_block=memory_block)
        else:
            if force_search or (self.auto_search and needs_search(user)):
                search_digest = search_web(clean_search_query(user), max_results=3)
                tool_block = f"[tool:search] {search_digest[:600]}"
            prompt = self._build_prompt(user, tool_block=tool_block, memory_block=memory_block)
        ids = self.tokenizer.encode(prompt)
        new_ids = self._generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        reply = self.tokenizer.decode(new_ids, skip_special=True).strip()
        if "\n\n" in reply:
            reply = reply.split("\n\n", 1)[0].strip()
        if not reply:
            reply = "(empty reply — train longer for better chat)"

        display = reply
        header_parts = []
        if agent_meta is not None:
            header_parts.append(
                f"[agent] plan={' - '.join(agent_meta.plan)} steps={agent_meta.steps_done}"
            )
        if memory_block:
            header_parts.append(f"[memory] used {len(memory_block)} chars "
                                f"({self.memory.stats()['tokens']:,}/{self.memory.max_tokens:,} tok store)")
        if search_digest:
            header_parts.append(f"[web]\n{search_digest}")
        if header_parts:
            display = "\n".join(header_parts) + f"\n\n[model]\n{reply}"

        clean = _clean_for_history(reply)
        self.history.append((user, clean))
        self.memory.add_turn(user, clean)
        return display, search_digest
