"""Inference: chat + 2M-token memory + agentic tool loop."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from tiny_slm.agent import looks_agentic
from tiny_slm.knowledge import (
    answer_from_code_template,
    answer_from_faq,
    answer_from_plan_template,
    looks_like_echo,
    looks_low_quality,
    scrub_generation,
)
from tiny_slm.memory import LongContextMemory, answer_from_memory, looks_like_recall
from tiny_slm.model import TinySLM
from tiny_slm.sara import run_sara, select_skills, try_eval_math
from tiny_slm.search import (
    answer_from_search,
    clean_search_query,
    needs_search,
    search_web,
    should_prefer_web_answer,
)
from tiny_slm.tokenizer import TinyTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "tinyslm.pt"
DEFAULT_TOK = ROOT / "checkpoints" / "tokenizer.json"


def _clean_for_history(reply: str) -> str:
    if "[model]" in reply:
        reply = reply.split("[model]", 1)[-1]
    reply = re.sub(r"^\[web\].*", "", reply, flags=re.S).strip()
    reply = re.sub(r"^\[agent\].*?\[model\]", "", reply, flags=re.S).strip()
    reply = scrub_generation(reply)
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
        self._seed_skill_cards()

    def _seed_skill_cards(self) -> None:
        """Seed procedural skill cards (tiny; helps agentic retrieve)."""
        try:
            from tiny_slm.sara import SKILL_CARDS

            for sk in SKILL_CARDS:
                self.memory.add_text(sk["card"], source="skill")
        except Exception:
            pass

    def reset(self) -> None:
        self.history.clear()
        # Keep long memory across resets unless caller clears it
        # Use clear_memory() for a full wipe.

    def clear_memory(self) -> None:
        self.memory.clear()
        self._seed_skill_cards()

    def ingest(self, text: str, source: str = "doc") -> dict:
        added = self.memory.add_text(text, source=source)
        return {"added_tokens": added, **self.memory.stats()}

    def save_memory(self, path: Path | str) -> Path:
        path = Path(path)
        self.memory.save(path)
        return path

    def load_memory(self, path: Path | str) -> dict:
        n = self.memory.load(path)
        self._seed_skill_cards()
        return {"loaded_chunks": n, **self.memory.stats()}

    def _history_window(self) -> List[Tuple[str, str]]:
        """Use up to 3 recent turns when they are short; else last 2.

        Keeps the neural prompt denser without growing KV past block_size.
        """
        if not self.history:
            return []
        recent3 = self.history[-3:]
        approx = sum(min(len(u), 100) + min(len(a), 120) for u, a in recent3)
        if len(recent3) == 3 and approx <= 420:
            return recent3
        return self.history[-2:]

    def _build_prompt(
        self,
        user: str,
        tool_block: str = "",
        memory_block: str = "",
    ) -> str:
        parts = ["<bos>"]
        for u, a in self._history_window():
            parts.append(f"<user>{u[:100]}<eos><assistant>{a[:120]}<eos>")
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
        tok = self.tokenizer
        stop_ids = {tok.eos_id}
        for name in ("<user>", "<bos>", "<assistant>"):
            tid = tok.vocab.get(name)
            if tid is not None:
                stop_ids.add(int(tid))
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
            if tid in stop_ids:
                break
            out_ids.append(tid)
            generated.append(tid)
            # Bail if special markers leak as decoded text
            if len(out_ids) % 8 == 0:
                frag = tok.decode(out_ids, skip_special=False)
                if any(m in frag for m in ("<user>", "<bos>", "<assistant>", "<eos>")):
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
        temperature: float = 0.35,
        top_k: int = 28,
        force_search: bool = False,
        force_agent: bool = False,
        repetition_penalty: float = 1.18,
        use_sara: bool = True,
    ) -> Tuple[str, Optional[str]]:
        search_digest: Optional[str] = None
        memory_block = ""
        if self.memory.chunks:
            memory_block = self.memory.retrieve(user, top_k=4, max_chars=800)

        # Extractive memory reply (no training) — tiny nets rarely copy rare codes
        mem_direct = answer_from_memory(user, memory_block) if memory_block else None
        if mem_direct:
            header = [
                f"[memory] extractive "
                f"({self.memory.stats()['tokens']:,}/{self.memory.max_tokens:,} tok store)"
            ]
            display = "\n".join(header) + f"\n\n[model]\n{mem_direct}"
            clean = _clean_for_history(mem_direct)
            self.history.append((user, clean))
            self.memory.add_turn(user, clean)
            return display, search_digest

        # Symbolic math before FAQ / neural (no training)
        math_ans = try_eval_math(user)
        if math_ans:
            self.history.append((user, math_ans[:280]))
            self.memory.add_turn(user, math_ans[:280])
            return math_ans, search_digest

        # Grounded FAQ cards (no training) — covers brittle short definitions
        faq = answer_from_faq(user)
        if faq:
            self.history.append((user, faq[:280]))
            self.memory.add_turn(user, faq[:280])
            return faq, search_digest

        plan = answer_from_plan_template(user)
        if plan:
            self.history.append((user, plan[:280]))
            self.memory.add_turn(user, plan[:280])
            return plan, search_digest

        code = answer_from_code_template(user)
        if code:
            self.history.append((user, code[:280]))
            self.memory.add_turn(user, code[:280])
            return code, search_digest

        # Auto web search for open knowledge / news before weak neural decode
        if force_search or (self.auto_search and needs_search(user)):
            search_digest = search_web(clean_search_query(user), max_results=4)
            web_ans = answer_from_search(search_digest, query=user)
            if web_ans and should_prefer_web_answer(user):
                display = f"[web]\n{search_digest}\n\n[model]\n{web_ans}"
                clean = _clean_for_history(web_ans)
                self.history.append((user, clean))
                self.memory.add_turn(user, clean)
                if search_digest and not search_digest.startswith("("):
                    self.memory.add_text(
                        f"WEB about {clean_search_query(user)}: {web_ans}",
                        source="web",
                    )
                return display, search_digest
            if search_digest and search_digest.startswith("("):
                # Search attempted but failed — avoid garbage neural fill
                msg = (
                    "I tried the web but got no usable results. "
                    "Try rephrasing, or ask a shorter factual question."
                )
                self.history.append((user, msg))
                self.memory.add_turn(user, msg)
                return f"[web]\n{search_digest}\n\n[model]\n{msg}", search_digest

        def _raw_generate(prompt_user: str) -> str:
            # Keep retrieved memory in the neural prompt for SARA drafts
            prompt = self._build_prompt(
                prompt_user, tool_block="", memory_block=memory_block[:500]
            )
            ids = self.tokenizer.encode(prompt)
            new_ids = self._generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            text = self.tokenizer.decode(new_ids, skip_special=True).strip()
            if "\n\n" in text:
                text = text.split("\n\n", 1)[0].strip()
            return text

        # SARA for agentic / math / memory recall / explicit force
        sara_gate = (
            force_agent
            or looks_agentic(user)
            or looks_like_recall(user)
            or try_eval_math(user) is not None
            or any(
                "plan_steps" in s or "compare_two" in s or "memory_answer" in s or "math_simple" in s
                for s in select_skills(user)
            )
        )
        if use_sara and sara_gate:
            sara = run_sara(
                user,
                generate_fn=_raw_generate,
                memory_retrieve=lambda q: self.memory.retrieve(q, top_k=4, max_chars=600),
                auto_search=self.auto_search or force_search,
                force_agent=force_agent or looks_agentic(user) or looks_like_recall(user),
            )
            reply = scrub_generation(sara.final or "(empty reply)")
            if looks_like_echo(user, reply) or looks_low_quality(reply):
                faq_fallback = answer_from_faq(user)
                plan_fallback = answer_from_plan_template(user)
                code_fallback = answer_from_code_template(user)
                web_fallback = None
                if self.auto_search and not search_digest:
                    search_digest = search_web(clean_search_query(user), max_results=4)
                    web_fallback = answer_from_search(search_digest, query=user)
                reply = (
                    faq_fallback
                    or plan_fallback
                    or code_fallback
                    or web_fallback
                    or (reply if not looks_low_quality(reply) else "")
                    or "I'm not sure I followed that — try a shorter question?"
                )
            header = [
                f"[sara] skills={len(sara.skills)} revised={sara.revised} reflect={sara.reflection}"
            ]
            if sara.agent is not None:
                header.append(
                    f"[agent] plan={' - '.join(sara.agent.plan)} steps={sara.agent.steps_done}"
                )
            if memory_block:
                header.append(
                    f"[memory] {self.memory.stats()['tokens']:,}/{self.memory.max_tokens:,} tok store"
                )
            if sara.verified_math:
                header.append("[verify] symbolic-math")
            display = "\n".join(header) + f"\n\n[model]\n{reply}"
            clean = _clean_for_history(reply)
            self.history.append((user, clean))
            self.memory.add_turn(user, clean)
            return display, search_digest

        # Simple chat path (search already attempted above when needed)
        tool_block = ""
        if search_digest:
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
        reply = scrub_generation(
            self.tokenizer.decode(new_ids, skip_special=True).strip()
        )
        if "\n\n" in reply:
            reply = reply.split("\n\n", 1)[0].strip()
        # Prefer first complete sentence for short chit-chat
        if reply and len(reply) > 160:
            m = re.match(r"^(.+?[.!?])(\s|$)", reply, re.S)
            if m and len(m.group(1)) >= 20:
                reply = m.group(1).strip()
        if not reply or looks_like_echo(user, reply) or looks_low_quality(reply):
            faq_fallback = answer_from_faq(user)
            plan_fallback = answer_from_plan_template(user)
            code_fallback = answer_from_code_template(user)
            if not search_digest and self.auto_search:
                search_digest = search_web(clean_search_query(user), max_results=4)
            web_fallback = (
                answer_from_search(search_digest, query=user) if search_digest else None
            )
            reply = (
                faq_fallback
                or plan_fallback
                or code_fallback
                or web_fallback
                or (reply if reply and not looks_low_quality(reply) else None)
                or "I'm not sure I followed that — try a shorter question?"
            )

        display = reply
        header_parts = []
        if memory_block:
            header_parts.append(
                f"[memory] used {len(memory_block)} chars "
                f"({self.memory.stats()['tokens']:,}/{self.memory.max_tokens:,} tok store)"
            )
        if search_digest:
            header_parts.append(f"[web]\n{search_digest}")
        if header_parts:
            display = "\n".join(header_parts) + f"\n\n[model]\n{reply}"

        clean = _clean_for_history(reply)
        self.history.append((user, clean))
        self.memory.add_turn(user, clean)
        return display, search_digest
