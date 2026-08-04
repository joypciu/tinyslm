"""Inference: chat + 2M-token memory + agentic tool loop."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from tiny_slm.agent import looks_agentic
from tiny_slm.code_verify import abstain_code_message, require_verified_code
from tiny_slm.compiler import CognitiveIR, should_run_stage
from tiny_slm.knowledge import (
    answer_from_code_template,
    answer_from_faq,
    answer_from_plan_template,
    looks_like_echo,
    looks_low_quality,
    looks_off_topic_math,
    looks_wrong_coding_answer,
    looks_wrong_sort_answer,
    repair_coding_answer,
    repair_plan_answer,
    repair_short_definition,
    repair_truncated_greeting,
    scrub_generation,
)
from tiny_slm.long_task import looks_long_task, run_long_task
from tiny_slm.memory import LongContextMemory, answer_from_memory, looks_like_recall
from tiny_slm.model import TinySLM
from tiny_slm.policy import ABSTAIN_GENERIC, decide_route
from tiny_slm.sara import run_sara, select_skills, try_eval_math
from tiny_slm.search import (
    answer_from_search,
    clean_search_query,
    needs_search,
    search_web,
    should_prefer_web_answer,
)
from tiny_slm.swarm import run_swarm, should_spawn_swarm
from tiny_slm.tokenizer import TinyTokenizer
from tiny_slm.traces import TraceStore
from tiny_slm.tvs import evidence_quorum, run_tvs

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
        log_traces: bool = True,
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
        self._search_cache: dict[str, str] = {}
        self.traces = TraceStore(enabled=log_traces)
        self._seed_skill_cards()

    def _trace(
        self,
        user: str,
        answer: str,
        ir: CognitiveIR,
        source: str,
    ) -> None:
        try:
            self.traces.record(
                user,
                answer,
                mode=ir.mode,
                source=source,
                ir_tag=ir.to_tag(),
                verify=list(ir.verify),
            )
        except Exception:
            pass

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
        self._search_cache.clear()
        self._seed_skill_cards()

    def _cached_search(self, user: str, max_results: int = 4) -> str:
        key = clean_search_query(user).lower()
        if key in self._search_cache:
            return self._search_cache[key]
        digest = search_web(key, max_results=max_results)
        if digest and not digest.startswith("("):
            self._search_cache[key] = digest
            # Bound cache size
            if len(self._search_cache) > 32:
                self._search_cache.pop(next(iter(self._search_cache)))
        return digest

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
        """Use more recent turns when the neural block is wider.

        Keeps the neural prompt denser without growing KV past block_size.
        """
        if not self.history:
            return []
        block = int(getattr(self.model.config, "block_size", 256) or 256)
        if block >= 512:
            recent = self.history[-5:]
            approx = sum(min(len(u), 140) + min(len(a), 160) for u, a in recent)
            if len(recent) >= 4 and approx <= 900:
                return recent
            return self.history[-3:]
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
        block = int(getattr(self.model.config, "block_size", 256) or 256)
        u_cap, a_cap = (140, 160) if block >= 512 else (100, 120)
        mem_cap = 1200 if block >= 512 else 700
        tool_cap = 1000 if block >= 512 else 700
        for u, a in self._history_window():
            parts.append(f"<user>{u[:u_cap]}<eos><assistant>{a[:a_cap]}<eos>")
        ctx_bits = []
        if memory_block:
            ctx_bits.append(f"[memory]\n{memory_block[:mem_cap]}\n")
        if tool_block:
            ctx_bits.append(f"[agent]\n{tool_block[:tool_cap]}\n")
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
        use_grounded: bool = True,
    ) -> Tuple[str, Optional[str]]:
        search_digest: Optional[str] = None
        memory_block = ""
        if self.memory.chunks:
            memory_block = self.memory.retrieve(user, top_k=4, max_chars=800)
        mem_direct = (
            answer_from_memory(user, memory_block)
            if memory_block and looks_like_recall(user)
            else None
        )

        # Production policy: grounded | search | agent | abstain (fail closed)
        route = decide_route(
            user,
            auto_search=self.auto_search or force_search,
            force_search=force_search,
            force_agent=force_agent,
            has_memory_hit=bool(mem_direct),
        )
        ir: CognitiveIR = route.ir
        ir_tag = route.to_tag()

        def _abstain(msg: str, source: str = "abstain") -> Tuple[str, Optional[str]]:
            display = f"[ir] {ir_tag}\n\n[model]\n{msg}"
            clean = _clean_for_history(msg)
            self.history.append((user, clean))
            self.memory.add_turn(user, clean)
            self._trace(user, msg, ir, source)
            return display, search_digest

        if route.action == "abstain" and route.message:
            return _abstain(route.message)

        # Thought-Verify Scratchpad for math/code/search/agent (novel fail-closed loop)
        if use_grounded and (
            route.action in ("agent", "search")
            or ir.mode in ("math", "code", "search", "plan", "long_task", "swarm")
        ):
            use_net = route.action == "search" or ir.mode == "search" or force_search
            tvs = run_tvs(
                user,
                memory_retrieve=lambda q: self.memory.retrieve(q, top_k=4, max_chars=600),
                auto_search=bool(use_net and (self.auto_search or force_search)),
            )
            if tvs.ok and tvs.answer:
                header = [f"[ir] {ir_tag}", tvs.header()]
                display = "\n".join(header) + f"\n\n[model]\n{tvs.answer}"
                clean = _clean_for_history(tvs.answer)
                self.history.append((user, clean))
                self.memory.add_turn(user, clean)
                self._trace(user, tvs.answer, ir, f"tvs-{tvs.domain}")
                return display, search_digest
            if tvs.abstained and route.action == "search":
                return _abstain(tvs.answer, "tvs-search")

        if use_grounded:
            # Extractive memory reply (no training) — tiny nets rarely copy rare codes
            if mem_direct:
                header = [
                    f"[ir] {ir_tag}",
                    f"[memory] extractive "
                    f"({self.memory.stats()['tokens']:,}/{self.memory.max_tokens:,} tok store)",
                ]
                display = "\n".join(header) + f"\n\n[model]\n{mem_direct}"
                clean = _clean_for_history(mem_direct)
                self.history.append((user, clean))
                self.memory.add_turn(user, clean)
                self._trace(user, mem_direct, ir, "memory")
                return display, search_digest

            # Verified math only (SymPy / safe legacy) — never neural arithmetic
            if should_run_stage(ir, "math") or ir.mode == "math":
                math_ans = try_eval_math(user)
                if math_ans:
                    display = f"[ir] {ir_tag}\n\n[model]\n{math_ans}"
                    self.history.append((user, math_ans[:280]))
                    self.memory.add_turn(user, math_ans[:280])
                    self._trace(user, math_ans, ir, "math")
                    return display, search_digest

            # Grounded FAQ cards (no training) — covers brittle short definitions
            if should_run_stage(ir, "faq") or route.action == "grounded":
                faq = answer_from_faq(user)
                if faq:
                    display = f"[ir] {ir_tag}\n\n[model]\n{faq}"
                    self.history.append((user, faq[:280]))
                    self.memory.add_turn(user, faq[:280])
                    self._trace(user, faq, ir, "faq")
                    return display, search_digest

            if should_run_stage(ir, "plan") or route.action in ("grounded", "agent"):
                plan = answer_from_plan_template(user)
                if plan:
                    display = f"[ir] {ir_tag}\n\n[model]\n{plan}"
                    self.history.append((user, plan[:280]))
                    self.memory.add_turn(user, plan[:280])
                    self._trace(user, plan, ir, "plan")
                    return display, search_digest

            if should_run_stage(ir, "code") or route.action in ("grounded", "agent"):
                code = answer_from_code_template(user)
                if code:
                    verified = require_verified_code(user, code) or code
                    # Templates are trusted; still prefer syntax-clean blocks
                    display = f"[ir] {ir_tag}\n\n[model]\n{verified}"
                    self.history.append((user, verified[:280]))
                    self.memory.add_turn(user, verified[:280])
                    self._trace(user, verified, ir, "code")
                    return display, search_digest

            # Complex research/projects: parallel search+crawl+vector RAG swarm
            if (
                self.auto_search
                and should_run_stage(ir, "swarm")
                and should_spawn_swarm(user, has_card=False)
            ):
                try:
                    swarm = run_swarm(user, max_workers=4, max_subgoals=5, max_pages_per_agent=2)
                except Exception as exc:
                    swarm = None
                    swarm_err = f"(swarm failed: {type(exc).__name__})"
                else:
                    swarm_err = ""
                if swarm and swarm.answer and len(swarm.answer) > 60:
                    search_digest = swarm.digest
                    header = [
                        f"[ir] {ir_tag}",
                        f"[swarm] subgoals={len(swarm.subgoals)} workers={swarm.workers} "
                        f"pages={swarm.pages_crawled} chunks={swarm.chunks} backend={swarm.backend}",
                    ]
                    display = "\n".join(header) + f"\n\n[model]\n{swarm.answer}"
                    clean = _clean_for_history(swarm.answer[:500])
                    self.history.append((user, clean))
                    self.memory.add_turn(user, clean)
                    self.memory.add_text(
                        f"SWARM about {clean_search_query(user)[:120]}: {swarm.answer[:600]}",
                        source="swarm",
                    )
                    self._trace(user, swarm.answer, ir, "swarm")
                    return display, search_digest
                elif swarm_err:
                    # Fall through to existing search/neural paths
                    pass

            # Long multi-step jobs: one sub-goal at a time + memory scratchpad
            if should_run_stage(ir, "long_task") and looks_long_task(user):

                def _solve(goal: str, step: str) -> str:
                    # Prefer grounded cards for the whole goal; else short neural draft
                    card = answer_from_code_template(goal) or answer_from_plan_template(goal)
                    if card:
                        return card
                    prompt = self._build_prompt(
                        f"{goal}\nSubstep: {step}",
                        memory_block=memory_block[:500],
                    )
                    ids = self.tokenizer.encode(prompt)
                    new_ids = self._generate(
                        ids,
                        max_new_tokens=min(max_new_tokens, 120),
                        temperature=min(temperature, 0.2),
                        top_k=min(top_k or 20, 14),
                        repetition_penalty=repetition_penalty,
                    )
                    text = scrub_generation(
                        self.tokenizer.decode(new_ids, skip_special=True).strip()
                    )
                    if "\n\n" in text:
                        text = text.split("\n\n", 1)[0].strip()
                    return text

                final, lt = run_long_task(
                    user,
                    solve_step=_solve,
                    memory_add=lambda t: self.memory.add_text(t, source="long_task"),
                )
                final = scrub_generation(final) or final
                # Never ship gibberish for long coding asks — prefer cards / clear ask.
                if (
                    looks_low_quality(final)
                    or looks_wrong_coding_answer(user, final)
                    or not any(
                        t in final.lower()
                        for t in (
                            "def ",
                            "class ",
                            "import ",
                            "step 1",
                            "1)",
                            "tkinter",
                            "pygame",
                            "pyqt",
                        )
                    )
                ):
                    rescue = answer_from_code_template(user) or answer_from_plan_template(user)
                    if rescue:
                        final = rescue
                    else:
                        final = (
                            "I can help with a concrete coding goal. "
                            "Example: Write a tkinter desktop app with a window, "
                            "text field, and a button that shows a message."
                        )
                header = [
                    f"[ir] {ir_tag}",
                    f"[long-task] steps={len(lt.steps)} block={self.model.config.block_size}",
                ]
                if memory_block:
                    header.append(
                        f"[memory] {self.memory.stats()['tokens']:,}/{self.memory.max_tokens:,} tok store"
                    )
                display = "\n".join(header) + f"\n\n[model]\n{final}"
                clean = _clean_for_history(final)
                self.history.append((user, clean))
                self.memory.add_turn(user, clean)
                self._trace(user, final, ir, "long_task")
                return display, search_digest

        # Auto web search for open knowledge / news — never invent facts neurally
        want_search = force_search or route.action == "search" or (
            self.auto_search
            and route.action != "agent"
            and (should_run_stage(ir, "search") or needs_search(user))
        )
        if want_search:
            search_digest = self._cached_search(user, max_results=5)
            ok_q, web_ans, q_note = evidence_quorum(search_digest, user, min_hits=2)
            if not ok_q:
                # Soft fallback to classic extractive if quorum barely misses
                web_ans = answer_from_search(search_digest, query=user)
                q_note = "extractive-fallback" if web_ans else q_note
            if web_ans and (
                should_prefer_web_answer(user)
                or route.action == "search"
                or force_search
                or ok_q
            ):
                display = (
                    f"[ir] {ir_tag}\n[web] evidence={q_note}\n{search_digest}\n\n"
                    f"[model]\n{web_ans}"
                )
                clean = _clean_for_history(web_ans)
                self.history.append((user, clean))
                self.memory.add_turn(user, clean)
                if search_digest and not search_digest.startswith("("):
                    self.memory.add_text(
                        f"WEB about {clean_search_query(user)}: {web_ans}",
                        source="web",
                    )
                self._trace(user, web_ans, ir, "web")
                return display, search_digest
            if route.action == "search" or force_search:
                # Search was the intended path — abstain rather than hallucinate
                msg = (
                    "I tried the web but could not verify a reliable answer "
                    f"(evidence={q_note}). I will not guess. Please rephrase, "
                    "name a source, or ask a checkable math/code/plan question instead."
                )
                return _abstain(msg, "search-miss")

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

        # SARA / agent loop for multi-step work — grounded tools preferred inside
        sara_gate = (
            force_agent
            or route.action == "agent"
            or should_run_stage(ir, "sara")
            or looks_agentic(user)
        )
        if use_sara and sara_gate:
            sara = run_sara(
                user,
                generate_fn=_raw_generate,
                memory_retrieve=lambda q: self.memory.retrieve(q, top_k=4, max_chars=600),
                auto_search=self.auto_search or force_search,
                force_agent=force_agent or looks_agentic(user) or route.action == "agent",
                need=list(ir.need or []),
            )
            reply = scrub_generation(sara.final or "")
            # Coding asks must pass syntax verification
            code_ask = any(
                w in (user or "").lower()
                for w in ("python", "function", "class ", "write a", "implement", "tkinter")
            )
            if code_ask:
                verified = require_verified_code(user, reply)
                if verified:
                    reply = verified
                else:
                    rescue = answer_from_code_template(user) or answer_from_plan_template(user)
                    if rescue:
                        reply = rescue
                    else:
                        return _abstain(abstain_code_message(user), "code-unverified")
            if looks_like_echo(user, reply) or looks_low_quality(reply) or not reply:
                faq_fallback = answer_from_faq(user)
                plan_fallback = answer_from_plan_template(user)
                code_fallback = answer_from_code_template(user)
                web_fallback = None
                if self.auto_search and not search_digest and route.action in ("agent", "search"):
                    search_digest = self._cached_search(user, max_results=4)
                    web_fallback = answer_from_search(search_digest, query=user)
                reply = faq_fallback or plan_fallback or code_fallback or web_fallback or ""
                if not reply:
                    return _abstain(ABSTAIN_GENERIC, "agent-miss")
            header = [
                f"[ir] {ir_tag}",
                f"[sara] skills={len(sara.skills)} revised={sara.revised} reflect={sara.reflection}",
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
            self._trace(user, reply, ir, "sara")
            return display, search_digest

        # Neural decode ONLY when policy explicitly allows (e.g. short chitchat)
        if not route.allow_neural:
            return _abstain(route.message or ABSTAIN_GENERIC, "policy-block")

        tool_block = ""
        if search_digest:
            tool_block = f"[tool:search] {search_digest[:600]}"
        prompt = self._build_prompt(user, tool_block=tool_block, memory_block=memory_block)
        ids = self.tokenizer.encode(prompt)
        new_ids = self._generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=min(temperature, 0.35),
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        reply = scrub_generation(
            self.tokenizer.decode(new_ids, skip_special=True).strip()
        )
        if "\n\n" in reply:
            reply = reply.split("\n\n", 1)[0].strip()
        reply = repair_truncated_greeting(user, reply)
        if looks_like_echo(user, reply) or looks_low_quality(reply) or not reply:
            faq_fallback = answer_from_faq(user)
            if faq_fallback:
                reply = faq_fallback
            else:
                return _abstain(ABSTAIN_GENERIC, "neural-miss")

        display = f"[ir] {ir_tag}\n\n[model]\n{reply}"
        clean = _clean_for_history(reply)
        self.history.append((user, clean))
        self.memory.add_turn(user, clean)
        self._trace(user, reply, ir, "neural")
        return display, search_digest
