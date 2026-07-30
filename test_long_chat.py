"""100-turn long-chat stress test for TinySLM context + memory.

One continuous conversation (100 exchanges). Plants unique facts early,
buries them under later turns, then probes recall. Reports how the neural
window (last 2 turns) and 2M LongContextMemory share the load.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from tiny_slm.chat import TinyChat
from tiny_slm.memory import approx_tokens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks"
OUT.mkdir(exist_ok=True)

# Unique needles planted at known turns (1-indexed)
NEEDLES = {
    1: ("BLUE_LANTERN_CODE", "Remember this secret project code: BLUE_LANTERN_CODE."),
    25: ("ORANGE_RIVER_42", "Note for later: the warehouse ID is ORANGE_RIVER_42."),
    50: ("SILVER_OWL_99", "Important: the meeting password is SILVER_OWL_99."),
    75: ("GREEN_COMET_7", "Store this token for the final check: GREEN_COMET_7."),
}

FILLER_TOPICS = [
    "I have been reading about plants and sunlight and how leaves make food.",
    "Today I practiced short math drills and reviewed my study notes carefully.",
    "Tell me something useful about sleep, focus, and keeping a calm routine.",
    "I want advice on writing a polite email and planning a small homework task.",
    "Can you explain simply how a CPU and RAM work inside a laptop computer?",
    "Let's talk about friendship, kindness, and learning with patience each day.",
    "I am curious about cities, capitals, travel tips, and clear short answers.",
    "Please keep answers warm and brief while we continue this long conversation.",
]


def make_long_user(turn: int, rng: random.Random) -> str:
    """Build a longer user utterance; inject needles on special turns."""
    if turn in NEEDLES:
        code, line = NEEDLES[turn]
        pad = " ".join(rng.choice(FILLER_TOPICS) for _ in range(3))
        return f"{pad} {line} Please acknowledge and continue chatting."
    pad = " ".join(rng.choice(FILLER_TOPICS) for _ in range(4))
    ask = rng.choice(
        [
            "What is one short tip for me?",
            "How should I plan my next small step?",
            "Give a friendly one-sentence reply.",
            "What is 2 + 2 as a quick check?",
            "Remind me to stay brief and clear.",
        ]
    )
    return f"Turn {turn}. {pad} {ask}"


def probe_prompt(code: str) -> str:
    return f"Using memory, what was the secret code or token related to {code.split('_')[0]}? Reply with the exact code if you know it."


def body(ans: str) -> str:
    if "[model]" in ans:
        ans = ans.split("[model]")[-1]
    return ans.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chat = TinyChat(auto_search=False)
    chat.clear_memory()
    chat.reset()

    block = chat.model.config.block_size
    print(
        f"Long-chat stress: {args.turns} turns | neural block_size={block} | "
        f"history_window=last 2 turns | memory_max={chat.memory.max_tokens:,}",
        flush=True,
    )

    latencies = []
    mem_curve = []
    samples = []
    t_all = time.time()

    for turn in range(1, args.turns + 1):
        user = make_long_user(turn, rng)
        t0 = time.time()
        ans, _ = chat.generate_reply(
            user,
            temperature=args.temperature,
            max_new_tokens=64,
            use_sara=True,
            force_agent=False,
        )
        dt = time.time() - t0
        latencies.append(dt)
        stats = chat.memory.stats()
        hist_n = len(chat.history)
        # Match TinyChat adaptive window (2-3 short turns)
        recent = chat._history_window()
        neural_tok_est = approx_tokens("".join(u + a for u, a in recent) + user)
        prompt_n = len(recent)

        mem_curve.append(
            {
                "turn": turn,
                "latency_s": round(dt, 3),
                "history_turns_stored": hist_n,
                "history_in_prompt": prompt_n,
                "memory_tokens": stats["tokens"],
                "memory_chunks": stats["chunks"],
                "memory_fill_pct": stats["fill_pct"],
                "neural_token_est": neural_tok_est,
                "neural_block_size": block,
                "compressed": neural_tok_est > block or hist_n > prompt_n,
            }
        )
        if turn in NEEDLES or turn % 10 == 0:
            print(
                f"turn {turn:3d}/{args.turns}  {dt:.2f}s  "
                f"mem={stats['tokens']:,} tok / {stats['chunks']} chunks  "
                f"hist={hist_n} (prompt uses last {prompt_n})  "
                f"neural~{neural_tok_est}/{block}",
                flush=True,
            )
        samples.append(
            {
                "turn": turn,
                "user": user[:180],
                "reply": body(ans)[:180],
                "latency_s": round(dt, 3),
            }
        )

    # Probe needles planted earlier (context compression / memory recall)
    print("\n=== Needle probes (after full 100-turn chat) ===", flush=True)
    probe_results = []
    for turn, (code, _) in NEEDLES.items():
        if turn > args.turns:
            continue
        q = probe_prompt(code)
        # Also direct retrieve from memory store
        retrieved = chat.memory.retrieve(code, top_k=3, max_chars=500)
        t0 = time.time()
        ans, _ = chat.generate_reply(q, temperature=0.1, force_agent=True, use_sara=True)
        dt = time.time() - t0
        reply = body(ans)
        hit_gen = code.lower() in reply.lower()
        hit_mem = code.lower() in retrieved.lower()
        probe_results.append(
            {
                "planted_turn": turn,
                "code": code,
                "memory_retrieve_hit": hit_mem,
                "model_reply_hit": hit_gen,
                "latency_s": round(dt, 3),
                "retrieved_preview": retrieved[:160],
                "reply_preview": reply[:160],
            }
        )
        print(
            f"  planted@{turn:3d} {code}: memory={'HIT' if hit_mem else 'MISS'}  "
            f"model={'HIT' if hit_gen else 'MISS'}  ({dt:.2f}s)",
            flush=True,
        )

    total = time.time() - t_all
    mem_hits = sum(1 for p in probe_results if p["memory_retrieve_hit"])
    gen_hits = sum(1 for p in probe_results if p["model_reply_hit"])
    n_probes = max(1, len(probe_results))

    # How compression works (documented from runtime behavior)
    mechanism = {
        "neural_window": (
            f"Only the last 2 chat turns enter the Transformer prompt; "
            f"block_size={block} truncates token ids from the left if needed."
        ),
        "long_memory": (
            "Every turn is appended to LongContextMemory (up to 2,000,000 tokens). "
            "Older dialogue is not kept in the neural KV cache; it is retrieved by BM25 "
            "when the user asks about past facts."
        ),
        "eviction": (
            "If memory exceeds max_tokens, oldest chunks are evicted and the index rebuilt."
        ),
    }

    summary = {
        "turns": args.turns,
        "total_seconds": round(total, 2),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3),
        "final_memory": chat.memory.stats(),
        "final_history_len": len(chat.history),
        "history_prompt_window": 2,
        "neural_block_size": block,
        "needle_memory_hits": f"{mem_hits}/{n_probes}",
        "needle_model_hits": f"{gen_hits}/{n_probes}",
        "managed_long_chat": mem_hits >= max(1, n_probes - 1),
        "context_compression": mechanism,
    }

    report = {
        "summary": summary,
        "memory_curve": mem_curve,
        "probes": probe_results,
        "sample_turns": samples[:: max(1, args.turns // 10)],
    }
    out_json = OUT / "long_chat_100_report.json"
    out_md = OUT / "long_chat_100_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# 100-turn long-chat stress test",
        "",
        f"- Turns: **{args.turns}** in one continuous session",
        f"- Total time: **{summary['total_seconds']}s** (avg **{summary['avg_latency_s']}s**/turn, p95 **{summary['p95_latency_s']}s**)",
        f"- Neural block_size: **{block}** | prompt history window: **last 2 turns**",
        f"- Final memory: **{summary['final_memory']['tokens']:,} tokens**, "
        f"**{summary['final_memory']['chunks']} chunks** "
        f"({summary['final_memory']['fill_pct']}% of 2M cap)",
        f"- Needle recall — memory retrieve: **{summary['needle_memory_hits']}**, "
        f"model reply: **{summary['needle_model_hits']}**",
        f"- Managed long chat: **{summary['managed_long_chat']}**",
        "",
        "## How context is compressed",
        "",
        f"1. {mechanism['neural_window']}",
        f"2. {mechanism['long_memory']}",
        f"3. {mechanism['eviction']}",
        "",
        "## Needle probes",
        "",
        "| Planted turn | Code | Memory HIT | Model HIT |",
        "|---:|---|:---:|:---:|",
    ]
    for p in probe_results:
        lines.append(
            f"| {p['planted_turn']} | `{p['code']}` | "
            f"{'yes' if p['memory_retrieve_hit'] else 'no'} | "
            f"{'yes' if p['model_reply_hit'] else 'no'} |"
        )
    lines += ["", f"Full JSON: `{out_json.name}`", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== SUMMARY ===", flush=True)
    print("\n".join(lines), flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
