"""End-to-end smoke for multi-agent web swarm (live network)."""

from __future__ import annotations

from tiny_slm.chat import TinyChat
from tiny_slm.knowledge import answer_from_faq
from tiny_slm.swarm import looks_complex_query, run_swarm


def main() -> None:
    fails = []

    # Cards still win for short FAQ
    faq = answer_from_faq("What is RAM?")
    if not faq or "memory" not in faq.lower():
        fails.append("faq_ram")

    # Router
    if looks_complex_query("What is RAM?"):
        fails.append("router_false_complex")
    q = (
        "Research architecture tradeoffs and investigate best practices "
        "to design a small Python microservice project from scratch"
    )
    if not looks_complex_query(q):
        fails.append("router_miss_complex")

    # Direct swarm
    swarm = run_swarm(q, max_workers=3, max_subgoals=3, max_pages_per_agent=2, use_cache=False)
    print("swarm_digest:", swarm.digest.replace("\n", " | "))
    print("swarm_answer_head:", swarm.answer[:400].replace("\n", " "))
    if len(swarm.subgoals) < 2:
        fails.append("few_subgoals")
    if swarm.chunks < 1:
        fails.append("no_chunks")
    if "Research summary" not in swarm.answer and len(swarm.answer) < 80:
        fails.append("weak_answer")
    if swarm.workers < 2:
        fails.append("not_parallel")

    # Chat path
    chat = TinyChat(auto_search=True)
    chat.reset()
    chat.clear_memory()
    ans, digest = chat.generate_reply(q, temperature=0.2, max_new_tokens=80, force_agent=False)
    print("chat_head:", (ans or "")[:350].replace("\n", " | "))
    if "[swarm]" not in (ans or "") and "Research summary" not in (ans or ""):
        # May still pass via agent/SARA with swarm tool — accept swarm markers
        if "swarm" not in (ans or "").lower() and "Sources:" not in (ans or ""):
            fails.append("chat_no_swarm")

    # Short FAQ via chat still card
    chat.reset()
    chat.clear_memory()
    ans2, _ = chat.generate_reply("What is RAM?", temperature=0.2, max_new_tokens=40)
    if "memory" not in (ans2 or "").lower():
        fails.append("chat_faq")
    if "[swarm]" in (ans2 or ""):
        fails.append("faq_used_swarm")

    print("fails", fails)
    print("OVERALL", "PASS" if not fails else "FAIL")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
