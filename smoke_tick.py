"""Fast regression smoke for improve-loop ticks (no weight updates)."""

from __future__ import annotations

from tiny_slm.agent import looks_agentic
from tiny_slm.chat import TinyChat
from tiny_slm.knowledge import answer_from_faq
from tiny_slm.sara import try_eval_math


def main() -> None:
    fails = []

    if looks_agentic("I have been reading about plants and sunlight. " * 6):
        fails.append("plants falsely agentic")
    if not looks_agentic("Plan a short study session step by step."):
        fails.append("plan not agentic")

    from tiny_slm.search import needs_search
    from tiny_slm.knowledge import answer_from_code_template

    if needs_search("Please keep answers warm and brief today."):
        fails.append("today falsely needs search")
    if not needs_search("search the web for python release news"):
        fails.append("explicit search not detected")
    if not needs_search("Explain gravity simply."):
        fails.append("explain should auto-search")
    code = answer_from_code_template("Write a Python function that adds two numbers.")
    if not code or "def add" not in code:
        fails.append("code template add")

    if try_eval_math("What is 10 percent of 200?") != "10% of 200 equals 20.":
        fails.append("percent math")
    if "4" not in (try_eval_math("What is 2 + 2?") or ""):
        fails.append("add math")
    if "8" not in (try_eval_math("2 to the power of 3") or ""):
        fails.append("power math")

    ram = answer_from_faq("What is RAM?") or ""
    if "memory" not in ram.lower():
        fails.append("ram faq")
    if "Ottawa" not in (answer_from_faq("What is the capital of Canada?") or ""):
        fails.append("canada capital")
    friend = answer_from_faq("Tell me a friendship tip.") or ""
    if "TinySLM" in friend and "Listen" not in friend:
        fails.append("hi matched friendship")
    from tiny_slm.sara import select_skills

    skills = " ".join(select_skills("Tell me about friendship and kindness."))
    if "friendly_chat" in skills.lower() and "listen" not in friend.lower():
        # skill card id text contains friendly_chat
        if "SKILL friendly_chat" in skills:
            fails.append("hi skill false positive")

    chat = TinyChat(auto_search=False)
    chat.clear_memory()
    chat.reset()
    chat.ingest(
        "Important project note: the launch code is ORBIT-77 and the deadline is Friday.",
        source="doc",
    )
    chat.ingest("Meeting notes about lunch. " * 50, source="noise")
    got = chat.memory.retrieve("Using memory, what is the launch code?", top_k=3)
    if "ORBIT-77" not in got or "SKILL" in got:
        fails.append(f"retrieve polluted or miss: {got[:120]!r}")
    reply, _ = chat.generate_reply(
        "Using memory, what is the launch code?", temperature=0.1, max_new_tokens=40
    )
    body = reply.split("[model]")[-1] if "[model]" in reply else reply
    if "orbit-77" not in body.lower():
        fails.append(f"orbit reply miss: {body[:80]!r}")

    chat2 = TinyChat(auto_search=False)
    chat2.clear_memory()
    chat2.reset()
    chat2.generate_reply(
        "Remember this secret project code: BLUE_LANTERN_CODE.",
        temperature=0.2,
        max_new_tokens=20,
        use_sara=False,
    )
    chat2.generate_reply("noise " * 40, temperature=0.2, max_new_tokens=20, use_sara=False)
    ans, _ = chat2.generate_reply(
        "Using memory, what was the secret code or token related to BLUE?",
        temperature=0.1,
    )
    body2 = ans.split("[model]")[-1] if "[model]" in ans else ans
    if "BLUE_LANTERN_CODE" not in body2:
        fails.append(f"blue needle miss: {body2[:80]!r}")

    # Import eval_ready readiness quickly
    import eval_ready

    er = eval_ready
    c = TinyChat(auto_search=False)
    b_ok, b_n, _ = er.section(c, er.BASIC, 0.2)
    s_ok, s_n, _ = er.section(c, er.SMART, 0.2)
    a_ok, a_n, _ = er.section(c, er.AGENTIC, 0.2, agent=True)
    mem_ok, _ = er.eval_memory_scale()
    long_ok, _ = er.eval_long_dialog_memory(c)
    ready = (
        b_ok / b_n >= 0.8
        and s_ok / s_n >= 0.67
        and a_ok / a_n >= 0.67
        and mem_ok
        and long_ok
    )
    print(
        f"smoke basic={b_ok}/{b_n} smart={s_ok}/{s_n} agentic={a_ok}/{a_n} "
        f"ready={ready} extra_fails={len(fails)}"
    )
    # Memory persistence round-trip
    tmp = __import__("pathlib").Path("checkpoints/_smoke_memory.json")
    chat.save_memory(tmp)
    chat3 = TinyChat(auto_search=False)
    info = chat3.load_memory(tmp)
    got3 = chat3.memory.retrieve("launch code ORBIT", top_k=2)
    tmp.unlink(missing_ok=True)
    if "ORBIT-77" not in got3 or info.get("loaded_chunks", 0) < 1:
        fails.append("memory save/load")

    for f in fails:
        print(f"  FAIL {f}")
    if fails or not ready:
        raise SystemExit(1)
    print("smoke_tick OK")


if __name__ == "__main__":
    main()
