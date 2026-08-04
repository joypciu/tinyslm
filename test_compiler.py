"""Unit tests for the Cognitive Compiler IR."""

from __future__ import annotations

from tiny_slm.compiler import (
    compile_query,
    parse_ir_tag,
    pipeline_stages,
    should_run_stage,
)


def test_math_mode() -> None:
    ir = compile_query("What is 2 + 2?")
    assert ir.mode == "math"
    assert "symbolic" in ir.verify
    assert pipeline_stages(ir) == ["math"]
    assert should_run_stage(ir, "math")
    assert not should_run_stage(ir, "swarm")


def test_recall_mode() -> None:
    ir = compile_query("Using memory, what is the launch code?")
    assert ir.mode == "recall"
    assert "memory" in ir.need
    assert should_run_stage(ir, "memory")
    assert not should_run_stage(ir, "swarm")


def test_code_mode() -> None:
    ir = compile_query("Write a Python function that adds two numbers.")
    assert ir.mode == "code"
    assert should_run_stage(ir, "code")


def test_plan_mode() -> None:
    ir = compile_query("Plan a short study session step by step.")
    assert ir.mode in ("plan", "swarm", "sara")
    assert should_run_stage(ir, "plan") or should_run_stage(ir, "sara")


def test_faq_mode() -> None:
    ir = compile_query("What is RAM?", auto_search=False)
    assert ir.mode == "faq"
    assert should_run_stage(ir, "faq")


def test_tag_roundtrip() -> None:
    ir = compile_query("Using memory, what was the secret code related to BLUE?")
    tag = ir.to_tag()
    assert "MODE=recall" in tag
    back = parse_ir_tag(tag)
    assert back is not None
    assert back.mode == ir.mode
    assert back.need == ir.need


def test_chat_skips_swarm() -> None:
    ir = compile_query("Hello, how are you?", auto_search=False)
    assert ir.mode in ("chat", "faq", "sara")
    assert not should_run_stage(ir, "swarm")


def test_compare_prefers_faq_card() -> None:
    ir = compile_query(
        "Compare RAM and SSD for a laptop buyer: speed, persistence, and when each matters.",
        auto_search=False,
    )
    assert ir.mode == "compare"
    assert should_run_stage(ir, "faq")
    from tiny_slm.knowledge import answer_from_faq

    faq = answer_from_faq(
        "Compare RAM and SSD for a laptop buyer: speed, persistence, and when each matters."
    )
    assert faq and "ram" in faq.lower() and "ssd" in faq.lower()


def test_end_to_end_ir_in_reply() -> None:
    from tiny_slm.chat import TinyChat

    chat = TinyChat(auto_search=False)
    chat.clear_memory()
    chat.reset()
    reply, _ = chat.generate_reply("What is 10 percent of 200?", temperature=0.1)
    assert "[ir]" in reply
    assert "MODE=math" in reply
    assert "20" in reply


def main() -> None:
    tests = [
        test_math_mode,
        test_recall_mode,
        test_code_mode,
        test_plan_mode,
        test_faq_mode,
        test_tag_roundtrip,
        test_chat_skips_swarm,
        test_compare_prefers_faq_card,
        test_end_to_end_ir_in_reply,
    ]
    fails = []
    for fn in tests:
        try:
            fn()
            print(f"  OK {fn.__name__}")
        except Exception as exc:
            fails.append(f"{fn.__name__}: {exc}")
            print(f"  FAIL {fn.__name__}: {exc}")
    if fails:
        raise SystemExit(1)
    print("test_compiler OK")


if __name__ == "__main__":
    main()
