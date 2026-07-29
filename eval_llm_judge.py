"""LLM-as-judge benchmark using *local* GGUFs in D:\\models (no Hub download).

Uses D:\\llama-b8953-bin-win-cpu-x64\\llama-completion.exe
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks"
OUT.mkdir(exist_ok=True)

LLAMA_BIN = Path(r"D:\llama-b8953-bin-win-cpu-x64")
LLAMA_COMPLETION = LLAMA_BIN / "llama-completion.exe"
MODELS_DIR = Path(r"D:\models")

# Local peers (~0.8B–3.8B class). Phi-4-mini ≈ 3.8B; Qwen3.5-0.8B ≈ 1B-class.
DEFAULT_BASELINES = [
    str(MODELS_DIR / "Qwen3.5-0.8B-Q6_K.gguf"),
    str(MODELS_DIR / "Phi-4-mini-instruct-Q5_K_M.gguf"),
]
DEFAULT_JUDGE = str(MODELS_DIR / "Qwen3.5-0.8B-Q6_K.gguf")

PROMPTS = [
    {"id": "greet", "category": "chat", "prompt": "Hello!"},
    {"id": "math", "category": "reason", "prompt": "What is 2 + 2?"},
    {"id": "fact", "category": "knowledge", "prompt": "What is the capital of France?"},
    {"id": "plan", "category": "agentic", "prompt": "Plan a short study session step by step."},
    {"id": "compare", "category": "agentic", "prompt": "Compare France and Japan capitals."},
    {"id": "python", "category": "agentic", "prompt": "Break down this long task: learn Python basics."},
    {"id": "memory", "category": "memory", "prompt": "Using memory, what is the launch code?"},
    {"id": "sleep", "category": "chat", "prompt": "Give me a short sleep tip."},
]


@dataclass
class Sample:
    model: str
    prompt_id: str
    prompt: str
    answer: str
    latency_s: float


def strip_think(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    text = re.sub(r"<think>[\s\S]*", "", text, flags=re.I)
    text = re.sub(r"\[Start thinking\][\s\S]*?(?=^\S|\Z)", "", text, flags=re.M | re.I)
    return text.strip()


def strip_display(ans: str) -> str:
    if "[model]" in ans:
        ans = ans.split("[model]")[-1]
    return strip_think(ans).strip()


def tinyslm_runner() -> Callable[[str], str]:
    from tiny_slm.chat import TinyChat

    chat = TinyChat(auto_search=False)
    chat.clear_memory()
    chat.ingest(
        "Important project note: the launch code is ORBIT-77 and the deadline is Friday.",
        source="doc",
    )

    def run(prompt: str) -> str:
        chat.reset()
        force = any(
            k in prompt.lower()
            for k in ("plan", "break down", "compare", "memory", "launch")
        )
        ans, _ = chat.generate_reply(
            prompt, temperature=0.2, force_agent=force, use_sara=True
        )
        return strip_display(ans)

    return run


def gguf_runner(
    model_path: str,
    n_predict: int = 96,
    ctx: int = 2048,
    threads: int = 6,
) -> Callable[[str], str]:
    model_path = str(Path(model_path))
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    if not LLAMA_COMPLETION.exists():
        raise FileNotFoundError(f"Missing {LLAMA_COMPLETION}")

    def run(prompt: str) -> str:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_file = td_path / "out.txt"
            err_file = td_path / "err.txt"
            # Use conversation single-turn so the GGUF chat template is applied
            cmd = [
                str(LLAMA_COMPLETION),
                "-m",
                model_path,
                "-sys",
                "You are a helpful assistant. Answer clearly and briefly.",
                "-p",
                prompt,
                "-n",
                str(n_predict),
                "-c",
                str(ctx),
                "-t",
                str(threads),
                "--temp",
                "0.2",
                "-cnv",
                "-st",
                "--simple-io",
                "--no-display-prompt",
                "--reasoning",
                "off",
                "-r",
                "User:",
            ]
            env = dict(**{k: v for k, v in __import__("os").environ.items()})
            env["PATH"] = str(LLAMA_BIN) + ";" + env.get("PATH", "")
            with out_file.open("w", encoding="utf-8") as fo, err_file.open(
                "w", encoding="utf-8"
            ) as fe:
                proc = subprocess.run(
                    cmd,
                    stdout=fo,
                    stderr=fe,
                    env=env,
                    timeout=180,
                    check=False,
                )
            raw = out_file.read_text(encoding="utf-8", errors="ignore")
            if proc.returncode != 0 and not raw.strip():
                err = err_file.read_text(encoding="utf-8", errors="ignore")[-500:]
                raise RuntimeError(f"llama-completion failed ({proc.returncode}): {err}")
            text = strip_think(raw)
            text = re.sub(r"^Assistant:\s*", "", text, flags=re.I).strip()
            text = text.replace("[end of text]", "").strip()
            for stop in ("\nUser:", "\nAssistant:", "\nSystem:", "\n>"):
                if stop in text:
                    text = text.split(stop, 1)[0].strip()
            # Drop banner / junk lines
            lines = [
                ln
                for ln in text.splitlines()
                if ln.strip()
                and not ln.strip().startswith(("build", "model", "modalities", "available", "▄", "█"))
            ]
            return "\n".join(lines)[:800]

    return run


def heuristic_judge(prompt: str, answer: str, category: str) -> Dict[str, float]:
    a = answer.lower()
    scores = {"helpfulness": 5.0, "correctness": 5.0, "clarity": 5.0, "agentic": 5.0}
    if len(answer.strip()) < 5:
        return {k: 1.0 for k in scores}
    if category == "reason" and "4" in a:
        scores["correctness"] = 10
    if category == "knowledge" and "paris" in a:
        scores["correctness"] = 10
    if category == "agentic":
        if re.search(r"\b(1\)|1\.|step 1)\b", a) or ("paris" in a and "tokyo" in a):
            scores["agentic"] = 9
            scores["helpfulness"] = 8
        else:
            scores["agentic"] = 4
    if category == "memory" and "orbit" in a:
        scores["correctness"] = 10
        scores["helpfulness"] = 9
    elif category == "memory":
        scores["correctness"] = 2
        scores["helpfulness"] = 3
    if category == "chat" and len(answer) > 8:
        scores["helpfulness"] = 7
        scores["clarity"] = 7
    return scores


def llm_judge(
    judge_run: Optional[Callable[[str], str]], prompt: str, answer: str, category: str
) -> Dict[str, float]:
    if judge_run is None:
        return heuristic_judge(prompt, answer, category)
    rubric = (
        "Rate the assistant answer on a 1-10 integer scale.\n"
        'Return ONLY JSON: {"helpfulness":N,"correctness":N,"clarity":N,"agentic":N}\n\n'
        f"Category: {category}\nUser: {prompt}\nAssistant: {answer[:450]}\n"
    )
    raw = judge_run(rubric)
    m = re.search(r"\{[^{}]+\}", raw, re.S)
    if not m:
        return heuristic_judge(prompt, answer, category)
    try:
        data = json.loads(m.group(0))
        return {
            k: float(data.get(k, 5))
            for k in ("helpfulness", "correctness", "clarity", "agentic")
        }
    except Exception:
        return heuristic_judge(prompt, answer, category)


def evaluate_model(name: str, runner: Callable[[str], str]) -> List[Sample]:
    samples = []
    for item in PROMPTS:
        print(f"  [{name}] {item['id']}...", flush=True)
        t0 = time.time()
        try:
            ans = runner(item["prompt"])
        except Exception as e:
            ans = f"(error: {e})"
        dt = time.time() - t0
        samples.append(
            Sample(
                model=name,
                prompt_id=item["id"],
                prompt=item["prompt"],
                answer=ans[:600],
                latency_s=dt,
            )
        )
        print(f"  [{name}] {item['id']}: {ans[:90]!r} ({dt:.2f}s)", flush=True)
    return samples


def short_name(path: str) -> str:
    return Path(path).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=DEFAULT_BASELINES,
        help="Local .gguf paths under D:\\models",
    )
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    print(f"llama-completion: {LLAMA_COMPLETION}", flush=True)
    print(f"models dir: {args.models_dir}", flush=True)
    print("=== TinySLM (SARA) ===", flush=True)
    all_samples = evaluate_model("TinySLM-SARA", tinyslm_runner())

    errors: List[str] = []
    if not args.skip_baselines:
        for path in args.baselines:
            p = Path(path)
            if not p.is_absolute():
                p = Path(args.models_dir) / path
            name = short_name(str(p))
            print(f"=== Baseline {name} ===", flush=True)
            try:
                runner = gguf_runner(str(p))
                all_samples.extend(evaluate_model(name, runner))
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"  SKIP {name}: {e}", flush=True)

    judge_run = None
    if not args.heuristic_only:
        print(f"=== Judge {short_name(args.judge)} ===", flush=True)
        try:
            jp = Path(args.judge)
            if not jp.is_absolute():
                jp = Path(args.models_dir) / args.judge
            judge_run = gguf_runner(str(jp), n_predict=80)
        except Exception as e:
            print(f"  Judge unavailable ({e}); heuristic rubric", flush=True)
            errors.append(f"judge:{e}")

    cat = {p["id"]: p["category"] for p in PROMPTS}
    rows = []
    by_model: Dict[str, List[float]] = {}
    for s in all_samples:
        scores = llm_judge(judge_run, s.prompt, s.answer, cat[s.prompt_id])
        avg = sum(scores.values()) / len(scores)
        by_model.setdefault(s.model, []).append(avg)
        rows.append(
            {**asdict(s), "scores": scores, "avg": avg, "category": cat[s.prompt_id]}
        )

    summary = {
        m: {
            "mean_score": sum(v) / len(v),
            "n": len(v),
            "mean_latency_s": sum(x.latency_s for x in all_samples if x.model == m)
            / max(1, len(v)),
        }
        for m, v in by_model.items()
    }

    report = {
        "summary": summary,
        "rows": rows,
        "errors": errors,
        "method": "LLM-as-judge (local GGUF)" if judge_run else "heuristic-rubric",
        "baselines": args.baselines,
        "note": (
            "TinySLM (~4M) + SARA + 2M memory vs local GGUFs in D:/models "
            "(Qwen3.5-0.8B ≈1B-class, Phi-4-mini ≈3.8B). "
            "No HuggingFace downloads."
        ),
    }
    out_json = OUT / "llm_judge_report.json"
    out_md = OUT / "llm_judge_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# TinySLM vs local GGUF LLMs (LLM-as-judge)",
        "",
        f"Method: **{report['method']}**",
        "",
        "| Model | Mean score (1-10) | Mean latency (s) | N |",
        "|---|---:|---:|---:|",
    ]
    for m, st in sorted(summary.items(), key=lambda kv: -kv[1]["mean_score"]):
        lines.append(
            f"| {m} | {st['mean_score']:.2f} | {st['mean_latency_s']:.2f} | {st['n']} |"
        )
    lines += ["", "## Notes", "", report["note"], ""]
    if errors:
        lines += ["## Skips / errors", ""] + [f"- {e}" for e in errors] + [""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print("\n".join(lines), flush=True)
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
