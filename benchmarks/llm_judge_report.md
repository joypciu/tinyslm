# TinySLM vs local GGUF LLMs (LLM-as-judge)

Method: **LLM-as-judge (local GGUF)**

| Model | Mean score (1-10) | Mean latency (s) | N |
|---|---:|---:|---:|
| TinySLM-SARA | 6.44 | 0.97 | 8 |
| Phi-4-mini-instruct-Q5_K_M | 6.09 | 12.66 | 8 |
| Qwen3.5-0.8B-Q6_K | 5.75 | 4.30 | 8 |

## Notes

TinySLM (~4M) + SARA + 2M memory vs local GGUFs in D:/models (Qwen3.5-0.8B ≈1B-class, Phi-4-mini ≈3.8B). No HuggingFace downloads.
