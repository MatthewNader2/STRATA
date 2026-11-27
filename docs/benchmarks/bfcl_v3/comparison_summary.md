# BFCL v3 (N=120) — STRATA Cross-Model Comparison Summary

| Model / Architecture | Params (B) | Tasks | Strict Passed (95% CI) | Acceptable (95% CI) | Mean Wall (ms) | Total Tokens | Peak HWM (MB) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STRATA v2 Profile C (qwen3.5-0.8b-instruct) | 0.8 | 120 | 2/120 (1.7%, [0.5-5.9%]) | 42/120 (35.0%, [27.1-43.9%]) | 2741.0 | 152846 | 2397.0 | 1785.0 |
| STRATA v2 Profile O [No-RR] (qwen3.5-0.8b-instruct) | 0.8 | 120 | 16/120 (13.3%, [8.4-20.6%]) | 18/120 (15.0%, [9.7-22.5%]) | 5485.3 | 219933 | 4885.7 | 1791.0 |
| qwen3.5-0.8b-instruct [ReAct-SKILL.md-3T] | 0.8 | 120 | 30/120 (25.0%, [18.1-33.4%]) | 39/120 (32.5%, [24.8-41.3%]) | 1338.5 | 84874 | 1198.7 | 1265.0 |
| qwen3.5-0.8b-instruct [SKILL.md] | 0.8 | 120 | 30/120 (25.0%, [18.1-33.4%]) | 39/120 (32.5%, [24.8-41.3%]) | 1085.3 | 66061 | 1198.9 | 1265.0 |
| Qwen2.5-Coder-7B-Instruct-GGUF [ReAct-SKILL.md-3T] | 7.6 | 120 | 76/120 (63.3%, [54.4-71.4%]) | 104/120 (86.7%, [79.4-91.6%]) | 520.9 | 40371 | 5118.7 | 4911.0 |
| Qwen2.5-Coder-7B-Instruct-GGUF [SKILL.md] | 7.6 | 120 | 76/120 (63.3%, [54.4-71.4%]) | 104/120 (86.7%, [79.4-91.6%]) | 523.7 | 40371 | 5117.0 | 4911.0 |

## Per-Category Acceptable Rate Breakdown

| Model / Architecture | bfcl_irrelevance | bfcl_multiple | bfcl_simple |
|---|---:|---:|---:|
| STRATA v2 Profile C (qwen3.5-0.8b-instruct) | 40/40 (100.0%) | 1/40 (2.5%) | 1/40 (2.5%) |
| STRATA v2 Profile O [No-RR] (qwen3.5-0.8b-instruct) | 2/40 (5.0%) | 8/40 (20.0%) | 8/40 (20.0%) |
| qwen3.5-0.8b-instruct [ReAct-SKILL.md-3T] | 9/40 (22.5%) | 18/40 (45.0%) | 12/40 (30.0%) |
| qwen3.5-0.8b-instruct [SKILL.md] | 9/40 (22.5%) | 18/40 (45.0%) | 12/40 (30.0%) |
| Qwen2.5-Coder-7B-Instruct-GGUF [ReAct-SKILL.md-3T] | 28/40 (70.0%) | 39/40 (97.5%) | 37/40 (92.5%) |
| Qwen2.5-Coder-7B-Instruct-GGUF [SKILL.md] | 28/40 (70.0%) | 39/40 (97.5%) | 37/40 (92.5%) |
