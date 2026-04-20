# OpenClaw Router Session Context (2026-04-20)

## What was accomplished
1. **OpenClaw router is working** — Docker container with `network_mode: host`, all three tiers (simple/medium/complex) route correctly
2. **qwen3:30b** runs on Ollama at **153.7 tok/s** baseline (18GB model, fits RTX 3090 24GB VRAM)
3. **Router fixes applied** — classifier reads `reasoning` field fallback, max_tokens=200 for classifier, `reasoning` stripped from responses, SIMPLE_MAX_TOKENS=2048
4. **Ollama updated** from v0.16.3 → v0.21.0 (required for qwen3.6 qwen35moe architecture)

## What's pending
1. **Benchmark qwen3.6:35b with CPU offloading** vs qwen3:30b baseline (153.7 tok/s)
   - qwen3.6:35b is 22.3GB, won't fully fit 24GB VRAM → needs partial CPU offload
   - Use `num_gpu` option to control GPU/CPU split (40 total layers)
   - Run same benchmark prompt: "Explain what a recursive function is in 2-3 sentences." with `num_predict=800, temperature=0.3`
2. **Test nvfp4 variant** — `qwen3.6:35b-a3b-nvfp4` (22GB) should fit fully in RTX 3090
   - **DO NOT DOWNLOAD DURING DAYTIME** — pull at night only (user preference)
3. **Clean up** `qwen3-nothink:30b` model (debugging artifact, no longer needed)

## Key files modified
- `/home/john/projects/openclaw-router/.env` — model config, URLs set to 127.0.0.1, SIMPLE_MAX_TOKENS=2048
- `/home/john/projects/openclaw-router/docker-compose.yml` — `network_mode: host`
- `/home/john/projects/openclaw-router/router.py` — classifier reasoning fallback, max_tokens=200, strip reasoning from responses

## Current .env state
```
ZAI_API_KEY=7a5d9979d59b4bb8812d3fa7f59e1912.8R0etEtGlD0Dbuxn
ROUTER_API_KEY=
OLLAMA_BASE_URL=http://127.0.0.1:11434
ZAI_BASE_URL=https://api.z.ai/api/anthropic
CLASSIFIER_MODEL=qwen3:30b
CLASSIFIER_BASE_URL=http://127.0.0.1:11434
CLASSIFIER_TIMEOUT=15
SIMPLE_MODEL=qwen3:30b
MEDIUM_MODEL=claude-sonnet-4-6-20250514
COMPLEX_MODEL=claude-sonnet-4-6-20250514
SIMPLE_MAX_TOKENS=2048
UPSTREAM_TIMEOUT=180
PORT=4100
LOG_LEVEL=INFO
```

## Hardware
- RTX 3090 24GB VRAM, 62GB RAM
- Ollama on 127.0.0.1:11434 (localhost only)

## Important notes
- **Only download Ollama models at night** to spare internet bandwidth
- qwen3:30b = MoE 30B total / 3B active per token
- qwen3.6:35b = MoE+SSM hybrid, 35B total, 40 layers, 256 experts (8 active)
- The router container must use `network_mode: host` because Ollama only listens on 127.0.0.1

## Immediate next steps (pick up here)
1. Verify Ollama updated: `ollama --version` → should be v0.21.0
2. Load qwen3.6:35b with CPU offloading: try `num_gpu` values (20, 25, 30) and benchmark each
3. Compare tok/s with 153.7 baseline — expect significantly slower due to CPU layers
4. If too slow, schedule nvfp4 pull for nighttime (22GB, should fit GPU fully)
5. Once best qwen3.6 variant identified, update `.env` SIMPLE_MODEL and CLASSIFIER_MODEL
