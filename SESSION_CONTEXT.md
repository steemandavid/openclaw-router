# OpenClaw Router Session Context (2026-04-20)

## What was accomplished
1. **OpenClaw router is working** — Docker container with `network_mode: host`, all three tiers (simple/medium/complex) route correctly
2. **qwen3:30b** runs on Ollama at **~151 tok/s** baseline (18GB model, fits RTX 3090 24GB VRAM)
3. **Router fixes committed** — classifier reads `reasoning` field fallback, max_tokens=200 for classifier, `reasoning` stripped from responses, SIMPLE_MAX_TOKENS=2048
4. **Ollama updated** from v0.16.3 → v0.21.0 (server + client both v0.21.0)
5. **qwen3.6:35b benchmarked** — best case 32.8 tok/s (num_gpu=37) vs qwen3:30b's 150.8 tok/s. **Verdict: qwen3:30b wins by 4.6x**, stick with it.
6. **Cleaned up** qwen3-nothink:30b and qwen3.6:35b models (deleted)
7. **Model storage fixed** — qwen3 models copied from john's personal storage to /storage/ollama/models so systemd service can see them

## Current state
- Router running healthy on port 4100
- Ollama v0.21.0 running via systemd, listening on 0.0.0.0:11434
- Classifier: qwen3:30b @ Ollama
- Simple: qwen3:30b @ Ollama (~151 tok/s)
- Medium/Complex: claude-sonnet-4-6-20250514 @ z.ai
- All uncommitted changes committed

## Key files
- `.env` — model config, URLs set to 127.0.0.1, SIMPLE_MAX_TOKENS=2048
- `docker-compose.yml` — `network_mode: host`
- `router.py` — classifier reasoning fallback, max_tokens=200, strip reasoning from responses

## Hardware
- RTX 3090 24GB VRAM, 62GB RAM
- Ollama on 0.0.0.0:11434 (systemd service)

## Benchmark data (2026-04-20)
| Model | GPU Layers | tok/s | VRAM |
|-------|-----------|-------|------|
| qwen3:30b | all (GPU) | 150.8 | 21.1 GB |
| qwen3.6:35b | 30 | 21.4 | 19.5 GB |
| qwen3.6:35b | 35 | 28.5 | 22.4 GB |
| qwen3.6:35b | 36 | 32.0 | ~23 GB |
| qwen3.6:35b | 37 | 32.8 | ~23.5 GB |
| qwen3.6:35b | 38+ | OOM | — |

## Important notes
- **Only download Ollama models at night** to spare internet bandwidth
- qwen3:30b = MoE 30B total / 3B active per token
- The router container must use `network_mode: host` because Ollama only listens on 127.0.0.1
- The systemd Ollama service uses `/storage/ollama/models` (configured in override.conf)
- The nvfp4 variant (qwen3.6:35b-a3b-nvfp4, 22GB) could be tested in the future but would still likely be slower than qwen3:30b due to larger active parameters per token
