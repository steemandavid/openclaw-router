# OpenClaw Router Session Context (2026-04-21)

## What was accomplished
1. **Streaming attribution footer working** — model info footer now appears in Discord responses from kreeft
2. **Tool calling enabled** — stopped stripping `tools`/`tool_choice` from Ollama requests; qwen3.6 supports native tool calling
3. **Fixed internal call detection** — only checks last 5 messages (was checking full history, causing false positives from old memory flush markers)
4. **Fixed empty thinking-phase chunks** — qwen3.6 thinking tokens no longer leak as empty content chunks in stream
5. **Classifier retry logic** — 3 attempts with escalating delays for Ollama connection drops
6. **OpenClaw container restart** — fixed Discord gateway connection issue (full stop/start needed, not just restart)
7. **All changes committed and pushed** to origin/main

## Current state
- Router running healthy on VM port 4100 (192.168.1.174)
- Ollama v0.21.0 running via systemd on host (192.168.1.165:11434)
- Classifier: qwen3.6:35b-a3b-q4_K_M @ Ollama
- Simple: qwen3.6:35b-a3b-q4_K_M @ Ollama (with OLLAMA_NUM_GPU=30 for CPU offloading)
- Medium/Complex: claude-sonnet-4-6-20250514 @ z.ai
- OpenClaw "kreeft" bot connected to Discord and responding with attribution footer

## Architecture
- **Host** (192.168.1.165): Ollama + host router (port 4100, for direct testing)
- **VM** (192.168.1.174): OpenClaw + VM router (port 4100, what kreeft uses)
- OpenClaw sends requests to `router:4100` (Docker network) → router classifies → routes to Ollama or z.ai
- Changes must be synced: edit local router.py → scp to VM → docker compose build --no-cache

## Key fixes applied (2026-04-21)
- **Footer not showing**: Three root causes found: (1) empty thinking chunks flooding the stream, (2) attribution chunk had different ID than other chunks, (3) `_is_internal_call()` scanned entire 50+ message history and found old "pre-compaction memory flush" markers, suppressing attribution on ALL requests
- **Tool call failures**: Router was stripping tools from Ollama requests, but qwen3.6 supports them natively
- **Discord gateway stuck**: "awaiting gateway readiness" → fixed with full `docker stop` + `docker start` (not just `restart`)

## Hardware
- RTX 3090 24GB VRAM, 62GB RAM
- Ollama on 0.0.0.0:11434 (systemd service)
- OLLAMA_NUM_GPU=30, OLLAMA_KV_CACHE_TYPE=q4_0

## Important notes
- Only download Ollama models at night to spare internet bandwidth
- qwen3:30b was faster (151 tok/s) but replaced by qwen3.6:35b-a3b-q4_K_M for tool support
- nvfp4 quantized models require macOS — not available on Linux/NVIDIA
- VM router is the production instance; host router is for testing
