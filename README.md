# OpenClaw LLM Router

A small FastAPI proxy that sits between [OpenClaw](https://github.com/coollabsio/openclaw) (or any OpenAI-compatible client) and multiple LLM backends. It classifies each incoming request with a tiny local model and routes it to the cheapest backend that can handle it.

```
client → router → classify → ┬─ simple   → Ollama       (free, local)
                             ├─ medium   → z.ai GLM-4.7 (cheap, cloud)
                             └─ complex  → z.ai GLM-4.7 (cheap, cloud)
```

The router speaks **OpenAI Chat Completions** on its public side and translates to whichever wire format the upstream needs. z.ai only ships an Anthropic-compatible API, so the router converts OpenAI ⇄ Anthropic Messages format (including SSE streaming) on the fly.

## Why

Self-hosted Discord bots like OpenClaw default to a single model. That's wasteful: a "hi" gets the same 30B coding model as a stack-trace debug. This router:

- Sends greetings and chitchat to a free local model
- Sends real questions to a cheap cloud model
- Costs nothing for trivia and ~pennies for the rest
- Is a drop-in replacement for any `openai-completions` provider — point your client's `baseUrl` at `http://router:4100/v1`

## Features

- **Three-tier routing** with a 1B–32B local classifier (configurable)
- **OpenAI ⇄ Anthropic Messages** translation, including streaming SSE
- **OpenClaw metadata stripping** — removes the three-block sender/conversation/untrusted wrapper before classification so the classifier sees the actual user text
- **Strict alternation enforcement** — merges consecutive same-role messages and strips leading assistant turns so Anthropic doesn't 400 the request
- **Per-tier `max_tokens` caps** — defends against degenerate local-model loops (we got bitten by a `# \n# \n#` spam incident; see the changelog in this repo's history)
- **Optional bearer auth** on the public port (`ROUTER_API_KEY`)
- **Health endpoint** at `/health` for Docker healthchecks

## Quick start

```bash
git clone git@github.com:steemandavid/openclaw-router.git
cd openclaw-router
cp .env.example .env
# edit .env — at minimum set ZAI_API_KEY
docker compose up -d --build
curl http://localhost:4100/health
```

Then point your OpenAI-compatible client at `http://<host>:4100/v1` with model `auto`.

## Configuration

All config is via env vars — see `.env.example` for the full list. Key knobs:

| Var | Default | Notes |
|---|---|---|
| `ZAI_API_KEY` | *(required for medium/complex)* | Get one at z.ai |
| `OLLAMA_BASE_URL` | `http://192.168.1.165:11434` | Where your Ollama lives |
| `ZAI_BASE_URL` | `https://api.z.ai/api/anthropic` | z.ai's Anthropic-compatible endpoint |
| `CLASSIFIER_MODEL` | `qwen2.5:32b-instruct-q4_K_M` | Same model as `SIMPLE_MODEL` by default so it stays pinned in VRAM |
| `SIMPLE_MODEL` | `qwen2.5:32b-instruct-q4_K_M` | Local Ollama model for the simple tier |
| `MEDIUM_MODEL` / `COMPLEX_MODEL` | `claude-sonnet-4-6-20250514` | z.ai aliases this to GLM-4.7 |
| `SIMPLE_MAX_TOKENS` | `512` | Hard cap to defend against degenerate loops |
| `ROUTER_API_KEY` | *(empty)* | Set to enable bearer auth |
| `PORT` | `4100` | |

### A word on classifier sizing

Tiny classifiers (`qwen2.5:0.5b`, `1.5b`) are fast but mis-classify coding requests as "simple". Worse: if your simple-tier and classifier are different models, Ollama will swap them in and out of VRAM, and a 30B reload can take 15+ minutes the first time. We recommend either:

1. **Path A (low VRAM)**: tiny classifier + tiny simple model — fast, accurate enough, no swapping
2. **Path B (lots of VRAM, recommended)**: same model for classifier and simple tier — pinned in VRAM with `OLLAMA_KEEP_ALIVE=-1`

`.env.example` defaults to Path B with a 23GB Qwen2.5-32B-Instruct.

## With OpenClaw

`init-models.sh` and `custom.json` are shipped as examples. They patch OpenClaw's `openclaw.json` at startup to add the router as a provider and set it as the default model. See `CLASSIFIER_AND_ROUTING_DESIGN.md` for the full story.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat completions (streaming and non-streaming)
- `GET /health` — liveness check
- `GET /v1/models` — lists `auto` so OpenAI clients are happy

## Notes and gotchas

- **z.ai has no native OpenAI API** — only Anthropic-compatible. The router does the translation.
- **z.ai's Zhipu native API** (`open.bigmodel.cn`) is **not** covered by the GLM Coding Plan — use `api.z.ai/api/anthropic` only.
- **`init-models.sh` deployment**: rebuilding the router image is required after editing `router.py` because the Dockerfile bakes the file in. `docker compose build && docker compose up -d`.
- **OpenClaw API types**: valid values for `models.providers.*.api` are `openai-completions`, `openai-responses`, `openai-codex-responses`, `anthropic-messages`, `google-generative-ai`, `github-copilot`, `bedrock-converse-stream`, `ollama`. `openai-compat` is **not** valid and will crash config validation.

## License

MIT — do whatever you want, no warranty.
