"""OpenClaw LLM Router Proxy

Sits between OpenClaw and model providers. Classifies incoming messages
with a cheap local model, then routes to the appropriate backend.

Architecture:
    OpenClaw → this router → classify with tiny local LLM
                                ↓
                            Ollama (free)    z.ai Anthropic (cheap/smart)

OpenClaw sends OpenAI-format requests. The router:
  - For Ollama (simple tier): passes through OpenAI format natively
  - For z.ai (medium/complex): converts OpenAI → Anthropic Messages API format,
    then converts the Anthropic response back to OpenAI format
"""

import os
import time
import json
import logging
from enum import Enum
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.165:11434")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/anthropic")
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen2.5:32b-instruct-q4_K_M")
CLASSIFIER_BASE_URL = os.getenv("CLASSIFIER_BASE_URL", OLLAMA_BASE_URL)
ROUTER_PORT = int(os.getenv("PORT", "4100"))

CLASSIFIER_TIMEOUT = float(os.getenv("CLASSIFIER_TIMEOUT", "15"))
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "180"))

# Optional bearer-token auth. If unset, router accepts all requests (backward compat).
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class ApiType(str, Enum):
    OPENAI = "openai"       # Ollama — native OpenAI format passthrough
    ANTHROPIC = "anthropic"  # z.ai — needs OpenAI ↔ Anthropic translation


class Tier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class Backend:
    url: str
    model: str
    api_type: ApiType = ApiType.OPENAI
    api_key: str = ""
    max_tokens_cap: int | None = None  # If set, clamp outgoing max_tokens to this value


TIERS: dict[Tier, Backend] = {
    Tier.SIMPLE: Backend(
        url=f"{OLLAMA_BASE_URL}/v1/chat/completions",
        model=os.getenv("SIMPLE_MODEL", "qwen2.5:32b-instruct-q4_K_M"),
        api_type=ApiType.OPENAI,
        # Bound worst-case simple-tier response length. Protects Discord against
        # degenerate token loops (e.g. thousands of `# ` headers) seen with
        # qwen3-coder:30b after VRAM starvation on 2026-04-13.
        max_tokens_cap=int(os.getenv("SIMPLE_MAX_TOKENS", "512")),
    ),
    Tier.MEDIUM: Backend(
        url=f"{ZAI_BASE_URL}/v1/messages",
        model=os.getenv("MEDIUM_MODEL", "claude-sonnet-4-6-20250514"),
        api_type=ApiType.ANTHROPIC,
        api_key=ZAI_API_KEY,
    ),
    Tier.COMPLEX: Backend(
        url=f"{ZAI_BASE_URL}/v1/messages",
        model=os.getenv("COMPLEX_MODEL", "claude-sonnet-4-6-20250514"),
        api_type=ApiType.ANTHROPIC,
        api_key=ZAI_API_KEY,
    ),
}

# ---------------------------------------------------------------------------
# Format conversion: OpenAI ↔ Anthropic
# ---------------------------------------------------------------------------


def _extract_text_content(content) -> str:
    """Flatten OpenAI message content (string or list of parts) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _openai_to_anthropic(body: dict, model: str) -> dict:
    """Convert OpenAI chat completion request to Anthropic Messages API format.

    Anthropic requires:
      - A top-level `system` string (not a message role)
      - Strict user/assistant alternation in `messages`
      - The first message must be `user`
      - Non-empty content blocks
    """
    messages = body.get("messages", [])
    system_text = ""
    anthropic_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        text = _extract_text_content(msg.get("content", ""))

        if role == "system":
            if text:
                system_text += text + "\n"
            continue

        if role not in ("user", "assistant"):
            continue
        if not text:
            continue

        # Merge consecutive same-role messages to enforce alternation
        if anthropic_messages and anthropic_messages[-1]["role"] == role:
            anthropic_messages[-1]["content"] += "\n\n" + text
        else:
            anthropic_messages.append({"role": role, "content": text})

    # Anthropic requires the first message to be from the user
    while anthropic_messages and anthropic_messages[0]["role"] != "user":
        anthropic_messages.pop(0)

    # Anthropic requires at least one message with non-empty content
    if not anthropic_messages:
        anthropic_messages.append({"role": "user", "content": " "})

    result = {
        "model": model,
        "max_tokens": body.get("max_tokens", 4096),
        "messages": anthropic_messages,
    }

    if system_text.strip():
        result["system"] = system_text.strip()

    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("stop") is not None:
        result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    return result


def _anthropic_to_openai(response_body: dict, model: str) -> dict:
    """Convert Anthropic Messages API response back to OpenAI format."""
    content_blocks = response_body.get("content", [])
    text = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text += block.get("text", "")

    usage = response_body.get("usage", {})

    return {
        "id": response_body.get("id", "chatcmpl-router"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop"
                if response_body.get("stop_reason") == "end_turn"
                else response_body.get("stop_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def _anthropic_stream_chunk_to_openai(line: str, model: str) -> str | None:
    """Convert a single Anthropic SSE line to OpenAI streaming format."""
    if not line.startswith("data: "):
        return None
    data = line[6:].strip()
    if data == "[DONE]":
        return "data: [DONE]\n\n"

    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None

    event_type = event.get("type", "")

    if event_type == "message_start":
        # Initial chunk with role
        chunk = {
            "id": "chatcmpl-router",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    elif event_type == "content_block_delta":
        delta = event.get("delta", {})
        text = delta.get("text", "")
        if text:
            chunk = {
                "id": "chatcmpl-router",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(chunk)}\n\n"

    elif event_type == "message_stop":
        # Final chunk with finish_reason, then [DONE]
        chunk = {
            "id": "chatcmpl-router",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = """\
You classify messages by complexity. Reply with exactly ONE word from: simple, medium, complex

simple - casual chat, greetings, opinions, yes/no, trivial questions:
  "hey how's it going?" -> simple
  "what's your favorite color?" -> simple
  "thanks!" -> simple
  "tell me a joke" -> simple

medium - factual questions, explanations, summaries, how-to, translations, comparisons:
  "what is a VPN and when should I use one?" -> medium
  "explain how DNS works" -> medium
  "summarize this article" -> medium
  "translate hello to French" -> medium
  "what's the difference between TCP and UDP?" -> medium

complex - coding, debugging, math proofs, multi-step reasoning, data analysis, system design:
  "write a Python function that implements binary search" -> complex
  "debug this error: NullPointerException at line 42" -> complex
  "design a database schema for an e-commerce app" -> complex
  "prove that sqrt(2) is irrational" -> complex
"""

CLASSIFIER_TEMPLATE = """\
Message: {message}

Classify (simple/medium/complex):"""


def _extract_user_message(body: dict) -> str:
    """Pull the last user message from the OpenAI messages array,
    stripping OpenClaw metadata wrappers to get just the actual text."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            raw = _extract_text_content(msg.get("content", ""))
            return _strip_openclaw_metadata(raw)
    return ""


def _strip_openclaw_metadata(text: str) -> str:
    """Remove OpenClaw's metadata wrappers to extract the user's actual message.

    OpenClaw wraps messages with three metadata blocks:
        1. Conversation info (untrusted metadata): ```json { ... } ```
        2. Sender (untrusted metadata): ```json { ... } ```
        3. Untrusted context (metadata, ...): <<<EXTERNAL_UNTRUSTED_CONTENT>>> ...

    The user's actual message sits between block 2 (last ```) and block 3.
    """
    marker = "Conversation info (untrusted metadata):"
    if marker not in text:
        return text

    # Find the last ``` closing marker — the user's text comes after it
    last_close = text.rfind("```")
    if last_close == -1:
        return text

    after = text[last_close + 3:].strip()

    # Strip the third metadata block ("Untrusted context ...") if present
    untrusted_marker = "Untrusted context"
    idx = after.find(untrusted_marker)
    if idx >= 0:
        after = after[:idx].strip()

    # Also strip <<<EXTERNAL_UNTRUSTED_CONTENT>>> markers if present
    ext_marker = "<<<EXTERNAL_UNTRUSTED_CONTENT"
    idx = after.find(ext_marker)
    if idx >= 0:
        after = after[:idx].strip()

    return after if after else text


def _parse_classification(text: str) -> Tier:
    """Parse the one-word classification from the LLM response."""
    text = text.strip().lower()
    for tier in Tier:
        if tier.value in text:
            return tier
    if any(w in text for w in ("greet", "hello", "hi", "chat")):
        return Tier.SIMPLE
    if any(w in text for w in ("code", "debug", "analy", "complex", "architect")):
        return Tier.COMPLEX
    return Tier.MEDIUM


async def classify(message: str) -> Tier:
    """Send the message to the classifier model and return the tier."""
    if not message.strip():
        return Tier.SIMPLE

    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": CLASSIFIER_TEMPLATE.format(message=message[:500])},
        ],
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=CLASSIFIER_TIMEOUT) as client:
            resp = await client.post(
                f"{CLASSIFIER_BASE_URL}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            tier = _parse_classification(text)
            log.info("Classified as %s (raw: %r)", tier.value, text[:60])
            return tier
    except Exception as exc:
        log.warning("Classifier failed (%s), defaulting to medium", exc)
        return Tier.MEDIUM


# ---------------------------------------------------------------------------
# Request proxying
# ---------------------------------------------------------------------------


_OLLAMA_UNSUPPORTED_FIELDS = (
    "tools",
    "tool_choice",
    "thinking",
    "thinking_budget",
    "reasoning_effort",
    "store",
    "max_completion_tokens",
)


def _normalize_messages(body: dict) -> dict:
    """Convert list-format message content to plain strings for Ollama compat.
    Also strip fields Ollama doesn't support (tools, thinking, etc.)."""
    body = {**body}
    messages = body.get("messages", [])
    normalized = []
    for msg in messages:
        msg = {**msg}
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = _extract_text_content(content)
        normalized.append(msg)
    body["messages"] = normalized
    for key in _OLLAMA_UNSUPPORTED_FIELDS:
        body.pop(key, None)
    return body


async def _proxy_openai(body: dict, backend: Backend, tier_name: str) -> Response:
    """Passthrough for OpenAI-compatible backends (Ollama)."""
    is_stream = body.get("stream", False)
    body = _normalize_messages({**body, "model": backend.model})
    if backend.max_tokens_cap is not None:
        original = body.get("max_tokens")
        capped = min(original, backend.max_tokens_cap) if original else backend.max_tokens_cap
        if capped != original:
            log.info("Clamping %s max_tokens %s → %d", tier_name, original, capped)
        body["max_tokens"] = capped
    log.debug("Sending to Ollama keys: %s", list(body.keys()))
    headers = {"Content-Type": "application/json"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"

    if not is_stream:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(backend.url, json=body, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )

    async def stream_gen():
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                async with client.stream("POST", backend.url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        log.error("Ollama stream error %d: %s", resp.status_code, error_body.decode(errors="replace")[:300])
                        yield error_body
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        finally:
            log.info("Streamed %s in %.1fs", tier_name, time.time() - t0)

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


async def _proxy_anthropic(body: dict, backend: Backend, tier_name: str) -> Response:
    """Translate OpenAI format → Anthropic Messages API, then translate response back."""
    anthropic_body = _openai_to_anthropic(body, backend.model)
    is_stream = body.get("stream", False)
    anthropic_body["stream"] = is_stream

    log.debug("Anthropic request body: %s", json.dumps(anthropic_body, ensure_ascii=False)[:500])

    headers = {
        "Content-Type": "application/json",
        "x-api-key": backend.api_key,
        "anthropic-version": "2023-06-01",
    }

    if not is_stream:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(backend.url, json=anthropic_body, headers=headers)
            if resp.status_code != 200:
                log.error("z.ai error %d: %s", resp.status_code, resp.text[:300])
                return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
            openai_resp = _anthropic_to_openai(resp.json(), backend.model)
            return Response(
                content=json.dumps(openai_resp).encode(),
                status_code=200,
                media_type="application/json",
            )

    # Streaming: translate Anthropic SSE → OpenAI SSE
    async def stream_gen():
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                async with client.stream("POST", backend.url, json=anthropic_body, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        log.error("z.ai stream error %d: %s", resp.status_code, error_body.decode(errors="replace")[:300])
                        # Surface the upstream error to OpenClaw as a synthetic OpenAI-shaped chunk
                        err_chunk = {
                            "id": "chatcmpl-router-error",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": backend.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"[router: upstream error {resp.status_code}]"},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n".encode()
                        return
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            converted = _anthropic_stream_chunk_to_openai(line, backend.model)
                            if converted:
                                yield converted.encode()
                    # Process remaining buffer
                    if buffer.strip():
                        converted = _anthropic_stream_chunk_to_openai(buffer.strip(), backend.model)
                        if converted:
                            yield converted.encode()
        finally:
            log.info("Streamed %s in %.1fs", tier_name, time.time() - t0)

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


async def proxy_to_backend(body: dict, tier: Tier) -> Response:
    """Forward the request to the chosen backend with appropriate format translation."""
    backend = TIERS[tier]
    if backend.api_type == ApiType.ANTHROPIC:
        return await _proxy_anthropic(body, backend, tier.value)
    return await _proxy_openai(body, backend, tier.value)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenClaw LLM Router")


def _check_auth(request: Request) -> bool:
    """If ROUTER_API_KEY is set, require a matching Bearer token. Otherwise allow all."""
    if not ROUTER_API_KEY:
        return True
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    return auth[7:].strip() == ROUTER_API_KEY


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _check_auth(request):
        return Response(
            content=json.dumps({"error": {"message": "unauthorized", "type": "auth_error"}}).encode(),
            status_code=401,
            media_type="application/json",
        )

    body = await request.json()

    user_msg = _extract_user_message(body)
    tier = await classify(user_msg)

    log.info(
        "Routing to %s (%s) for: %.80s",
        tier.value,
        TIERS[tier].model,
        user_msg[:80],
    )

    return await proxy_to_backend(body, tier)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "auto", "object": "model", "owned_by": "router"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "service": "openclaw-router",
        "classifier": CLASSIFIER_MODEL,
        "tiers": {t.value: {"model": b.model, "api": b.api_type.value} for t, b in TIERS.items()},
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting OpenClaw Router on port %d", ROUTER_PORT)
    log.info("Classifier: %s @ %s", CLASSIFIER_MODEL, CLASSIFIER_BASE_URL)
    log.info("Tiers: %s", {t.value: b.model for t, b in TIERS.items()})
    if not ZAI_API_KEY:
        log.warning("ZAI_API_KEY not set — z.ai backends will fail")
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_PORT, log_level=LOG_LEVEL.lower())
