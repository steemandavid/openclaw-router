#!/bin/bash
# Init script to patch OpenClaw model configuration
# Points OpenClaw at the router proxy instead of direct Ollama

CONFIG_FILE="/data/.openclaw/openclaw.json"

(
    echo "[init-models] Waiting for configure.js to write config..."
    for i in $(seq 1 30); do
        if [ -f "$CONFIG_FILE" ]; then
            if grep -q '"llama3.3"' "$CONFIG_FILE" 2>/dev/null; then
                sleep 1
                break
            fi
        fi
        sleep 1
    done

    if [ -f "$CONFIG_FILE" ]; then
        echo "[init-models] Patching router config..."
        node -e "
const fs = require('fs');
const configPath = '$CONFIG_FILE';
try {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

    // Set up router provider
    config.models = config.models || {};
    config.models.providers = config.models.providers || {};
    config.models.providers.router = {
        api: 'openai-completions',
        baseUrl: 'http://router:4100/v1',
        models: [
            { id: 'auto', name: 'Auto-Route (LLM Router)', contextWindow: 32768 }
        ],
        apiKey: 'router-local'
    };

    // Keep Ollama provider for memory search
    if (!config.models.providers.ollama) {
        config.models.providers.ollama = {
            api: 'openai-completions',
            baseUrl: 'http://192.168.1.165:11434/v1',
            models: [
                { id: 'qwen3-coder:30b-q5_k_m', name: 'Qwen3 Coder 30B (Q5)', contextWindow: 8192 },
                { id: 'qwen2.5:32b-instruct-q4_K_M', name: 'Qwen2.5 32B', contextWindow: 32768 }
            ],
            apiKey: 'ollama-local'
        };
    }

    // Set primary model to router
    config.agents = config.agents || {};
    config.agents.defaults = config.agents.defaults || {};
    config.agents.defaults.model = config.agents.defaults.model || {};
    config.agents.defaults.model.primary = 'router/auto';

    // Ensure memory search still uses Ollama embeddings
    config.agents.defaults.memorySearch = {
        provider: 'ollama',
        model: 'nomic-embed-text',
        remote: { baseUrl: 'http://192.168.1.165:11434' }
    };

    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    console.log('[init-models] Router config applied');
} catch (err) { console.error('[init-models] Error:', err.message); }
"
    else
        echo "[init-models] Config file not found after waiting: $CONFIG_FILE"
    fi
) &
echo "[init-models] Background patcher started"
