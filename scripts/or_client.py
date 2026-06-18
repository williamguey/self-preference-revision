"""Minimal OpenRouter client with retries. Used for all model calls."""
import json
import os
import time
import urllib.request
import urllib.error

# Read the OpenRouter API key from the environment. Never hard-code a key here.
#   export OPENROUTER_API_KEY=your-key-here     (bash)
#   $env:OPENROUTER_API_KEY="your-key-here"     (PowerShell)
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE = "https://openrouter.ai/api/v1"


def _require_key():
    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it as an environment variable "
            "before running (see README).")

# rough USD pricing per 1M tokens (input, output) for cost estimation only
PRICING = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
}

_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "est_usd": 0.0}


def usage():
    return dict(_usage)


def list_models():
    _require_key()
    req = urllib.request.Request(f"{BASE}/models",
                                 headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]


def set_pricing(model_ids):
    """Populate PRICING (USD per 1M tokens) for given model ids from the API."""
    by_id = {m["id"]: m for m in list_models()}
    for mid in model_ids:
        if mid in by_id:
            p = by_id[mid].get("pricing", {})
            PRICING[mid] = (float(p.get("prompt", 0)) * 1e6,
                            float(p.get("completion", 0)) * 1e6)
    return {k: PRICING[k] for k in model_ids if k in PRICING}


def chat(model, messages, temperature=0.0, max_tokens=1200, retries=5):
    """Call chat completions. Returns (text, raw_json). Retries on errors."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    _require_key()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{BASE}/chat/completions",
                                         data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            if "choices" not in data or not data["choices"]:
                raise RuntimeError(f"no choices: {json.dumps(data)[:300]}")
            text = data["choices"][0]["message"]["content"]
            u = data.get("usage", {}) or {}
            pt = u.get("prompt_tokens", 0)
            ct = u.get("completion_tokens", 0)
            _usage["prompt_tokens"] += pt
            _usage["completion_tokens"] += ct
            _usage["calls"] += 1
            if model in PRICING:
                ip, op = PRICING[model]
                _usage["est_usd"] += pt / 1e6 * ip + ct / 1e6 * op
            return text, data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            last_err = f"HTTP {e.code}: {body}"
            # 4xx other than 429 won't fix on retry
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(last_err)
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")
