"""Thin OpenAI-compatible client for the configured endpoint.

Reads base URL / key / model from the gitignored .env file in the project
root. Never stores or prints the key. Supports structured JSON output with
robust recovery when the model wraps or mangles JSON.
"""
import json
import os
import re
import time
from pathlib import Path
import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _strip_quotes(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1]
    return v


def _load_env():
    env = {}
    p = ENV_PATH
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = _strip_quotes(v)
    # env vars override file (non-empty)
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not env.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing env config: {missing}. Create a .env file (see .env.example) "
            "or export LLM_BASE_URL/LLM_API_KEY/LLM_MODEL."
        )
    return env


def _extract_json(text: str):
    """Best-effort extraction of a JSON object from a model response."""
    if text is None:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # Strip markdown fences
    fences = re.findall(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    for f in fences:
        try:
            return json.loads(f.strip())
        except Exception:
            continue
    # Find the first {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


class LLMClient:
    def __init__(self, temperature=0.7, max_tokens=800, n_retries=5, backoff=2.0):
        env = _load_env()
        self.base_url = env["LLM_BASE_URL"].rstrip("/")
        self.api_key = env["LLM_API_KEY"]
        self.model = env["LLM_MODEL"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_retries = n_retries
        self.backoff = backoff

    def chat(self, messages, response_format=None, temperature=None, max_tokens=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(self.n_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError) as e:
                        last_err = f"unexpected response shape: {e}"
                        time.sleep(self.backoff)
                        continue
                    return content
                else:
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                last_err = f"request error: {e}"
            time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.n_retries} attempts: {last_err}")

    def structured(self, messages, temperature=0.0, max_tokens=None, json_retries=3):
        """Request a JSON object and return it as a dict, with fallback parsing.
        If the model's output is truncated/None (reasoning models sometimes eat
        the budget before any visible content), retry with an expanded budget."""
        mt = self.max_tokens if max_tokens is None else max_tokens
        last = None
        for attempt in range(json_retries):
            try:
                content = self.chat(messages, response_format={"type": "json_object"},
                                    temperature=temperature, max_tokens=mt)
            except Exception as e:
                last = e
                mt = mt * 2
                continue
            obj = _extract_json(content)
            if obj is not None:
                return obj
            last = RuntimeError(f"could not parse JSON from model output: {content!r}")
            mt = mt * 2
        raise RuntimeError(f"structured() failed after {json_retries} attempts: {last}")
