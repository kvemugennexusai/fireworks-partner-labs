"""shared/fw_client.py — thin helpers over the OpenAI-compatible Fireworks API.

Every project imports from here so conventions stay consistent. The API key is read
from the FIREWORKS_API_KEY environment variable — never hardcode it.
"""
import os

from openai import OpenAI

BASE_URL = "https://api.fireworks.ai/inference/v1"

# Catalog as of writing — verify with list_models() at runtime; catalogs change.
TEXT_MODELS = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/kimi-k2p5",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/gpt-oss-120b",
]


def client():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("Set FIREWORKS_API_KEY in your environment (see .env.example).")
    return OpenAI(base_url=BASE_URL, api_key=key)


def list_models():
    """Return the ids your account can actually reach right now."""
    return [m.id for m in client().models.list().data]


def chat(model, messages, max_tokens=256, temperature=0.0, **kw):
    """One non-streaming chat completion -> text."""
    r = client().chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature, **kw,
    )
    return r.choices[0].message.content or ""


def short_name(model_id):
    """'accounts/fireworks/models/glm-5p1' -> 'glm-5p1' for table rows."""
    return model_id.rstrip("/").split("/")[-1]


if __name__ == "__main__":
    for m in list_models():
        print(m)
