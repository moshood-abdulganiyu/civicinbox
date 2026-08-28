"""
Thin wrapper around the OpenAI API for CivicInbox's LLM-based
classification (Steps 12-13). This step only proves the client can be
configured and called end-to-end (mocked) — structured JSON parsing,
Pydantic validation, and retry logic land in Step 12, once there's a
schema on the other end worth retrying for.

This is the ONE place in the whole project that costs real money
(master plan, budget-aware stack rule). Keep calls out of hot paths
and out of anything that runs in a loop without a human deciding to
trigger it.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Loads .env into os.environ if present (local dev). No-op in CI/prod
# where OPENAI_API_KEY is set directly as a real environment variable
# and no .env file exists.
load_dotenv()

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy singleton — same pattern as baseline_classifier._get_pipeline().
    Importing this module (e.g. via app.main during test collection)
    must not require a real API key. The key is only checked once a
    call is actually attempted.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and "
                "fill in a real key from https://platform.openai.com/api-keys"
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a single-turn prompt to the LLM and return the raw text
    response. No JSON parsing, schema validation, or retry logic yet —
    Step 12 adds all three once app/models/schema.py's TriageResult is
    what the prompt is asking for.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
