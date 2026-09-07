"""Groq client setup for the hand-rolled agent loop.

Model choice: llama-3.3-70b-versatile (GROQ_MODEL_DEFAULT) is Groq's own
recommended default for tool-calling workloads and what the project spec
asks for. openai/gpt-oss-120b is kept as a documented fallback constant
(GROQ_MODEL_FALLBACK) rather than a choice made from an in-repo,
empirical side-by-side test: this build environment has no live
GROQ_API_KEY, so the "test both, document which handles tool-calling
better in practice" comparison the spec calls for could not actually be
run here (see the README's Limitations section - this is flagged
honestly rather than faked). The mechanism to compare is in place -
GROQ_MODEL is a single env var the loop reads, not hardcoded - so
whoever runs this with a real key can flip it and compare directly.
"""

from __future__ import annotations

import os

from groq import Groq

GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "openai/gpt-oss-120b"


def get_model() -> str:
    return os.environ.get("GROQ_MODEL", GROQ_MODEL_DEFAULT)


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Get a free key at console.groq.com and set it "
            "in your .env (see .env.example)."
        )
    return Groq(api_key=api_key)
