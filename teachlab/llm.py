"""OpenAI-compatible model endpoints, one per role (student / teacher / judge).

Any server that speaks the chat-completions API works: OpenAI, DeepSeek, or a local vLLM serving Qwen3.
"""

import json
import os
import re
import threading
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)

# Requests that fail the same way every time (e.g. a prompt longer than the context window, a bad model name or
# key) are raised at once instead of being retried with backoff.
NON_RETRYABLE = (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError,
                 openai.NotFoundError, openai.UnprocessableEntityError)


class Model:
    def __init__(self, name, base_url=None, api_key_env="OPENAI_API_KEY", extra_body=None, max_tokens=None,
                 timeout=300):
        self.name = name
        # A hard per-request timeout: a dropped connection otherwise hangs the whole run. Retries are ours (chat()).
        # Raise it for long generations on a shared GPU (e.g. a thinking model at ~30 tokens/s per request).
        self.client = OpenAI(api_key=os.getenv(api_key_env) or "EMPTY", base_url=base_url, timeout=timeout,
                             max_retries=0)
        self.extra_body = extra_body or {}
        self.max_tokens = max_tokens
        self.usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        self._lock = threading.Lock()

    def _count(self, usage):
        if usage is None:
            return
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        # OpenAI reports cache hits in prompt_tokens_details, DeepSeek as prompt_cache_hit_tokens
        cached = getattr(prompt_details, "cached_tokens", None) or getattr(usage, "prompt_cache_hit_tokens", None) or 0
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += usage.prompt_tokens or 0
            self.usage["cached_tokens"] += cached
            self.usage["completion_tokens"] += usage.completion_tokens or 0
            self.usage["reasoning_tokens"] += getattr(completion_details, "reasoning_tokens", None) or 0

    def __repr__(self):
        return self.name

    def chat(self, messages, max_retries=6, retry_delay=3):
        for attempt in range(max_retries):
            try:
                kwargs = {"extra_body": self.extra_body} if self.extra_body else {}
                if self.max_tokens:  # max_completion_tokens: required by OpenAI's reasoning models, accepted by vLLM
                    kwargs["max_completion_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(model=self.name, messages=messages, **kwargs)
                self._count(response.usage)
                # Reasoning models served by vLLM may inline their thinking; keep only the answer.
                return _THINK.sub("", response.choices[0].message.content or "").strip()
            except NON_RETRYABLE:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = min(60, retry_delay * 2 ** attempt)
                print(f"   ⚠️ {self.name} call failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)[:100]}")
                time.sleep(delay)


def add_model_args(parser, role, default):
    parser.add_argument(f"--{role}", default=default, help=f"{role} model name")
    parser.add_argument(f"--{role}-url", default=None, help=f"base URL of the {role}'s endpoint (default: OpenAI)")
    parser.add_argument(f"--{role}-key-env", default="OPENAI_API_KEY", help=f"env var holding the {role}'s API key")
    parser.add_argument(f"--{role}-extra", default=None, help=f"JSON extra_body for the {role} (e.g. vLLM options)")
    parser.add_argument(f"--{role}-max-tokens", type=int, default=None,
                        help=f"cap on the {role}'s completion tokens (default: the server's)")
    parser.add_argument(f"--{role}-timeout", type=float, default=300, help=f"per-request timeout for the {role}, seconds")


def model_from_args(args, role):
    extra = getattr(args, f"{role}_extra")
    return Model(getattr(args, role), getattr(args, f"{role}_url"), getattr(args, f"{role}_key_env"),
                 json.loads(extra) if extra else None, getattr(args, f"{role}_max_tokens"), getattr(args, f"{role}_timeout"))
