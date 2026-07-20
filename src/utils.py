# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared utilities for the agent and agent manager."""

import random
import time
from typing import Callable, TypeVar

import printer


T = TypeVar("T")

# Rate limit retry configuration
MAX_RATE_LIMIT_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10
MAX_BACKOFF_SECONDS = 120

# API connectivity retry configuration: retry indefinitely with constant
# backoff so that a long endpoint outage (minutes to hours) does not stop
# the optimization. Worker agents keep their default fast-fail behaviour
# (failed TaskResult, worker stays alive); only orchestrator/prompt-engineer
# LLM calls go through this loop.
API_DOWN_BACKOFF_SECONDS = 30

# Parse-failure retry configuration
MAX_PARSE_RETRIES = 3


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit error (HTTP 429)."""
    error_str = str(error).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "rate_limit" in error_str
        or "too many requests" in error_str
        or "model capacity reached" in error_str
    )


def is_api_connectivity_error(error: Exception) -> bool:
    """Check if an exception indicates the endpoint is unreachable or
    transiently broken (network drop, server-side 5xx, gateway timeout).

    Matches against the OpenAI SDK exception types when available, and
    falls back to substring matching on the error message so wrapped
    or stringified errors are still caught.
    """
    try:
        import openai
        if isinstance(error, (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )):
            return True
    except ImportError:
        pass
    msg = str(error).lower()
    return (
        "connection error" in msg
        or "connection reset" in msg
        or "connection aborted" in msg
        or "connection refused" in msg
        or "remote disconnected" in msg
        or "temporarily unavailable" in msg
        or "service unavailable" in msg
        or "bad gateway" in msg
        or "gateway timeout" in msg
        or "internal server error" in msg
        or " 500 " in f" {msg} "
        or " 502 " in f" {msg} "
        or " 503 " in f" {msg} "
        or " 504 " in f" {msg} "
    )


def invoke_llm_with_retry(llm, prompt: str, context: str = "LLM"):
    """
    Invoke LLM with a prompt; retry on transient errors.

    Two retry policies are layered:

    * **Rate limit** (HTTP 429 and friends): bounded exponential backoff
      with jitter, up to ``MAX_RATE_LIMIT_RETRIES`` attempts.
    * **API connectivity** (network drops, server 5xx, gateway timeouts):
      **unbounded** constant backoff at ``API_DOWN_BACKOFF_SECONDS``
      seconds per attempt, so an endpoint outage of minutes to hours
      does not stop the optimization. The caller may interrupt the loop
      with Ctrl-C; ``time.sleep`` propagates ``KeyboardInterrupt``.

    All other exceptions (auth errors, malformed prompts, etc.) are
    raised immediately so genuine bugs surface fast.

    Returns the response object (use ``.content`` for the string).
    """
    rate_limit_attempt = 0
    connectivity_attempt = 0
    while True:
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if (
                is_rate_limit_error(e)
                and rate_limit_attempt < MAX_RATE_LIMIT_RETRIES
            ):
                backoff = min(
                    INITIAL_BACKOFF_SECONDS * (2 ** rate_limit_attempt),
                    MAX_BACKOFF_SECONDS,
                )
                jitter = random.uniform(0, backoff * 0.25)
                wait_time = backoff + jitter
                printer.log(
                    f"Rate limit hit ({context}), attempt "
                    f"{rate_limit_attempt + 1}/{MAX_RATE_LIMIT_RETRIES}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                rate_limit_attempt += 1
                continue
            if is_api_connectivity_error(e):
                connectivity_attempt += 1
                # Throttle log noise on long outages: log the first 5
                # attempts, then every 10th, with cumulative wait time.
                should_log = (
                    connectivity_attempt <= 5
                    or connectivity_attempt % 10 == 0
                )
                if should_log:
                    cumulative_s = (
                        connectivity_attempt * API_DOWN_BACKOFF_SECONDS
                    )
                    printer.log(
                        f"API endpoint unavailable ({context}): "
                        f"{type(e).__name__}: {e}. "
                        f"Retry {connectivity_attempt} in "
                        f"{API_DOWN_BACKOFF_SECONDS}s "
                        f"(cumulative wait ~{cumulative_s}s, "
                        f"will retry indefinitely)..."
                    )
                time.sleep(API_DOWN_BACKOFF_SECONDS)
                continue
            raise


def retry_on_parse_failure(
    fn: Callable[[], T],
    max_retries: int = MAX_PARSE_RETRIES,
    context: str = "LLM",
) -> T:
    """Call *fn* up to *max_retries* + 1 times, retrying on any exception.

    Intended for wrapping LLM-call-then-parse sequences where the LLM may
    return unparseable output.  No backoff is applied between attempts.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                printer.log(
                    f"Parse/validation failure ({context}), "
                    f"attempt {attempt + 1}/{max_retries}: {e}. Retrying..."
                )
    raise last_error  # type: ignore[misc]
