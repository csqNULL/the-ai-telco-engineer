# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""JSON extraction utilities for parsing LLM output."""

import json
import re
from typing import Optional

_JSON_DECODER = json.JSONDecoder()


# Match a fence whose opener is a run of 3 or more backticks. The closing
# fence must use the *same* run length, so an inner ``` block does not
# terminate an outer ```` block (markdown's standard nesting convention).
_FENCE_RE = re.compile(r"(`{3,})[^\n`]*\n?([\s\S]*?)\1")


def strip_code_fences(text: str) -> str:
    """Return the inner content of the **last** fenced block in *text*.

    The framework's contract with LLMs is: the final answer is the last
    triple-backtick fenced block in the reply; any preamble, reasoning
    trace (``<think>...</think>``, ``<|channel|>analysis...``, plain
    prose, draft fenced blocks, etc.) is ignored. Taking the *last*
    fence — not the first — is what makes this robust to reasoning
    models that emit draft answers inside their scratchpad.

    Fences using a longer backtick run (e.g. four backticks) are
    matched against a closer of the same length, so models can wrap
    content that itself contains ``` blocks by using ```` outside.
    Any language tag (``json``, ``python``, none, ...) is accepted.

    If no fenced block is found, the original text is returned
    unchanged so downstream parsers (e.g. ``extract_json_fragment``)
    can still try.
    """
    text = (text or "").strip()
    matches = list(_FENCE_RE.finditer(text))
    return matches[-1].group(2).strip() if matches else text


def extract_json_fragment(text: str, open_char: str) -> Optional[str]:
    """Extract a JSON array or object from *text*.

    Scans every position where *open_char* (``[`` or ``{``) appears and
    attempts to JSON-decode from there. Returns the **last** position
    that yields a valid JSON value of the matching shape. Returns
    ``None`` if no position decodes successfully.

    "Last" matches the same philosophy as ``strip_code_fences``: when a
    reasoning model emits draft JSON inside its scratchpad and the real
    answer afterwards, the real answer wins because final answers come
    last by convention. Using ``json.JSONDecoder.raw_decode`` rather
    than a hand-rolled bracket-balancing scan automatically rejects
    pseudo-JSON drafts that contain placeholders like ``...``, since
    those fail JSON validation.
    """
    expected = list if open_char == "[" else dict
    last: Optional[str] = None
    i = 0
    n = len(text)
    while i < n:
        i = text.find(open_char, i)
        if i == -1:
            break
        try:
            value, end = _JSON_DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(value, expected):
            last = text[i:end]
        i = end
    return last
