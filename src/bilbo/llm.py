from __future__ import annotations

import json
import urllib.request
import urllib.error

_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3:8b"
OLLAMA_HEALTH_TIMEOUT = 5
OLLAMA_GENERATE_TIMEOUT = 30


def is_available() -> bool:
    """Check if ollama is reachable."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_BASE}/", method="GET")
        with urllib.request.urlopen(req, timeout=OLLAMA_HEALTH_TIMEOUT):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _generate(prompt: str, model: str, schema: dict | None = None) -> str:
    """Call ollama generate API and return the response text.

    If schema is provided, uses structured output (format=json_schema).
    """
    body: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    if schema is not None:
        body["format"] = schema
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_GENERATE_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "")


def _simple_merge(l1: str, l2: str) -> str:
    """Fallback: join with ' / ', deduplicating if identical."""
    if l1 == l2:
        return l1
    return f"{l1} / {l2}"


def merge_metadata_text(
    pairs: dict[str, tuple[str, str]],
    model: str = DEFAULT_LLM_MODEL,
) -> dict[str, str]:
    """Batch-merge multiple L1/L2 text pairs via ollama in a single prompt.

    pairs: {"title": ("English Title", "German Title"), ...}
    Returns: {"title": "merged title", ...}
    Falls back to " / " joining on any error.
    """
    if not pairs:
        return {}

    fallback = {
        key: _simple_merge(l1, l2) for key, (l1, l2) in pairs.items()
    }

    try:
        items = []
        for key, (l1, l2) in pairs.items():
            items.append({"key": key, "l1": l1, "l2": l2})

        prompt = (
            "We are creating a bilingual audiobook that interleaves two audiobooks of the same book in different languages.\n"
            "The output file needs a single set of metadata. Merge these L1/L2 metadata pairs into clean, concise strings.\n"
            "Rules:\n"
            "- Identical values → use one copy\n"
            "- Shared series/franchise name → keep it once, show both unique parts with \" / \"\n"
            "- Different values → join with \" / \"\n"
            "- For artist/author: drop translators, narrators, keep only the original author\n\n"
            f"Input:\n{json.dumps(items, ensure_ascii=False)}\n"
        )

        keys = list(pairs.keys())
        schema = {
            "type": "object",
            "properties": {k: {"type": "string"} for k in keys},
            "required": keys,
        }

        raw = _generate(prompt, model, schema=schema)
        result = json.loads(raw)
        if not isinstance(result, dict):
            return fallback

        merged = {}
        for key in pairs:
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                merged[key] = val.strip()
            else:
                merged[key] = fallback[key]
        return merged

    except Exception:
        return fallback


def merge_chapter_titles(
    chapters: list[tuple[str, list[str]]],
    model: str = DEFAULT_LLM_MODEL,
) -> list[str]:
    """Merge chapter titles: each entry is (l1_title, [l2_titles...]).

    Returns list of merged title strings, same length as input.
    Falls back to "l1 / l2a, l2b" on error.
    """
    if not chapters:
        return []

    fallback = []
    for l1_title, l2_titles in chapters:
        if l2_titles:
            l2_part = ", ".join(l2_titles)
            fallback.append(_simple_merge(l1_title, l2_part))
        else:
            fallback.append(l1_title)

    try:
        items = []
        for i, (l1_title, l2_titles) in enumerate(chapters):
            items.append({"index": i, "l1": l1_title, "l2": l2_titles})

        prompt = (
            "Merge chapter titles from two editions of the same audiobook (L1 and L2 are different languages).\n"
            "Each entry has one L1 title and one or more L2 titles that correspond to it.\n"
            "The L1 title is the PRIMARY title — always keep it. Add L2 info only if it adds value.\n"
            "Rules:\n"
            "- Equivalent titles (e.g. \"Chapter 1\" and \"Kapitel 1\") → just use the L1 title\n"
            "- Multiple L2 titles with sequential numbers → compress and append (e.g. L1=\"Chapter 2\", L2=[\"Track 1\",\"Track 2\",\"Track 3\"] → \"Chapter 2 (Tracks 1-3)\")\n"
            "- Different titles → \"L1 / L2\"\n"
            "- Keep it short\n\n"
            f"Input:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            "Return the merged titles as t0, t1, t2, ... keys.\n"
        )

        # Use a wrapper object since ollama structured output requires an object at top level
        prop_keys = [f"t{i}" for i in range(len(chapters))]
        schema = {
            "type": "object",
            "properties": {k: {"type": "string"} for k in prop_keys},
            "required": prop_keys,
        }

        raw = _generate(prompt, model, schema=schema)
        result = json.loads(raw)
        if not isinstance(result, dict):
            return fallback

        merged = []
        for i in range(len(chapters)):
            val = result.get(f"t{i}")
            if isinstance(val, str) and val.strip():
                merged.append(val.strip())
            else:
                merged.append(fallback[i])
        return merged

    except Exception:
        return fallback
