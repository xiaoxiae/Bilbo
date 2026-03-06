# Alignment algorithm inspired by Bertalign (https://github.com/bfsujason/bertalign)
# Copyright (C) 2021 Jason Li, licensed under GPL-3.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .models import Alignment, AlignmentPair, SegmentedText

if TYPE_CHECKING:
    from .log import PipelineLog

# All (di, dj) moves for fill-between DP: di in [1..5], dj in [1..5]
MOVES = [(di, dj) for di in range(1, 6) for dj in range(1, 6)]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"



def _normalize(emb: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows in-place and return."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    emb /= norms
    return emb


def _block_similarity(
    l1_emb: np.ndarray, l2_emb: np.ndarray, i: int, j: int, di: int, dj: int
) -> float:
    """Average cosine similarity for a block of (di, dj) sentences starting at (i, j).

    Assumes embeddings are already L2-normalized.
    """
    block = l1_emb[i : i + di] @ l2_emb[j : j + dj].T
    return float(block.mean())


def _find_anchors(
    l1_emb: np.ndarray,
    l2_emb: np.ndarray,
    win: int = 50,
    skip_penalty: float = 0.0,
) -> list[tuple[int, int]]:
    """Pass 1: windowed 1-1 DP to find high-confidence anchor pairs."""
    n = len(l1_emb)
    m = len(l2_emb)
    INF = -1e18

    # Similarity matrix (already L2-normalized)
    sim = l1_emb @ l2_emb.T  # (n, m)

    # DP table: dp[i][j] = best score aligning l1[:i] with l2[:j]
    # Transitions from (i, j):
    #   skip L1: go to (i+1, j) with cost skip_penalty
    #   skip L2: go to (i, j+1) with cost skip_penalty
    #   align 1-1: go to (i+1, j+1) with cost sim[i, j]
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    # backpointer: 0=none, 1=skip_l1 (from i-1,j), 2=skip_l2 (from i,j-1), 3=align (from i-1,j-1)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(n + 1):
        # Window: j should be near i * m / n
        center = int(round(i * m / n)) if n > 0 else 0
        j_lo = max(0, center - win)
        j_hi = min(m, center + win)

        for j in range(j_lo, j_hi + 1):
            if dp[i, j] == INF:
                continue
            val = dp[i, j]

            # Skip L1 (advance i, keep j)
            if i < n:
                new_val = val + skip_penalty
                if new_val > dp[i + 1, j]:
                    dp[i + 1, j] = new_val
                    back[i + 1, j] = 1

            # Skip L2 (keep i, advance j)
            if j < m:
                new_val = val + skip_penalty
                if new_val > dp[i, j + 1]:
                    dp[i, j + 1] = new_val
                    back[i, j + 1] = 2

            # Align 1-1
            if i < n and j < m:
                new_val = val + sim[i, j]
                if new_val > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = new_val
                    back[i + 1, j + 1] = 3

    # Backtrace from (n, m)
    pairs = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        b = back[ci, cj]
        if b == 0:
            break
        if b == 1:  # skip L1
            ci -= 1
        elif b == 2:  # skip L2
            cj -= 1
        elif b == 3:  # aligned
            ci -= 1
            cj -= 1
            pairs.append((ci, cj))
    pairs.reverse()

    if not pairs:
        return []

    # Filter to high-confidence anchors: sim > mean + 0.5 * std
    sims = np.array([sim[i, j] for i, j in pairs])
    threshold = sims.mean() + 0.5 * sims.std()
    anchors = [(i, j) for (i, j), s in zip(pairs, sims) if s >= threshold]

    return anchors


def _fill_between(
    l1_emb: np.ndarray,
    l2_emb: np.ndarray,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
) -> list[tuple[list[int], list[int]]]:
    """Pass 2: small unconstrained m-n DP between two anchor boundaries."""
    n = i_end - i_start
    m = j_end - j_start

    if n == 0 or m == 0:
        return []

    INF = -1e18
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    back = {}

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] == INF:
                continue
            val = dp[i, j]
            for di, dj in MOVES:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                score = val + _block_similarity(
                    l1_emb, l2_emb, i_start + i, j_start + j, di, dj
                )
                if score > dp[ni, nj]:
                    dp[ni, nj] = score
                    back[(ni, nj)] = (di, dj)

    # Backtrace
    path = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        if (ci, cj) not in back:
            break
        di, dj = back[(ci, cj)]
        pi, pj = ci - di, cj - dj
        src_idxs = list(range(i_start + pi, i_start + pi + di))
        tgt_idxs = list(range(j_start + pj, j_start + pj + dj))
        path.append((src_idxs, tgt_idxs))
        ci, cj = pi, pj

    path.reverse()
    return path


def _two_pass_align(
    l1_emb: np.ndarray,
    l2_emb: np.ndarray,
    on_progress: Callable[[float, float | None], None] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Two-pass alignment: find anchors, then fill between them."""
    n = len(l1_emb)
    m = len(l2_emb)

    if on_progress:
        on_progress(0, 2)

    anchors = _find_anchors(l1_emb, l2_emb)

    if on_progress:
        on_progress(1, 2)

    # Build boundary list: gaps before first anchor, between anchors, after last anchor
    boundaries = []
    prev_i, prev_j = 0, 0
    for ai, aj in anchors:
        boundaries.append((prev_i, ai, prev_j, aj))
        # The anchor itself is a 1:1 pair
        prev_i = ai + 1
        prev_j = aj + 1
    # After last anchor
    boundaries.append((prev_i, n, prev_j, m))

    result = []
    for idx, (i_start, i_end, j_start, j_end) in enumerate(boundaries):
        # Fill the gap
        filled = _fill_between(l1_emb, l2_emb, i_start, i_end, j_start, j_end)
        result.extend(filled)

        # If this is not the last boundary, add the anchor pair
        if idx < len(anchors):
            ai, aj = anchors[idx]
            result.append(([ai], [aj]))

    if on_progress:
        on_progress(2, 2)

    return result


def _silence_hf_logging() -> None:
    """Suppress noisy output from the HuggingFace / transformers stack.

    Loading a SentenceTransformer model produces three kinds of spam:
    1. "Warning: You are sending unauthenticated requests to the HF Hub" —
       emitted by huggingface_hub's HTTP layer when it sees an X-HF-Warning
       response header.  Silenced via hf_hub's own verbosity API.
    2. "BertModel LOAD REPORT … embeddings.position_ids UNEXPECTED" —
       emitted by transformers' state-dict loader when checkpoint keys don't
       match the model 1:1 (harmless for LaBSE).  Silenced via transformers'
       verbosity API.
    3. "Loading weights: 100%" tqdm bar — transformers wraps safetensors
       loading in its own tqdm.  Silenced via transformers' progress-bar toggle.
    """
    import logging

    import huggingface_hub.utils.logging as hf_logging
    import transformers.utils.logging as tf_logging

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()


def align_texts(
    l1: SegmentedText,
    l2: SegmentedText,
    device: str = "cpu",
    log: PipelineLog | None = None,
) -> Alignment:
    resolved = _resolve_device(device)
    if log:
        log.info(f"Computing embeddings (LaBSE) on {resolved}...")

    l1_texts = [s.text for s in l1.sentences]
    l2_texts = [s.text for s in l2.sentences]

    from sentence_transformers import SentenceTransformer

    _silence_hf_logging()
    model = SentenceTransformer("sentence-transformers/LaBSE", device=_resolve_device(device))
    all_emb = _normalize(model.encode(l1_texts + l2_texts, show_progress_bar=False, convert_to_numpy=True))
    l1_emb = all_emb[:len(l1_texts)]
    l2_emb = all_emb[len(l1_texts):]

    p = log.progress("Alignment", unit="") if log else None
    raw_pairs = _two_pass_align(l1_emb, l2_emb, on_progress=p.update if p else None)
    if p:
        p.finish(f"{len(raw_pairs)} pairs")

    pairs = []
    for src_idxs, tgt_idxs in raw_pairs:
        l1_segs = [l1.sentences[i] for i in src_idxs]
        l2_segs = [l2.sentences[i] for i in tgt_idxs]
        pairs.append(AlignmentPair(l1=l1_segs, l2=l2_segs))

    return Alignment(pairs=pairs)
