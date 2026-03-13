# Alignment algorithm inspired by Bertalign (https://github.com/bfsujason/bertalign)
# Copyright (C) 2021 Jason Li, licensed under GPL-3.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .models import Alignment, AlignmentPair, SegmentedText

MIN_STEP_RATIO = 0.25
MAX_STEP_RATIO = 4.0

if TYPE_CHECKING:
    from .log import PipelineLog

# Allowed (di, dj) moves for fill-between DP
MOVES = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]




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
    l1_texts: list[str],
    l2_texts: list[str],
    win: int = 250,
    min_words: int = 4,
    min_sim: float = 0.7,
) -> list[tuple[int, int]]:
    """Find anchor pairs: high-similarity, long-enough sentences, monotonic.

    For each L1 sentence with enough words, find the best L2 match within
    a window around the expected diagonal position.
    """
    n = len(l1_emb)
    m = len(l2_emb)

    anchors = []
    last_i, last_j = 0, 0
    for i in range(n):
        if len(l1_texts[i].split()) < min_words:
            continue
        # Search forward from the last anchor, scaled by how far i advanced
        center = last_j + int(round((i - last_i) * m / n))
        j_lo = max(last_j + 1, center - win)
        j_hi = min(m, center + win)
        if j_lo >= j_hi:
            continue
        sims = l1_emb[i] @ l2_emb[j_lo:j_hi].T
        # Try candidates in descending similarity order
        order = np.argsort(sims)[::-1]
        for idx in order:
            score = float(sims[idx])
            if score < min_sim:
                break
            j = j_lo + int(idx)
            if len(l2_texts[j].split()) < min_words:
                continue
            # Reject if step ratio deviates too far from global m/n
            di = i - last_i
            dj = j - last_j
            expected_dj = di * m / n
            if expected_dj > 0 and not (MIN_STEP_RATIO < dj / expected_dj < MAX_STEP_RATIO):
                continue
            anchors.append((i, j))
            last_i, last_j = i, j
            break

    return anchors


def _fill_between(
    l1_emb: np.ndarray,
    l2_emb: np.ndarray,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
) -> list[tuple[list[int], list[int], float]]:
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
                sim = _block_similarity(
                    l1_emb, l2_emb, i_start + i, j_start + j, di, dj
                )
                score = val + sim * (di + dj)
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
        pair_score = _block_similarity(
            l1_emb, l2_emb, i_start + pi, j_start + pj, di, dj
        )
        path.append((src_idxs, tgt_idxs, pair_score))
        ci, cj = pi, pj

    path.reverse()
    return path


def _two_pass_align(
    l1_emb: np.ndarray,
    l2_emb: np.ndarray,
    l1_texts: list[str],
    l2_texts: list[str],
    on_gap_progress: Callable[[int, int], None] | None = None,
    log: PipelineLog | None = None,
) -> tuple[list[tuple[list[int], list[int], float]], list[tuple[int, int]]]:
    """Two-pass alignment: find anchors, then fill between them.

    Returns (pairs, anchors).
    """
    n = len(l1_emb)
    m = len(l2_emb)

    a = log.activity("Finding anchors...") if log else None
    anchors = _find_anchors(l1_emb, l2_emb, l1_texts, l2_texts)
    if a:
        a.done(f"{len(anchors)} anchors")

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

    result: list[tuple[list[int], list[int], float]] = []
    for idx, (i_start, i_end, j_start, j_end) in enumerate(boundaries):
        # Fill the gap
        filled = _fill_between(l1_emb, l2_emb, i_start, i_end, j_start, j_end)
        result.extend(filled)

        # If this is not the last boundary, add the anchor pair
        if idx < len(anchors):
            ai, aj = anchors[idx]
            anchor_score = float(l1_emb[ai] @ l2_emb[aj])
            result.append(([ai], [aj], anchor_score))

        if on_gap_progress:
            on_gap_progress(idx + 1, len(boundaries))

    return result, anchors


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
    # Phase 1: Load model
    a = log.activity("Loading LaBSE model...", detail=f"({device})") if log else None
    from sentence_transformers import SentenceTransformer
    _silence_hf_logging()
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    if a:
        a.done("LaBSE model loaded")

    # Phase 2: Compute embeddings
    l1_texts = [s.text for s in l1.sentences]
    l2_texts = [s.text for s in l2.sentences]
    a = log.activity("Computing embeddings...") if log else None
    all_emb = _normalize(model.encode(l1_texts + l2_texts, show_progress_bar=False, convert_to_numpy=True))
    l1_emb = all_emb[:len(l1_texts)]
    l2_emb = all_emb[len(l1_texts):]
    if a:
        a.done("Embeddings computed")

    # Phase 3+4: Find anchors + fill gaps (handled inside _two_pass_align)
    p = log.progress("Filling gaps") if log else None
    raw_pairs, _anchors = _two_pass_align(
        l1_emb, l2_emb, l1_texts, l2_texts,
        on_gap_progress=p.update if p else None,
        log=log,
    )
    if p:
        p.finish(f"{len(raw_pairs)} pairs")

    pairs = []
    for src_idxs, tgt_idxs, score in raw_pairs:
        l1_segs = [l1.sentences[i] for i in src_idxs]
        l2_segs = [l2.sentences[i] for i in tgt_idxs]
        pairs.append(AlignmentPair(l1=l1_segs, l2=l2_segs, score=score))

    return Alignment(pairs=pairs)
