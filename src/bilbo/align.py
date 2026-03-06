from __future__ import annotations

import click
import numpy as np

from .models import Alignment, AlignmentPair, SegmentedText

# Allowed alignment moves: (src_step, tgt_step)
# Covers 1:1, 1:2, 2:1, 1:3, 3:1, 2:2
MOVES = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]

DEFAULT_PADDING = 50


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embed(texts: list[str], device: str = "cpu") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = _resolve_device(device)
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


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


def _dp_align(
    l1_emb: np.ndarray, l2_emb: np.ndarray, padding: int
) -> list[tuple[list[int], list[int]]]:
    """Frontier-tracking banded DP alignment. Returns list of (src_indices, tgt_indices)."""
    n = len(l1_emb)
    m = len(l2_emb)
    INF = -1e18

    cost: dict[tuple[int, int], float] = {(0, 0): 0.0}
    back: dict[tuple[int, int], tuple[int, int]] = {}
    # Per-row reachable j range: row -> (j_lo, j_hi) inclusive
    reached: dict[int, tuple[int, int]] = {0: (0, 0)}

    for i in range(n + 1):
        if i not in reached:
            continue
        r_lo, r_hi = reached[i]
        j_lo = max(0, r_lo - padding)
        j_hi = min(m, r_hi + padding)

        for j in range(j_lo, j_hi + 1):
            c = cost.get((i, j), INF)
            if c == INF:
                continue
            for di, dj in MOVES:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                score = c + _block_similarity(l1_emb, l2_emb, i, j, di, dj)
                if score > cost.get((ni, nj), INF):
                    cost[(ni, nj)] = score
                    back[(ni, nj)] = (di, dj)
                    # Expand reached range for target row
                    if ni in reached:
                        old_lo, old_hi = reached[ni]
                        reached[ni] = (min(old_lo, nj), max(old_hi, nj))
                    else:
                        reached[ni] = (nj, nj)

    # Backtrace
    path = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        if (ci, cj) not in back:
            break
        di, dj = back[(ci, cj)]
        pi, pj = ci - di, cj - dj
        src_idxs = list(range(pi, pi + di))
        tgt_idxs = list(range(pj, pj + dj))
        path.append((src_idxs, tgt_idxs))
        ci, cj = pi, pj

    path.reverse()
    return path


def align_texts(
    l1: SegmentedText,
    l2: SegmentedText,
    device: str = "cpu",
    padding: int = DEFAULT_PADDING,
) -> Alignment:
    resolved = _resolve_device(device)
    click.echo(f"  Computing sentence embeddings (LaBSE) on {resolved}...")

    l1_texts = [s.text for s in l1.sentences]
    l2_texts = [s.text for s in l2.sentences]

    l1_emb = _normalize(_embed(l1_texts, device=device))
    l2_emb = _normalize(_embed(l2_texts, device=device))

    click.echo(f"  Running banded DP alignment (padding={padding})...")
    raw_pairs = _dp_align(l1_emb, l2_emb, padding)

    pairs = []
    for src_idxs, tgt_idxs in raw_pairs:
        l1_segs = [l1.sentences[i] for i in src_idxs]
        l2_segs = [l2.sentences[i] for i in tgt_idxs]
        pairs.append(AlignmentPair(l1=l1_segs, l2=l2_segs))

    click.echo(f"  Aligned into {len(pairs)} pairs")
    return Alignment(pairs=pairs)
