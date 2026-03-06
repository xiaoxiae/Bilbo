from __future__ import annotations

import click
import numpy as np

from .models import Alignment, AlignmentPair, SegmentedText

# Allowed alignment moves: (src_step, tgt_step)
# Covers 1:1, 1:2, 2:1, 1:3, 3:1, 2:2
MOVES = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]


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


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def _block_similarity(
    sim: np.ndarray, i: int, j: int, di: int, dj: int
) -> float:
    """Average similarity for a block of (di, dj) sentences starting at (i, j)."""
    block = sim[i : i + di, j : j + dj]
    return float(block.mean())


def _dp_align(sim: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """DP alignment over the similarity matrix. Returns list of (src_indices, tgt_indices)."""
    n, m = sim.shape
    INF = -1e18

    # cost[i][j] = best total similarity to align src[:i] with tgt[:j]
    cost = np.full((n + 1, m + 1), INF)
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if cost[i][j] == INF:
                continue
            for di, dj in MOVES:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                score = cost[i][j] + _block_similarity(sim, i, j, di, dj)
                if score > cost[ni][nj]:
                    cost[ni][nj] = score
                    back[ni][nj] = (i, j, di, dj)

    # Backtrace
    path = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        if back[ci][cj] is None:
            break
        pi, pj, di, dj = back[ci][cj]
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
) -> Alignment:
    resolved = _resolve_device(device)
    click.echo(f"  Computing sentence embeddings (LaBSE) on {resolved}...")

    l1_texts = [s.text for s in l1.sentences]
    l2_texts = [s.text for s in l2.sentences]

    l1_emb = _embed(l1_texts, device=device)
    l2_emb = _embed(l2_texts, device=device)

    click.echo(f"  Building similarity matrix ({len(l1_texts)}x{len(l2_texts)})...")
    sim = _cosine_similarity_matrix(l1_emb, l2_emb)

    click.echo("  Running DP alignment...")
    raw_pairs = _dp_align(sim)

    pairs = []
    for src_idxs, tgt_idxs in raw_pairs:
        l1_segs = [l1.sentences[i] for i in src_idxs]
        l2_segs = [l2.sentences[i] for i in tgt_idxs]
        pairs.append(AlignmentPair(l1=l1_segs, l2=l2_segs))

    click.echo(f"  Aligned into {len(pairs)} pairs")
    return Alignment(pairs=pairs)
