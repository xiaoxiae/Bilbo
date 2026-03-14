# Alignment algorithm ported from Bertalign (https://github.com/bfsujason/bertalign)
# Copyright (C) 2021 Jason Li, licensed under GPL-3.0
#
# Original uses numba JIT, FAISS, and torch — this port uses pure numpy.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .models import Alignment, AlignmentPair, SegmentedText

if TYPE_CHECKING:
    from .log import PipelineLog


def _normalize(emb: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows in-place and return."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    emb /= norms
    return emb


# ── Overlap encoding (ported from bertalign/encoder.py + utils.py) ────


def _preprocess_line(line: str) -> str:
    line = line.strip()
    if len(line) == 0:
        line = "BLANK_LINE"
    return line


def _yield_overlaps(sents: list[str], num_overlaps: int):
    """Generate combined sentence strings for each overlap layer.

    For overlap=1, yields individual sentences.
    For overlap=2, yields pairs "sent_i sent_{i+1}", padded at the start.
    etc.

    Ported from bertalign/utils.py: yield_overlaps, _layer, _preprocess_line.
    """
    lines = [_preprocess_line(line) for line in sents]
    for overlap in range(1, num_overlaps + 1):
        for out_line in _layer(lines, overlap):
            yield out_line[:10000]


def _layer(lines: list[str], num_overlaps: int, comb: str = " ") -> list[str]:
    if num_overlaps < 1:
        raise ValueError("num_overlaps must be >= 1")
    out = ["PAD"] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii : ii + num_overlaps]))
    return out


def _encode_overlaps(
    model, sents: list[str], num_overlaps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Encode sentences with overlap layers, returning (vecs, lens).

    vecs: shape (num_overlaps, num_sents, embed_dim)
    lens: shape (num_overlaps, num_sents) — byte lengths

    Ported from bertalign/encoder.py: Encoder.transform.
    """
    overlaps = list(_yield_overlaps(sents, num_overlaps))

    sent_vecs = model.encode(overlaps, show_progress_bar=False, convert_to_numpy=True)
    embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
    sent_vecs = sent_vecs.reshape(num_overlaps, len(sents), embedding_dim)

    len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps])
    len_vecs = len_vecs.reshape(num_overlaps, len(sents))

    return sent_vecs, len_vecs


# ── Top-k search (replaces FAISS) ────────────────────────────────────


def _find_top_k(
    src_vecs: np.ndarray, tgt_vecs: np.ndarray, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Find top-k most similar target vectors for each source vector.

    Returns (D, I) matching FAISS IndexFlatIP output format:
      D: (num_src, k) similarity scores
      I: (num_src, k) target indices

    Ported from bertalign/corelib.py: find_top_k_sents — replaces FAISS.
    """
    sims = src_vecs @ tgt_vecs.T  # (num_src, num_tgt)
    k = min(k, sims.shape[1])
    # argpartition is O(n) vs O(n log n) for argsort
    top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    # Gather scores and sort within top-k
    rows = np.arange(sims.shape[0])[:, None]
    top_k_scores = sims[rows, top_k_idx]
    sorted_order = np.argsort(-top_k_scores, axis=1)
    I = np.take_along_axis(top_k_idx, sorted_order, axis=1)
    D = np.take_along_axis(top_k_scores, sorted_order, axis=1)
    return D, I


# ── Alignment types ──────────────────────────────────────────────────


def _get_alignment_types(max_align: int) -> np.ndarray:
    """Get all possible alignment types where src+tgt <= max_align.

    Ported from bertalign/corelib.py: get_alignment_types.
    """
    alignment_types = [[0, 1], [1, 0]]
    for x in range(1, max_align):
        for y in range(1, max_align):
            if x + y <= max_align:
                alignment_types.append([x, y])
    return np.array(alignment_types)


# ── First pass: 1-1 anchor finding via diagonal-band DP ──────────────


def _find_first_search_path(
    src_len: int,
    tgt_len: int,
    min_win_size: int = 250,
    percent: float = 0.06,
) -> tuple[int, np.ndarray]:
    """Find window size and search path for first-pass alignment.

    Ported from bertalign/corelib.py: find_first_search_path.
    """
    win_size = max(min_win_size, int(max(src_len, tgt_len) * percent))
    search_path = []
    yx_ratio = tgt_len / src_len
    for i in range(0, src_len + 1):
        center = int(yx_ratio * i)
        win_start = max(0, center - win_size)
        win_end = min(center + win_size, tgt_len)
        search_path.append([win_start, win_end])
    return win_size, np.array(search_path)


def _first_pass_align(
    src_len: int,
    w: int,
    search_path: np.ndarray,
    align_types: np.ndarray,
    dist: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """First-pass DP to extract 1-1 anchor pairs.

    Ported from bertalign/corelib.py: first_pass_align (without @nb.jit).
    """
    cost = np.zeros((src_len + 1, 2 * w + 1), dtype=np.float32)
    pointers = np.zeros((src_len + 1, 2 * w + 1), dtype=np.uint8)

    top_k = index.shape[1]

    for i in range(src_len + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]
        for j in range(i_start, i_end + 1):
            if i + j == 0:
                continue
            best_score = -np.inf
            best_a = -1
            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2
                if prev_i < 0 or prev_j < 0:
                    continue
                prev_i_start = search_path[prev_i][0]
                prev_i_end = search_path[prev_i][1]
                if prev_j < prev_i_start or prev_j > prev_i_end:
                    continue
                prev_j_offset = prev_j - prev_i_start
                score = cost[prev_i][prev_j_offset]

                # Extract score for 1-1 bead from top-k
                if a_1 > 0 and a_2 > 0:
                    for k in range(top_k):
                        if index[i - 1][k] == j - 1:
                            score += dist[i - 1][k]
                if score > best_score:
                    best_score = score
                    best_a = a

            j_offset = j - i_start
            cost[i][j_offset] = best_score
            pointers[i][j_offset] = best_a

    return pointers


def _first_back_track(
    i: int,
    j: int,
    pointers: np.ndarray,
    search_path: np.ndarray,
    a_types: np.ndarray,
) -> list[tuple[int, int]]:
    """Retrieve 1-1 alignments from first-pass DP table.

    Ported from bertalign/corelib.py: first_back_track.
    """
    alignment = []
    while True:
        j_offset = j - search_path[i][0]
        a = pointers[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        if a == 2:  # best 1-1 alignment
            alignment.append((i, j))

        i = i - s
        j = j - t

        if i == 0 and j == 0:
            return alignment[::-1]


# ── Second pass: m:n alignment with margin scoring ───────────────────


def _find_second_search_path(
    align: list[tuple[int, int]],
    w: int,
    src_len: int,
    tgt_len: int,
) -> tuple[int, np.ndarray]:
    """Convert 1-1 first-pass alignment to second-pass search path.

    Ported from bertalign/corelib.py: find_second_search_path.
    """
    # Adjust so last bead is (src_len, tgt_len)
    last_bead_src = align[-1][0]
    last_bead_tgt = align[-1][1]
    if last_bead_src != src_len:
        if last_bead_tgt == tgt_len:
            align.pop()
        align.append((src_len, tgt_len))
    else:
        if last_bead_tgt != tgt_len:
            align.pop()
            align.append((src_len, tgt_len))

    prev_src, prev_tgt = 0, 0
    path = []
    max_w = -np.inf
    for src, tgt in align:
        lower_bound = max(0, prev_tgt - w)
        upper_bound = min(tgt_len, tgt + w)
        path.extend(
            [(lower_bound, upper_bound) for _ in range(prev_src + 1, src + 1)]
        )
        prev_src, prev_tgt = src, tgt
        width = upper_bound - lower_bound
        if width > max_w:
            max_w = width
    path = [path[0]] + path  # add search path for row 0
    return int(max_w) + 1, np.array(path)


def _second_pass_align(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    src_lens: np.ndarray,
    tgt_lens: np.ndarray,
    w: int,
    search_path: np.ndarray,
    align_types: np.ndarray,
    char_ratio: float,
    skip: float,
    margin: bool = False,
    len_penalty: bool = False,
) -> np.ndarray:
    """Second-pass DP for m:n alignment with margin scoring.

    Ported from bertalign/corelib.py: second_pass_align (without @nb.jit).
    """
    src_len = src_vecs.shape[1]
    tgt_len = tgt_vecs.shape[1]
    cost = np.zeros((src_len + 1, w), dtype=np.float32)
    pointers = np.zeros((src_len + 1, w), dtype=np.uint8)

    for i in range(src_len + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]
        for j in range(i_start, i_end + 1):
            if i + j == 0:
                continue
            best_score = -np.inf
            best_a = -1
            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2

                if prev_i < 0 or prev_j < 0:
                    continue
                prev_i_start = search_path[prev_i][0]
                prev_i_end = search_path[prev_i][1]
                if prev_j < prev_i_start or prev_j > prev_i_end:
                    continue
                prev_j_offset = prev_j - prev_i_start
                score = cost[prev_i][prev_j_offset]

                if a_1 == 0 or a_2 == 0:  # deletion or insertion
                    cur_score = skip
                else:
                    cur_score = _calculate_similarity_score(
                        src_vecs, tgt_vecs,
                        i, j, a_1, a_2,
                        src_len, tgt_len,
                        margin=margin,
                    )
                    if len_penalty:
                        penalty = _calculate_length_penalty(
                            src_lens, tgt_lens, i, j, a_1, a_2, char_ratio
                        )
                        cur_score *= penalty

                score += cur_score
                if score > best_score:
                    best_score = score
                    best_a = a

            j_offset = j - i_start
            cost[i][j_offset] = best_score
            pointers[i][j_offset] = best_a

    return pointers


def _second_back_track(
    i: int,
    j: int,
    pointers: np.ndarray,
    search_path: np.ndarray,
    a_types: np.ndarray,
) -> list[tuple[list[int], list[int]]]:
    """Retrieve m:n alignments from second-pass DP table.

    Ported from bertalign/corelib.py: second_back_track.
    """
    alignment = []
    while True:
        j_offset = j - search_path[i][0]
        a = pointers[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        src_range = [i - offset - 1 for offset in range(s)][::-1]
        tgt_range = [j - offset - 1 for offset in range(t)][::-1]
        alignment.append((src_range, tgt_range))

        i = i - s
        j = j - t

        if i == 0 and j == 0:
            return alignment[::-1]


# ── Scoring helpers ──────────────────────────────────────────────────


def _calculate_similarity_score(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    src_idx: int,
    tgt_idx: int,
    src_overlap: int,
    tgt_overlap: int,
    src_len: int,
    tgt_len: int,
    margin: bool = False,
) -> float:
    """Semantics-based similarity score for a bitext segment.

    Ported from bertalign/corelib.py: calculate_similarity_score (without @nb.jit).
    """
    src_v = src_vecs[src_overlap - 1, src_idx - 1, :]
    tgt_v = tgt_vecs[tgt_overlap - 1, tgt_idx - 1, :]
    similarity = float(np.dot(src_v, tgt_v))
    if margin:
        tgt_neighbor_ave_sim = _calculate_neighbor_similarity(
            src_v, tgt_overlap, tgt_idx, tgt_len, tgt_vecs
        )
        src_neighbor_ave_sim = _calculate_neighbor_similarity(
            tgt_v, src_overlap, src_idx, src_len, src_vecs
        )
        neighbor_ave_sim = (tgt_neighbor_ave_sim + src_neighbor_ave_sim) / 2
        similarity -= neighbor_ave_sim

    return similarity


def _calculate_neighbor_similarity(
    vec: np.ndarray,
    overlap: int,
    sent_idx: int,
    sent_len: int,
    db: np.ndarray,
) -> float:
    """Average similarity to neighboring sentences.

    Ported from bertalign/corelib.py: calculate_neighbor_similarity (without @nb.jit).
    """
    left_idx = sent_idx - overlap
    right_idx = sent_idx + 1

    if right_idx <= sent_len:
        right_embed = db[0, right_idx - 1, :]
        neighbor_right_sim = float(np.dot(vec, right_embed))
    else:
        neighbor_right_sim = 0.0

    if left_idx > 0:
        left_embed = db[0, left_idx - 1, :]
        neighbor_left_sim = float(np.dot(vec, left_embed))
    else:
        neighbor_left_sim = 0.0

    neighbor_ave_sim = neighbor_left_sim + neighbor_right_sim
    if neighbor_right_sim and neighbor_left_sim:
        neighbor_ave_sim /= 2

    return neighbor_ave_sim


def _calculate_length_penalty(
    src_lens: np.ndarray,
    tgt_lens: np.ndarray,
    src_idx: int,
    tgt_idx: int,
    src_overlap: int,
    tgt_overlap: int,
    char_ratio: float,
) -> float:
    """Length-based penalty for bitext segment.

    Ported from bertalign/corelib.py: calculate_length_penalty (without @nb.jit).
    """
    src_l = src_lens[src_overlap - 1, src_idx - 1]
    tgt_l = tgt_lens[tgt_overlap - 1, tgt_idx - 1]
    tgt_l = tgt_l * char_ratio
    min_len = min(src_l, tgt_l)
    max_len = max(src_l, tgt_l)
    return float(np.log2(1 + min_len / max_len))


# ── Two-pass orchestration ───────────────────────────────────────────


def _two_pass_align(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    src_lens: np.ndarray,
    tgt_lens: np.ndarray,
    char_ratio: float,
    max_align: int = 5,
    top_k: int = 3,
    win: int = 5,
    skip: float = -0.1,
    log: PipelineLog | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Two-pass Bertalign alignment using overlap embeddings.

    Args:
        src_vecs: (num_overlaps, num_sents, embed_dim)
        tgt_vecs: (num_overlaps, num_sents, embed_dim)
        src_lens: (num_overlaps, num_sents) byte lengths
        tgt_lens: (num_overlaps, num_sents) byte lengths
        char_ratio: src/tgt byte length ratio
        max_align: max src+tgt sentences per alignment type
        top_k: number of top-k similar sentences for first pass
        win: window size for second pass
        skip: cost for insertion/deletion
        log: pipeline logger

    Returns:
        List of (src_indices, tgt_indices) pairs.
    """
    src_len = src_vecs.shape[1]
    tgt_len = tgt_vecs.shape[1]

    # First pass: find 1-1 anchors via diagonal-band DP
    a = log.activity("Finding anchors (first pass)...") if log else None
    D, I = _find_top_k(src_vecs[0, :], tgt_vecs[0, :], k=top_k)
    first_at = _get_alignment_types(2)
    first_w, first_path = _find_first_search_path(src_len, tgt_len)
    first_ptrs = _first_pass_align(
        src_len, first_w, first_path, first_at, D, I
    )
    first_align = _first_back_track(
        src_len, tgt_len, first_ptrs, first_path, first_at
    )
    if a:
        a.done(f"{len(first_align)} anchors")

    # Second pass: m:n alignment with margin scoring
    a = log.activity("Aligning (second pass)...") if log else None
    second_at = _get_alignment_types(max_align)
    second_w, second_path = _find_second_search_path(
        first_align, win, src_len, tgt_len
    )
    second_ptrs = _second_pass_align(
        src_vecs, tgt_vecs, src_lens, tgt_lens,
        second_w, second_path, second_at,
        char_ratio, skip, margin=True, len_penalty=True,
    )
    result = _second_back_track(
        src_len, tgt_len, second_ptrs, second_path, second_at
    )
    if a:
        a.done(f"{len(result)} pairs")

    return result


# ── HuggingFace logging suppression ─────────────────────────────────


def _silence_hf_logging() -> None:
    """Suppress noisy output from the HuggingFace / transformers stack."""
    import logging

    import huggingface_hub.utils.logging as hf_logging
    import transformers.utils.logging as tf_logging

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()


# ── Public API ───────────────────────────────────────────────────────


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

    # Phase 2: Encode with overlaps (4 layers for max_align=5)
    l1_texts = [s.text for s in l1.sentences]
    l2_texts = [s.text for s in l2.sentences]
    num_overlaps = 4  # max_align - 1
    a = log.activity("Computing embeddings...") if log else None
    src_vecs, src_lens = _encode_overlaps(model, l1_texts, num_overlaps)
    tgt_vecs, tgt_lens = _encode_overlaps(model, l2_texts, num_overlaps)
    if a:
        a.done("Embeddings computed")

    # Phase 3: Compute char_ratio from byte lengths
    char_ratio = float(np.sum(src_lens[0,]) / np.sum(tgt_lens[0,]))

    # Phase 4: Two-pass alignment
    raw_pairs = _two_pass_align(
        src_vecs, tgt_vecs, src_lens, tgt_lens,
        char_ratio,
        log=log,
    )

    # Phase 5: Build AlignmentPair objects, filtering out skip pairs
    # Compute per-pair scores from layer-0 cosine similarity
    src_norm = _normalize(src_vecs[0].copy())
    tgt_norm = _normalize(tgt_vecs[0].copy())

    pairs = []
    covered_l1: set[int] = set()
    covered_l2: set[int] = set()
    for src_idxs, tgt_idxs in raw_pairs:
        if not src_idxs or not tgt_idxs:
            continue  # skip insertion/deletion pairs
        covered_l1.update(src_idxs)
        covered_l2.update(tgt_idxs)
        l1_segs = [l1.sentences[i] for i in src_idxs]
        l2_segs = [l2.sentences[i] for i in tgt_idxs]
        block = src_norm[src_idxs] @ tgt_norm[tgt_idxs].T
        score = float(block.mean())
        pairs.append(AlignmentPair(l1=l1_segs, l2=l2_segs, score=score))

    # Log skipped sentences (insertions/deletions with no alignment match)
    if log:
        skipped_l1 = sorted(set(range(len(l1_texts))) - covered_l1)
        skipped_l2 = sorted(set(range(len(l2_texts))) - covered_l2)
        if skipped_l1:
            log.warn(f"{len(skipped_l1)} L1 sentences skipped (no alignment match)")
            for i in skipped_l1[:5]:
                log.detail(f"L1[{i}]: {l1_texts[i][:80]}")
            if len(skipped_l1) > 5:
                log.detail(f"... and {len(skipped_l1) - 5} more")
        if skipped_l2:
            log.warn(f"{len(skipped_l2)} L2 sentences skipped (no alignment match)")
            for i in skipped_l2[:5]:
                log.detail(f"L2[{i}]: {l2_texts[i][:80]}")
            if len(skipped_l2) > 5:
                log.detail(f"... and {len(skipped_l2) - 5} more")

    return Alignment(pairs=pairs)
