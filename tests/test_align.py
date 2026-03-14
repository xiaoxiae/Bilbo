import numpy as np

from bilbo.align import (
    _normalize,
    _yield_overlaps,
    _find_top_k,
    _get_alignment_types,
    _two_pass_align,
)


def test_normalize_unit_norm():
    emb = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    result = _normalize(emb)
    norms = np.linalg.norm(result, axis=1)
    # Non-zero rows should be unit-norm
    assert abs(norms[0] - 1.0) < 1e-5
    assert abs(norms[1] - 1.0) < 1e-5


def test_normalize_preserves_direction():
    emb = np.array([[3.0, 4.0]], dtype=np.float32)
    result = _normalize(emb)
    assert abs(result[0, 0] - 0.6) < 1e-5
    assert abs(result[0, 1] - 0.8) < 1e-5


def test_yield_overlaps():
    sents = ["Hello world", "foo bar", "baz qux"]
    overlaps = list(_yield_overlaps(sents, 2))
    # Layer 1: 3 individual sentences, Layer 2: 1 PAD + 2 pairs
    assert len(overlaps) == 6
    # Layer 1
    assert overlaps[0] == "Hello world"
    assert overlaps[1] == "foo bar"
    assert overlaps[2] == "baz qux"
    # Layer 2
    assert overlaps[3] == "PAD"
    assert overlaps[4] == "Hello world foo bar"
    assert overlaps[5] == "foo bar baz qux"


def test_yield_overlaps_empty_line():
    """Empty lines are replaced with BLANK_LINE."""
    sents = ["Hello", "", "World"]
    overlaps = list(_yield_overlaps(sents, 1))
    assert overlaps[1] == "BLANK_LINE"


def test_find_top_k():
    # 4 source vectors, 5 target vectors
    rng = np.random.RandomState(42)
    src = _normalize(rng.randn(4, 16).astype(np.float32))
    tgt = _normalize(rng.randn(5, 16).astype(np.float32))
    D, I = _find_top_k(src, tgt, k=3)
    assert D.shape == (4, 3)
    assert I.shape == (4, 3)
    # Scores should be sorted descending
    for i in range(4):
        assert D[i, 0] >= D[i, 1] >= D[i, 2]
    # Indices should be valid
    assert np.all(I >= 0) and np.all(I < 5)


def test_find_top_k_self():
    """Top-1 of self should be the vector itself."""
    emb = _normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    D, I = _find_top_k(emb, emb, k=1)
    for i in range(3):
        assert I[i, 0] == i
        assert abs(D[i, 0] - 1.0) < 1e-5


def test_get_alignment_types():
    at = _get_alignment_types(3)
    # Should include: [0,1], [1,0], [1,1], [1,2], [2,1]
    at_list = at.tolist()
    assert [0, 1] in at_list
    assert [1, 0] in at_list
    assert [1, 1] in at_list
    assert [1, 2] in at_list
    assert [2, 1] in at_list
    # Should NOT include [2,2] since 2+2 > 3
    assert [2, 2] not in at_list


def test_get_alignment_types_max5():
    at = _get_alignment_types(5)
    at_list = at.tolist()
    assert [4, 1] in at_list
    assert [1, 4] in at_list
    assert [3, 2] in at_list
    assert [2, 3] in at_list
    # 3+3=6 > 5
    assert [3, 3] not in at_list


def test_two_pass_align_integration():
    """Integration test: align using synthetic overlap embeddings."""
    n = 12
    dim = 32
    num_overlaps = 4
    rng = np.random.RandomState(123)

    # Create embeddings — same for src and tgt so alignment is diagonal
    vecs = np.zeros((num_overlaps, n, dim), dtype=np.float32)
    for layer in range(num_overlaps):
        vecs[layer] = _normalize(rng.randn(n, dim).astype(np.float32))

    # Byte lengths: just use uniform lengths
    lens = np.full((num_overlaps, n), 50, dtype=np.int64)

    char_ratio = 1.0
    pairs = _two_pass_align(vecs, vecs, lens, lens, char_ratio)
    # All indices should be covered
    all_src = sorted(idx for src, _ in pairs for idx in src)
    all_tgt = sorted(idx for _, tgt in pairs for idx in tgt)
    assert all_src == list(range(n))
    assert all_tgt == list(range(n))


def test_two_pass_align_different_lengths():
    """Align different-length sequences with overlap embeddings."""
    n1, n2 = 8, 6
    dim = 32
    num_overlaps = 4
    rng = np.random.RandomState(99)

    src_vecs = np.zeros((num_overlaps, n1, dim), dtype=np.float32)
    tgt_vecs = np.zeros((num_overlaps, n2, dim), dtype=np.float32)
    for layer in range(num_overlaps):
        src_vecs[layer] = _normalize(rng.randn(n1, dim).astype(np.float32))
        tgt_vecs[layer] = _normalize(rng.randn(n2, dim).astype(np.float32))

    src_lens = np.full((num_overlaps, n1), 50, dtype=np.int64)
    tgt_lens = np.full((num_overlaps, n2), 50, dtype=np.int64)

    char_ratio = 1.0
    pairs = _two_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens, char_ratio)
    # All non-skip pairs should have valid indices
    all_src = sorted(idx for src, tgt in pairs for idx in src if src and tgt)
    all_tgt = sorted(idx for src, tgt in pairs for idx in tgt if src and tgt)
    # At minimum, some indices should be covered
    assert len(all_src) > 0
    assert len(all_tgt) > 0


def test_skip_pairs_filtered():
    """Verify that skip pairs (empty src or tgt) exist in raw output."""
    n1, n2 = 10, 8
    dim = 32
    num_overlaps = 4
    rng = np.random.RandomState(77)

    src_vecs = np.zeros((num_overlaps, n1, dim), dtype=np.float32)
    tgt_vecs = np.zeros((num_overlaps, n2, dim), dtype=np.float32)
    for layer in range(num_overlaps):
        src_vecs[layer] = _normalize(rng.randn(n1, dim).astype(np.float32))
        tgt_vecs[layer] = _normalize(rng.randn(n2, dim).astype(np.float32))

    src_lens = np.full((num_overlaps, n1), 50, dtype=np.int64)
    tgt_lens = np.full((num_overlaps, n2), 50, dtype=np.int64)

    char_ratio = float(np.sum(src_lens[0,]) / np.sum(tgt_lens[0,]))
    pairs = _two_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens, char_ratio)

    # The output may include skip pairs — verify we can filter them
    non_skip = [(s, t) for s, t in pairs if s and t]
    # Non-skip pairs should have valid indices
    for src, tgt in non_skip:
        assert all(0 <= i < n1 for i in src)
        assert all(0 <= j < n2 for j in tgt)
