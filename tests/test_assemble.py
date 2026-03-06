from bilbo.models import AlignmentPair, ExportConfig, Segment, Word


def _make_pair(l1_start, l1_end, l1_text, l1_words, l2_start, l2_end, l2_text, l2_words):
    return AlignmentPair(
        l1=[Segment(l1_start, l1_end, l1_text, words=l1_words)],
        l2=[Segment(l2_start, l2_end, l2_text, words=l2_words)],
    )


def test_export_config_defaults():
    c = ExportConfig()
    assert c.intra_gap_ms == 300
    assert c.inter_gap_ms == 600
    assert c.format == "m4b"
    assert c.order == "l1-first"
