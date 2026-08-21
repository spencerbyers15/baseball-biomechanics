from idle_mind.foreground import parse_surfaced_marker

VALID = {"T1", "T2", "Q1", "R1"}


def test_marker_stripped_and_parsed():
    text = "I was thinking about tides while you were gone.\n⟦surfaced: T1, Q1⟧"
    result = parse_surfaced_marker(text, VALID)
    assert result.display_text == "I was thinking about tides while you were gone."
    assert result.surfaced_ids == ["T1", "Q1"]
    assert "⟦" not in result.display_text


def test_marker_none():
    result = parse_surfaced_marker("Plain reply.\n⟦surfaced: none⟧", VALID)
    assert result.display_text == "Plain reply."
    assert result.surfaced_ids == []


def test_marker_missing_entirely():
    result = parse_surfaced_marker("Reply without bookkeeping.", VALID)
    assert result.surfaced_ids is None
    assert result.raw_marker is None
    assert result.display_text == "Reply without bookkeeping."


def test_marker_invalid_ids_filtered():
    result = parse_surfaced_marker("Reply.\n⟦surfaced: T1, T9, X2⟧", VALID)
    assert result.surfaced_ids == ["T1"]
    assert result.raw_marker == "T1, T9, X2"


def test_marker_case_and_spacing_tolerated():
    result = parse_surfaced_marker("Reply.\n⟦Surfaced:  t2 , q1 ⟧", VALID)
    assert result.surfaced_ids == ["T2", "Q1"]


def test_multiple_markers_all_stripped_last_wins():
    text = "A ⟦surfaced: T1⟧ middle.\nEnd.\n⟦surfaced: Q1⟧"
    result = parse_surfaced_marker(text, VALID)
    assert "⟦" not in result.display_text
    assert result.surfaced_ids == ["Q1"]
