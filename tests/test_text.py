from absa_data.text import align_targets, clean_text, dedup_key


def spans(text, targets, **kw):
    return [(text[s.start:s.end] if s.found else None, s.match) for s in align_targets(text, targets, **kw)]


def test_clean_text_repairs_mojibake_and_whitespace():
    assert clean_text("Intertek swings to Â£347 mln  loss") == "Intertek swings to £347 mln loss"
    assert clean_text("Deutsche BÃ¶rse") == "Deutsche Börse"
    assert clean_text("  plain text  ") == "plain text"


def test_dedup_key_ignores_urls_case_and_punctuation():
    a = "Costco: A Premier Retail Dividend Play https://t.co/Fa5cnh2t0t $COST"
    b = "Costco: A Premier Retail Dividend Play https://t.co/J3UhTs022M $COST"
    assert dedup_key(a) == dedup_key(b)
    assert dedup_key("Infosys' Q2 results") == dedup_key("Infosys Q2 results")


def test_word_boundaries_prevent_matches_inside_words():
    # Naive substring search would find "IT" inside "Nitin".
    text = "Stock-specific action likely on IT counters: Nitin Raheja"
    assert spans(text, ["IT"]) == [("IT", "exact")]
    s = align_targets(text, ["IT"])[0]
    assert text[s.start - 3:s.start] == "on "


def test_longest_first_avoids_nested_assignment():
    text = "Add IT, pharma names to portfolio; Mastek, Aurobindo Pharma top bets"
    got = align_targets(text, ["pharma", "Aurobindo Pharma"])
    assert got[0].start == text.index("pharma names")
    assert text[got[1].start:got[1].end] == "Aurobindo Pharma"


def test_jet_does_not_land_inside_spicejet():
    text = "Jet, SpiceJet shares soar on Etihad deal"
    got = align_targets(text, ["Jet", "SpiceJet"])
    assert (got[0].start, got[1].start) == (0, 5)


def test_case_insensitive_and_cashtag():
    assert spans("$aapl looking strong", ["AAPL"]) == [("aapl", "case_insensitive")]
    assert spans("$SPY c'mon bulls buy DIPS!", ["SPY"]) == [("SPY", "exact")]


def test_fuzzy_fixes_typos_but_not_short_or_absent_targets():
    assert spans("UK housing market steadies, Persimmon says", ["Perssimon"]) == [("Persimmon", "fuzzy")]
    assert spans("AB InBev offers SABMiller $3 billion breakup fee", ["SAB Miller"]) == [("SABMiller", "fuzzy")]
    assert spans("StanChart and RBS struggle in stress tests", ["Standard Chartered"]) == [(None, "not_found")]
    assert spans("$XLI potential intermediate top", ["CRM"]) == [(None, "not_found")]
    assert spans("UK housing market steadies, Persimmon says", ["Perssimon"], fuzzy=False) == [(None, "not_found")]
