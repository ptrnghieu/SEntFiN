from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "combined" / "texts.parquet").exists(), reason="run `build` first")
def test_eda_report_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from absa_data.eda import run

    out = run(PROCESSED, tmp_path, None, REPO / "annotations" / "calibration" / "fiqa_thresholds.json")
    text = out.read_text()
    for section in ("## 1. Corpus", "## 5. Entities", "## 8. Aspect", "## 11. Implications"):
        assert section in text
    figs = {p.name for p in (tmp_path / "figures").glob("*.png")}
    assert {"length.png", "sentiment.png", "aspects.png", "soft_labels.png"} <= figs
    for ref in [l.split("(")[1].rstrip(")") for l in text.splitlines() if l.startswith("![")]:
        assert (tmp_path / ref).exists()
