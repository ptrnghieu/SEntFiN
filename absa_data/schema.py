"""Unified schema shared by every source.

One row of the *pairs* table is one (text, target) annotation. Columns that a
source does not provide are left empty (NaN / None), e.g. ``score`` and
``aspect`` only exist for FiQA.
"""

LABELS = ("negative", "neutral", "positive")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SPLITS = ("train", "valid", "test")

PAIR_COLUMNS = [
    "uid",              # stable id of the pair: <text_id>#<k>
    "source",           # "sentfin" | "fiqa"
    "text_id",          # stable id of the text (several pairs can share it)
    "split",            # train | valid | test
    "text_type",        # "headline" | "post"
    "text",             # cleaned text used for training
    "text_raw",         # text exactly as distributed by the source
    "target",           # cleaned target surface form
    "target_raw",       # target exactly as distributed by the source
    "start",            # char offset of target in `text` (None if not found)
    "end",              # exclusive end offset (None if not found)
    "span_match",       # exact | case_insensitive | substring | fuzzy | not_found
    "polarity",         # negative | neutral | positive (FiQA: score cut at the thresholds used by the build)
    "label",            # LABEL2ID[polarity]
    "score",            # FiQA continuous score in [-1, 1]
    "polarity_finben",  # FiQA only: class under FinBen's thresholds (±0.1), for comparison with published results
    "label_type",       # "hard" (SEntFiN: annotated class) | "soft" (FiQA: derived from a noisy score)
    "p_negative",       # soft label: P(negative); see calibration.py for the noise models
    "p_neutral",        # soft label: P(neutral)
    "p_positive",       # soft label: P(positive)
    "label_confidence",  # max of the three probabilities
    "aspect",           # unified aspect category (taxonomy.py), e.g. Corporate/Financial Results
    "aspect_l1",        # its first level: Corporate | Stock | Market | Economy | Other
    "aspect_source",    # "human" | "llm:<model>@<prompt version>" | None (not labelled yet)
    "aspect_confidence",  # LLM self-reported confidence: high | medium | low
    "fiqa_aspect",      # FiQA's original gold aspect path, normalized (FiQA only)
    "fiqa_aspect_l1",   # its first level (Stock, Corporate, Market, Economy)
    "fiqa_aspect_l2",   # its first two levels (Stock/Price Action, ...)
    "entity_id",        # canonical entity key used for seen/unseen analysis
    "n_targets",        # number of annotated targets in this text
    "is_mixed",         # text has targets with different polarities
    "group_id",         # near-duplicate cluster; never split across train/valid/test
    "entity_seen_in_train",  # entity_id occurs in this source's train split
    "quality_flags",    # ';'-separated data-quality flags (see README)
]
