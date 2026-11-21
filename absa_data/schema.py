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
    "polarity",         # negative | neutral | positive
    "label",            # LABEL2ID[polarity]
    "score",            # FiQA continuous score in [-1, 1]
    "aspect",           # FiQA aspect path, normalized (e.g. Stock/Price Action/Bullish)
    "aspect_l1",        # first level of the aspect path (Stock, Corporate, ...)
    "aspect_l2",        # first two levels (Stock/Price Action, ...)
    "entity_id",        # canonical entity key used for seen/unseen analysis
    "n_targets",        # number of annotated targets in this text
    "is_mixed",         # text has targets with different polarities
    "group_id",         # near-duplicate cluster; never split across train/valid/test
    "entity_seen_in_train",  # entity_id occurs in this source's train split
    "quality_flags",    # ';'-separated data-quality flags (see README)
]
