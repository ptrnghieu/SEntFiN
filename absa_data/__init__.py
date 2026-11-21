"""Data engineering for aspect/target-based financial sentiment analysis.

Loads SEntFiN 1.0 and FiQA-2018 Task 1, cleans them, maps both onto one
(text, target, polarity) schema, builds leakage-safe splits and exports
training files for encoder and LLM fine-tuning. See README.md.
"""

from .fiqa import load_fiqa, score_to_polarity
from .schema import LABEL2ID, LABELS, PAIR_COLUMNS
from .sentfin import load_sentfin

__all__ = ["LABELS", "LABEL2ID", "PAIR_COLUMNS", "load_fiqa", "load_sentfin", "score_to_polarity"]
