"""Silver aspect labels from an LLM (OpenAI Responses API, strict JSON output).

Each (text, target) pair gets one category of ``taxonomy.LABELS`` plus a
self-reported confidence. Results are appended to a JSONL file, one line per
pair, so an interrupted run resumes where it stopped and nothing paid for is
lost. A label is reused only if it was produced by the same model and the same
prompt version (a hash of the instructions and schema): editing the taxonomy
or the prompt automatically invalidates old labels instead of mixing them.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .taxonomy import CATEGORIES, CONFIDENCE, EXAMPLES, LABELS, RULES, TAXONOMY_VERSION

DEFAULT_MODEL = "gpt-6-luna"


def build_instructions() -> str:
    lines = [
        "You annotate financial news headlines and social-media posts for aspect-based sentiment analysis.",
        "For every (text, target) pair choose exactly one aspect category from the list below: the topic of "
        "the text as it concerns that target. Do not judge sentiment.",
        "",
        "Categories:",
        *[f"- {c.label}: {c.definition}" for c in CATEGORIES],
        "",
        "Rules:",
        *[f"{i}. {r}" for i, r in enumerate(RULES, 1)],
        "",
        "Examples:",
    ]
    for text, labels in EXAMPLES:
        lines.append(f"Text: {json.dumps(text, ensure_ascii=False)}")
        lines += [f"  target {json.dumps(t, ensure_ascii=False)} -> {a}" for t, a in labels.items()]
    lines += [
        "",
        "Input: numbered texts, each followed by its targets. Each target has a key T<text>.<target> "
        "(T3.2 = text 3, target 2). Return exactly one label for every key, in input order. Copy the key "
        "and the target string exactly, and use the category strings exactly as written above.",
    ]
    return "\n".join(lines)


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "target": {"type": "string"},
                        "aspect": {"type": "string", "enum": list(LABELS)},
                        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
                    },
                    "required": ["key", "target", "aspect", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


INSTRUCTIONS = build_instructions()
PROMPT_VERSION = f"{TAXONOMY_VERSION}-" + hashlib.sha1(
    (INSTRUCTIONS + json.dumps(response_schema(), sort_keys=True)).encode()
).hexdigest()[:8]


@dataclass
class Item:
    """One text and the pairs (uid, target) to label in it."""
    text_id: str
    text: str
    text_type: str
    pairs: list[tuple[str, str]]


def make_items(pairs: pd.DataFrame) -> list[Item]:
    items: dict[str, Item] = {}
    for r in pairs[["uid", "text_id", "text", "text_type", "target"]].itertuples(index=False):
        it = items.setdefault(r.text_id, Item(r.text_id, r.text, r.text_type, []))
        it.pairs.append((r.uid, r.target))
    return list(items.values())


def render_batch(items: list[Item]) -> tuple[str, dict[str, tuple[str, str, str]]]:
    """Prompt input for a batch and key -> (uid, text_id, target)."""
    lines, keys = [], {}
    for i, it in enumerate(items, 1):
        lines.append(f"Text {i} ({it.text_type}): {json.dumps(it.text, ensure_ascii=False)}")
        for j, (uid, target) in enumerate(it.pairs, 1):
            key = f"T{i}.{j}"
            keys[key] = (uid, it.text_id, target)
            lines.append(f"  {key}: target {json.dumps(target, ensure_ascii=False)}")
    return "\n".join(lines), keys


def _same_target(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


def parse_output(output_text: str, keys) -> dict[str, tuple[str, str]]:
    """Validate the model's JSON; returns key -> (aspect, confidence). Raises ValueError.

    The echoed target must match the target of its key: this catches the model
    shifting labels by one position inside a batch, which a schema alone cannot.
    """
    data = json.loads(output_text)
    got: dict[str, tuple[str, str]] = {}
    for lab in data["labels"]:
        if lab["aspect"] not in LABELS or lab["confidence"] not in CONFIDENCE:
            raise ValueError(f"invalid label {lab}")
        if lab["key"] in got:
            raise ValueError(f"duplicate key {lab['key']}")
        if lab["key"] in keys and not _same_target(lab["target"], keys[lab["key"]][2]):
            raise ValueError(f"misaligned: key {lab['key']} echoed {lab['target']!r}, expected {keys[lab['key']][2]!r}")
        got[lab["key"]] = (lab["aspect"], lab["confidence"])
    missing = set(keys) - set(got)
    if missing:
        raise ValueError(f"missing keys {sorted(missing)[:5]}")
    return {k: got[k] for k in keys}


def _usage(resp) -> dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    inp = getattr(u, "input_tokens_details", None)
    out = getattr(u, "output_tokens_details", None)
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "cached_input_tokens": (getattr(inp, "cached_tokens", 0) or 0) if inp else 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "reasoning_tokens": (getattr(out, "reasoning_tokens", 0) or 0) if out else 0,
    }


class Annotator:
    def __init__(self, client, model: str = DEFAULT_MODEL, reasoning_effort: str | None = None,
                 max_attempts: int = 3):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts

    def _request(self, prompt: str):
        kwargs = dict(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=prompt,
            store=False,
            text={"format": {"type": "json_schema", "name": "aspect_labels",
                             "schema": response_schema(), "strict": True}},
        )
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return self.client.responses.create(**kwargs)

    def label_batch(self, items: list[Item]) -> tuple[list[dict], dict[str, int], list[tuple[Item, str]]]:
        """Label one batch. Returns (records, token usage, failed items with reason).

        If the output stays invalid after `max_attempts`, the batch is split in
        halves so one problematic text cannot sink its neighbours.
        """
        prompt, keys = render_batch(items)
        usage: dict[str, int] = {}
        error = None
        for _ in range(self.max_attempts):
            resp = self._request(prompt)
            for k, v in _usage(resp).items():
                usage[k] = usage.get(k, 0) + v
            try:
                labels = parse_output(resp.output_text, keys)
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                error = e
                usage["invalid_outputs"] = usage.get("invalid_outputs", 0) + 1
        else:
            if len(items) == 1:
                return [], usage, [(items[0], f"invalid output after {self.max_attempts} attempts: {error}")]
            mid = len(items) // 2
            records, failed = [], []
            for half in (items[:mid], items[mid:]):
                r, u, f = self.label_batch(half)
                records += r
                failed += f
                for k, v in u.items():
                    usage[k] = usage.get(k, 0) + v
            return records, usage, failed
        records = [
            {"uid": uid, "text_id": text_id, "target": target, "aspect": labels[key][0],
             "confidence": labels[key][1], "model": self.model, "prompt_version": PROMPT_VERSION,
             "reasoning_effort": self.reasoning_effort}
            for key, (uid, text_id, target) in keys.items()
        ]
        return records, usage, []

    def run(self, pairs: pd.DataFrame, out_path: str | Path, batch_size: int = 10, workers: int = 4,
            limit_texts: int | None = None, log=print) -> dict:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = done_uids(out_path, self.model, PROMPT_VERSION, self.reasoning_effort)
        items = make_items(pairs[~pairs["uid"].isin(done)])
        if limit_texts is not None:
            items = items[:limit_texts]
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        stats = {"model": self.model, "prompt_version": PROMPT_VERSION, "reasoning_effort": self.reasoning_effort,
                 "already_done": len(done), "batches": len(batches), "pairs_labelled": 0, "pairs_failed": 0,
                 "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                 "invalid_outputs": 0}
        errors: list[str] = []
        t0 = time.time()
        with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.label_batch, b): b for b in batches}
            for n, fut in enumerate(as_completed(futures), 1):
                batch = futures[fut]
                try:
                    records, usage, failed = fut.result()
                except Exception as e:  # API error that survived the SDK's retries
                    stats["pairs_failed"] += sum(len(it.pairs) for it in batch)
                    errors.append(f"{batch[0].text_id}: {type(e).__name__}: {e}")
                    continue
                for it, reason in failed:
                    stats["pairs_failed"] += len(it.pairs)
                    errors.append(f"{it.text_id}: {reason}")
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                stats["pairs_labelled"] += len(records)
                for k, v in usage.items():
                    stats[k] += v
                if n % 20 == 0 or n == len(batches):
                    log(f"  {n}/{len(batches)} batches, {stats['pairs_labelled']} pairs, "
                        f"{stats['input_tokens']:,} in / {stats['output_tokens']:,} out tokens, {time.time() - t0:.0f}s")
        stats["seconds"] = round(time.time() - t0, 1)
        stats["errors"] = errors[:20]
        stats["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with out_path.with_suffix(".usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        return stats


def read_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["uid", "text_id", "target", "aspect", "confidence", "model", "prompt_version"])
    df = pd.read_json(path, lines=True, dtype=False)
    return df


def done_uids(path: str | Path, model: str, prompt_version: str, reasoning_effort: str | None = None) -> set[str]:
    """uids already labelled by this exact configuration (model, prompt, reasoning effort)."""
    df = read_labels(path)
    if df.empty:
        return set()
    effort = df["reasoning_effort"] if "reasoning_effort" in df else pd.Series(None, index=df.index)
    same_effort = effort.isna() if reasoning_effort is None else effort.eq(reasoning_effort)
    return set(df.loc[(df["model"] == model) & (df["prompt_version"] == prompt_version) & same_effort, "uid"])


def latest_labels(path: str | Path) -> pd.DataFrame:
    """One row per uid: the last label written for the current prompt version, if any,
    else the last label of any version (so stale labels are visible but marked)."""
    df = read_labels(path)
    if df.empty:
        return df
    df = df.assign(_current=df["prompt_version"] == PROMPT_VERSION, _order=range(len(df)))
    df = df.sort_values(["_current", "_order"]).drop_duplicates("uid", keep="last")
    return df.drop(columns=["_current", "_order"]).sort_values("uid").reset_index(drop=True)


def make_client():
    import os

    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (export it, or put it in your environment's secrets).")
    return OpenAI(max_retries=6, timeout=180)
