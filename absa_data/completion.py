"""LLM target completion: add the targets a source's annotators did not label.

FiQA labels only the target its score is about (93 posts with >= 2 cashtags
have exactly one target) and SEntFiN's entities come from matching a fixed
list, so neither is exhaustive. Trained as ordinary span tagging, a model would
learn that a missed entity is "not a target". This module asks an LLM, for each
text, to (1) type the gold targets and (2) list the missing ones, and turns the
answers into extra pair rows marked ``target_source = "llm"``.

Rules enforced in code, not trusted to the model: an added target must occur
verbatim in the text, must not overlap or repeat a gold target, and is dropped
otherwise (counted in the usage log). Gold targets are never changed.

The extractor is validated on FinEntity, the one exhaustively annotated source:
part of each sentence's gold targets is given, the rest hidden, and the hidden
ones must be found (`validate`).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .annotate import CONFIDENCE, DEFAULT_MODEL, Item, _same_target, _usage, read_labels
from .text import align_targets

TARGET_TYPES = (
    "company", "index_market", "sector", "commodity", "currency", "bonds_rates",
    "fund_investor_group", "regulator_government", "country_economy", "other",
)
# What FinEntity annotates consistently (companies incl. data/rating firms, indices, regulators).
FINENTITY_SCOPE = ("company", "index_market", "regulator_government")

DEFINITION = [
    "A target is an entity that the text gives financial information or a view about. Types:",
    "- company: companies and other organisations, including banks, brokerages, rating agencies, data "
    "providers and fund houses; a cashtag such as $AAPL is the company (write the target without the $).",
    "- index_market: stock indices and futures on them, stock exchanges as markets, the stock market as a "
    "whole ('markets', 'European shares', 'Wall Street').",
    "- sector: an industry or group of stocks ('IT stocks', 'banks', 'healthcare', 'midcaps').",
    "- commodity: gold, crude oil, metals, agricultural products. - currency: rupee, dollar, euro, bitcoin.",
    "- bonds_rates: bonds, yields, interest rates as market variables.",
    "- fund_investor_group: mutual funds, ETFs, FIIs, hedge funds, retail investors as a group.",
    "- regulator_government: regulators, central banks, ministries, governments.",
    "- country_economy: a country or region as an economy. - other: any other organisation that fits the "
    "definition.",
]
RULES = [
    "Include organisations quoted as the source of a view or data (brokerages, research firms, Refinitiv).",
    "Do not list people, job titles, products, or amounts of money; do not list a currency or commodity that "
    "only appears as a unit or a cause ('$5 billion', 'citing a stronger dollar').",
    "Copy each added target exactly as it appears in the text (same spelling and case), as the full name "
    "used there ('Spirit Airlines Inc', not 'Spirit').",
    "Never repeat, shorten, extend or overlap a given target, and never list one entity twice.",
    "If nothing is missing, return an empty list. Most texts have 0-2 missing targets.",
]
# (text, given targets, expected additions [(target, type)], types of the given targets)
# FinEntity / SEntFiN examples are from their *train* splits; the post is invented.
EXAMPLES = (
    ("AMC Entertainment Holdings Inc tumbled 39% after the cinema chain's preferred stock listing started trading "
     "and its UK-based rival Cineworld Group warned of a possible bankruptcy filing.",
     {"AMC Entertainment Holdings Inc": "company"}, [("Cineworld Group", "company")]),
    ("TI reported revenue of $5.24 billion in the third quarter, compared with analysts' average estimate of "
     "$5.14 billion, per Refinitiv data.", {"TI": "company"}, [("Refinitiv", "company")]),
    ("At 08:52 a.m. ET, Dow e-minis were down 62 points, or 0.18%, S&P 500 e-minis were down 2.25 points, or "
     "0.05%, and Nasdaq 100 e-minis were down 0.5 points, or flat.", {"Dow e-minis": "index_market"},
     [("S&P 500 e-minis", "index_market"), ("Nasdaq 100 e-minis", "index_market")]),
    ("In commodities, Brent crude futures rose 0.1% to $92.43 a barrel while U.S. West Texas Intermediate (WTI) "
     "crude gained 0.22% to $86.72 per barrel.", {"Brent crude": "commodity"},
     [("U.S. West Texas Intermediate (WTI) crude", "commodity")]),
    ("(Reporting by Bansari Mayur Kamdar and Praveen Paramasivam in Bengaluru; Editing by Sriraj Kalluvila)",
     {}, []),
    ("Brent drops below $87 as Fed optimism lifts dollar", {"Brent": "commodity"},
     [("Fed", "regulator_government"), ("dollar", "currency")]),
    ("Syngenta surges as European shares edge higher", {"Syngenta": "company"},
     [("European shares", "index_market")]),
    ("Be tactful on IT stocks: Deven Choksey", {"IT stocks": "sector"}, []),
    ("$TSLA and $NIO both ripping today while $SPY sits flat", {"TSLA": "company"},
     [("NIO", "company"), ("SPY", "index_market")]),
)


def build_instructions() -> str:
    lines = ["You complete target annotations for financial news headlines, social-media posts and news sentences.",
             "For every text you get the targets already annotated. (1) Give the type of each given target. "
             "(2) List the targets that are missing.", "", *DEFINITION, "", "Rules:",
             *[f"{i}. {r}" for i, r in enumerate(RULES, 1)], "", "Examples:"]
    for text, given, added in EXAMPLES:
        lines.append(f"Text: {json.dumps(text, ensure_ascii=False)}")
        lines.append("  given: " + (", ".join(f"{json.dumps(t)} ({ty})" for t, ty in given.items()) or "none"))
        lines.append("  added: " + (", ".join(f"{json.dumps(t)} ({ty})" for t, ty in added) or "none"))
    lines += ["", "Input: numbered texts T1, T2, ...; given targets have keys T<text>.<n>. Return one entry per "
              "text with its key, the type of every given target (copy key and target exactly) and the added "
              "targets."]
    return "\n".join(lines)


def response_schema() -> dict:
    typ = {"type": "string", "enum": list(TARGET_TYPES)}
    return {
        "type": "object", "additionalProperties": False, "required": ["texts"],
        "properties": {"texts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["key", "given", "added"],
            "properties": {
                "key": {"type": "string"},
                "given": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["key", "target", "type"],
                    "properties": {"key": {"type": "string"}, "target": {"type": "string"}, "type": typ}}},
                "added": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["target", "type", "confidence"],
                    "properties": {"target": {"type": "string"}, "type": typ,
                                   "confidence": {"type": "string", "enum": list(CONFIDENCE)}}}},
            }}}},
    }


INSTRUCTIONS = build_instructions()
PROMPT_VERSION = "t1-" + hashlib.sha1((INSTRUCTIONS + json.dumps(response_schema(), sort_keys=True)).encode()
                                      ).hexdigest()[:8]


def render(items: list[Item]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"T{i} ({it.text_type}): {json.dumps(it.text, ensure_ascii=False)}")
        if not it.pairs:
            lines.append("  given: none")
        for j, (_, target) in enumerate(it.pairs, 1):
            lines.append(f"  T{i}.{j}: {json.dumps(target, ensure_ascii=False)}")
    return "\n".join(lines)


_ANAPHORIC = re.compile(r"(?i)^(the|this|that|these|those|its|their|his|her|our|a|an)\s")
_BARE_GENERIC = {"benchmark", "index", "the index", "company", "firm", "group", "shares", "stock"}


def is_unlinkable(t: str) -> bool:
    """References that cannot be tied to a specific entity ('the market', 'the index', 'benchmark').
    Qualified generics stay ('Japanese shares', 'Wall Street', 'markets' as SEntFiN annotates them)."""
    return bool(_ANAPHORIC.match(t)) or t.casefold() in _BARE_GENERIC


def filter_added(added: list[dict]) -> list[dict]:
    return [a for a in added if not is_unlinkable(a["target"])]


def clean_added(text: str, given: list[str], added: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep added targets that occur verbatim and do not repeat/overlap gold; return (kept, dropped)."""
    kept, dropped, seen = [], [], {g.casefold() for g in given}
    gold_spans = [s for s in align_targets(text, given) if s.found]
    for a in added:
        t = a["target"].strip()
        if re.fullmatch(r"\$[A-Za-z][A-Za-z.]{0,9}", t) and t[1:] in text:  # cashtag -> ticker
            t = t[1:]
        if is_unlinkable(t):
            dropped.append(f"unlinkable reference: {t!r}")
            continue
        if not re.search(r"[A-Za-z]", t):
            dropped.append(f"no letters: {a['target']!r}")
            continue
        if t not in text:
            dropped.append(f"not verbatim: {a['target']!r}")
            continue
        if t.casefold() in seen:
            dropped.append(f"duplicate: {t!r}")
            continue
        free = [o for o in _all_occurrences(text, t)
                if not any(o[0] < g.end and g.start < o[1] for g in gold_spans)]
        if not free:
            dropped.append(f"overlaps gold: {t!r}")
            continue
        seen.add(t.casefold())
        kept.append({"target": t, "type": a["type"], "confidence": a["confidence"]})
    return kept, dropped


def _all_occurrences(text: str, t: str) -> list[tuple[int, int]]:
    out, i = [], text.find(t)
    while i >= 0:
        out.append((i, i + len(t)))
        i = text.find(t, i + 1)
    return out


def parse(output_text: str, items: list[Item]) -> list[dict]:
    data = json.loads(output_text)["texts"]
    by_key = {d["key"]: d for d in data}
    records = []
    for i, it in enumerate(items, 1):
        d = by_key.get(f"T{i}")
        if d is None:
            raise ValueError(f"missing text T{i}")
        gk = {g["key"]: g for g in d["given"]}
        given = []
        for j, (uid, target) in enumerate(it.pairs, 1):
            g = gk.get(f"T{i}.{j}")
            if g is None or not _same_target(g["target"], target):
                raise ValueError(f"given target T{i}.{j} missing or misaligned")
            given.append({"uid": uid, "target": target, "type": g["type"]})
        kept, dropped = clean_added(it.text, [t for _, t in it.pairs], d["added"])
        records.append({"text_id": it.text_id, "given": given, "added": kept, "dropped": dropped})
    return records


class TargetCompleter:
    def __init__(self, client, model: str = DEFAULT_MODEL, reasoning_effort: str | None = "low",
                 max_attempts: int = 3):
        self.client, self.model, self.reasoning_effort, self.max_attempts = client, model, reasoning_effort, max_attempts

    def _request(self, prompt: str):
        kw = dict(model=self.model, instructions=INSTRUCTIONS, input=prompt, store=False,
                  text={"format": {"type": "json_schema", "name": "target_completion",
                                   "schema": response_schema(), "strict": True}})
        if self.reasoning_effort:
            kw["reasoning"] = {"effort": self.reasoning_effort}
        return self.client.responses.create(**kw)

    def complete_batch(self, items: list[Item]):
        usage, error = {}, None
        for _ in range(self.max_attempts):
            resp = self._request(render(items))
            for k, v in _usage(resp).items():
                usage[k] = usage.get(k, 0) + v
            try:
                recs = parse(resp.output_text, items)
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                error = e
                usage["invalid_outputs"] = usage.get("invalid_outputs", 0) + 1
        else:
            if len(items) == 1:
                return [], usage, [(items[0], f"invalid output: {error}")]
            mid = len(items) // 2
            recs, failed = [], []
            for half in (items[:mid], items[mid:]):
                r, u, f = self.complete_batch(half)
                recs += r
                failed += f
                for k, v in u.items():
                    usage[k] = usage.get(k, 0) + v
            return recs, usage, failed
        meta = {"model": self.model, "prompt_version": PROMPT_VERSION, "reasoning_effort": self.reasoning_effort}
        return [{**r, **meta} for r in recs], usage, []

    def run(self, items: list[Item], out_path: Path, batch_size: int = 10, workers: int = 6, log=print) -> dict:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = done_text_ids(out_path, self.model, self.reasoning_effort)
        todo = [it for it in items if it.text_id not in done]
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        stats = {"task": "target_completion", "model": self.model, "prompt_version": PROMPT_VERSION,
                 "already_done": len(items) - len(todo), "batches": len(batches), "texts_done": 0, "texts_failed": 0,
                 "added": 0, "dropped": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
                 "reasoning_tokens": 0, "invalid_outputs": 0}
        errors, t0 = [], time.time()
        with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.complete_batch, b): b for b in batches}
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    recs, usage, failed = fut.result()
                except Exception as e:
                    stats["texts_failed"] += len(futs[fut])
                    errors.append(f"{futs[fut][0].text_id}: {type(e).__name__}: {e}")
                    continue
                for it, why in failed:
                    stats["texts_failed"] += 1
                    errors.append(f"{it.text_id}: {why}")
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    stats["added"] += len(r["added"])
                    stats["dropped"] += len(r["dropped"])
                f.flush()
                stats["texts_done"] += len(recs)
                for k, v in usage.items():
                    stats[k] = stats.get(k, 0) + v
                if n % 50 == 0 or n == len(batches):
                    log(f"  {n}/{len(batches)} batches, {stats['texts_done']} texts, +{stats['added']} targets, "
                        f"{stats['input_tokens']:,} in / {stats['output_tokens']:,} out, {time.time() - t0:.0f}s")
        stats["seconds"] = round(time.time() - t0, 1)
        stats["errors"] = errors[:20]
        stats["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with out_path.with_suffix(".usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        return stats


def done_text_ids(path: Path, model: str, reasoning_effort: str | None) -> set[str]:
    df = read_labels(path)
    if df.empty:
        return set()
    same = (df["model"] == model) & (df["prompt_version"] == PROMPT_VERSION)
    same &= df["reasoning_effort"].isna() if reasoning_effort is None else df["reasoning_effort"].eq(reasoning_effort)
    return set(df.loc[same, "text_id"])


def latest_completions(path: Path) -> pd.DataFrame:
    """One row per text_id: the last record of the current prompt version."""
    df = read_labels(path)
    if df.empty:
        return df
    df = df[df["prompt_version"] == PROMPT_VERSION]
    return df.drop_duplicates("text_id", keep="last").reset_index(drop=True)


def items_from(pairs: pd.DataFrame, texts: pd.DataFrame, given: dict[str, list[tuple[str, str]]] | None = None
               ) -> list[Item]:
    """One Item per text; `given` overrides the (uid, target) list (used to hide gold in validation)."""
    by_text = pairs.groupby("text_id")[["uid", "target"]].apply(lambda g: list(map(tuple, g.to_numpy())))
    items = []
    for r in texts.itertuples(index=False):
        g = given[r.text_id] if given is not None else by_text.get(r.text_id, [])
        items.append(Item(r.text_id, r.text, r.text_type, list(g)))
    return items


def hide_gold(pairs: pd.DataFrame, texts: pd.DataFrame, keep_prob: float = 0.5, seed: int = 0
              ) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[str]]]:
    """Validation split of FinEntity gold: per text, keep each distinct target with `keep_prob`."""
    rng = random.Random(seed)
    given, hidden = {}, {}
    for tid in texts["text_id"]:
        g = pairs[pairs["text_id"] == tid]
        values = list(dict.fromkeys(g["target"]))
        keep = {v for v in values if rng.random() < keep_prob}
        given[tid] = [(u, t) for u, t in zip(g["uid"], g["target"]) if t in keep]
        hidden[tid] = [v for v in values if v not in keep]
    return given, hidden


def score_validation(records: pd.DataFrame, hidden: dict[str, list[str]], gold_all: dict[str, list[str]]) -> dict:
    """Recall of hidden gold; precision of additions overall and within FinEntity's scope.
    A match is exact (case-insensitive) or one string containing the other (partial)."""
    def match(a, b):
        a, b = a.casefold(), b.casefold()
        return "exact" if a == b else ("partial" if a in b or b in a else None)

    tp_e = tp_p = n_hidden = n_added = n_added_scope = tp_scope = fp_on_empty = n_added_match = 0
    n_empty = 0
    for r in records.itertuples(index=False):
        r = r._replace(added=filter_added(r.added))
        hid = hidden.get(r.text_id, [])
        n_hidden += len(hid)
        found = {h: None for h in hid}
        if not gold_all.get(r.text_id):
            n_empty += 1
            fp_on_empty += len(r.added)
        for a in r.added:
            n_added += 1
            scope = a["type"] in FINENTITY_SCOPE
            n_added_scope += scope
            m = [(h, match(a["target"], h)) for h in hid]
            m = [x for x in m if x[1]]
            if m:
                h, kind = sorted(m, key=lambda x: x[1] != "exact")[0]
                n_added_match += 1
                tp_scope += scope
                if found[h] != "exact":
                    found[h] = kind
        tp_e += sum(v == "exact" for v in found.values())
        tp_p += sum(v is not None for v in found.values())
    return {"hidden": n_hidden, "added": n_added, "recall_exact": tp_e / max(1, n_hidden),
            "recall_partial": tp_p / max(1, n_hidden),
            "precision": n_added_match / max(1, n_added),
            "precision_in_finentity_scope": tp_scope / max(1, n_added_scope), "added_in_scope": n_added_scope,
            "empty_texts": n_empty, "additions_on_empty_texts": fp_on_empty}


def completion_files(annotations_dir: Path, source: str) -> list[Path]:
    return sorted(p for p in (annotations_dir / "completion").glob(f"{source}_*.jsonl")
                  if not p.name.endswith(".usage.jsonl") and "_validation_" not in p.name)


def add_completed_targets(pairs: pd.DataFrame, texts: pd.DataFrame, annotations_dir: Path,
                          aliases: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add LLM-completed targets as pair rows (target_source='llm', polarity filled later by the
    sentiment labels) and type the gold targets. Texts with a completion record become 'llm_completed'."""
    from .entities import entity_id

    info, new_rows = {"texts_completed": 0, "targets_added": 0}, []
    pairs = pairs.copy()
    texts = texts.copy()
    for source in ("sentfin", "fiqa"):
        recs = pd.concat([latest_completions(f) for f in completion_files(annotations_dir, source)] or [pd.DataFrame()],
                         ignore_index=True)
        if recs.empty:
            continue
        recs = recs.drop_duplicates("text_id", keep="last").set_index("text_id")
        src = pairs[(pairs["source"] == source) & (pairs["target_source"] == "gold")]
        recs = recs[recs.index.isin(src["text_id"])]
        types = {g["uid"]: g["type"] for gs in recs["given"] for g in gs}
        mask = pairs["uid"].isin(types)
        pairs.loc[mask, "target_type"] = pairs.loc[mask, "uid"].map(types)
        first = src.drop_duplicates("text_id").set_index("text_id")
        gold_targets = src.groupby("text_id")["target"].agg(list)
        for tid, r in recs.iterrows():
            base = first.loc[tid]
            text = base["text"]
            gold = [s for s in align_targets(text, gold_targets[tid]) if s.found]
            taken = [(s.start, s.end) for s in gold]
            for k, a in enumerate(filter_added(r["added"])):
                occ = [o for o in _all_occurrences(text, a["target"])
                       if not any(o[0] < e and s < o[1] for s, e in taken)]
                if not occ:
                    continue
                s, e = occ[0]
                taken.append((s, e))
                row = base.to_dict()
                row.update({"uid": f"{tid}#x{k}", "text_id": tid, "target": a["target"], "target_raw": a["target"],
                            "start": s, "end": e, "span_match": "exact", "polarity": None, "label": pd.NA,
                            "score": float("nan"), "polarity_finben": None, "fiqa_aspect": None,
                            "fiqa_aspect_l1": None, "fiqa_aspect_l2": None,
                            "entity_id": entity_id(a["target"], aliases), "quality_flags": "llm_target",
                            "target_source": "llm", "sentiment_source": "llm", "target_type": a["type"],
                            "target_annotation": "llm_completed",
                            "target_confidence": a["confidence"]})
                new_rows.append(row)
        done = set(recs.index)
        pairs.loc[pairs["text_id"].isin(done), "target_annotation"] = "llm_completed"
        texts.loc[texts["text_id"].isin(done), "target_annotation"] = "llm_completed"
        info["texts_completed"] += len(done)
    if new_rows:
        pairs = pd.concat([pairs, pd.DataFrame(new_rows)], ignore_index=True)
        pairs["start"] = pairs["start"].astype("Int64")
        pairs["end"] = pairs["end"].astype("Int64")
    info["targets_added"] = len(new_rows)
    return pairs, texts, info
