"""Evaluate on (a) a generated test split and (b) the hand-written held-out tickets.

    python -m triage.evaluate      (from src/)
"""
from __future__ import annotations

import json
from pathlib import Path

from .core import DATA, Copilot
from .data import make_tickets

RESULTS = Path(__file__).resolve().parents[2] / "results"


def score(copilot: Copilot, rows: list[dict]) -> dict:
    res = [copilot.triage(r["text"]) for r in rows]
    n = len(rows)
    cat_acc = sum(t.category == r["category"] for t, r in zip(res, rows)) / n
    hit1 = sum(t.kb_id == r["kb"] for t, r in zip(res, rows)) / n
    hit3 = sum(r["kb"] in [a["id"] for a, _ in copilot.ret.search(r["text"], 3)] for r in rows) / n
    auto = [(t, r) for t, r in zip(res, rows) if t.route == "auto_suggest"]
    auto_correct = sum(t.kb_id == r["kb"] for t, r in auto)
    misses = [r["text"] for t, r in zip(res, rows) if t.kb_id != r["kb"]]
    return {"n": n, "category_accuracy": round(cat_acc, 3), "kb_hit@1": round(hit1, 3),
            "kb_hit@3": round(hit3, 3), "auto_suggest_rate": round(len(auto) / n, 3),
            "auto_suggest_precision": round(auto_correct / max(len(auto), 1), 3), "kb_misses": misses[:10]}


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    copilot = Copilot(make_tickets(1200, seed=11))
    generated_test = make_tickets(400, seed=12)
    heldout = [json.loads(l) for l in (DATA / "heldout_tickets.jsonl").read_text().splitlines() if l.strip()]
    out = {"generated_test": score(copilot, generated_test), "heldout_handwritten": score(copilot, heldout)}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
