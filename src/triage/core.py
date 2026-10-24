"""Ticket triage copilot: classify -> retrieve KB -> grounded draft reply -> route.

Route rule: auto-suggest a reply only when both the classifier and the
retriever are confident AND agree on the category; otherwise hand to an agent.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .data import priority

DATA = Path(__file__).resolve().parents[2] / "data"


def _vectorizers():
    return (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))


class Classifier:
    def __init__(self):
        self.word, self.char = _vectorizers()
        self.clf = LogisticRegression(max_iter=2000, C=4.0)

    def fit(self, texts: list[str], labels: list[str]) -> "Classifier":
        X = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)])
        self.clf.fit(X, labels)
        return self

    def predict(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        X = hstack([self.word.transform(texts), self.char.transform(texts)])
        proba = self.clf.predict_proba(X)
        return list(self.clf.classes_[proba.argmax(1)]), proba.max(1)


class KBRetriever:
    def __init__(self, kb: list[dict]):
        self.kb = kb
        docs = [a["title"] + ". " + " ".join(a["steps"]) for a in kb]
        self.word, self.char = _vectorizers()
        self.Mw, self.Mc = self.word.fit_transform(docs), self.char.fit_transform(docs)

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        s = 0.5 * cosine_similarity(self.word.transform([query]), self.Mw)[0] \
            + 0.5 * cosine_similarity(self.char.transform([query]), self.Mc)[0]
        order = s.argsort()[::-1][:k]
        return [(self.kb[i], float(s[i])) for i in order]


def offline_llm(prompt: str) -> str:
    """Stand-in LLM: returns the numbered steps from the context unchanged."""
    return prompt.split("CONTEXT:\n", 1)[-1].strip()


def grounded(reply: str, article: dict) -> bool:
    """Every sentence of the reply must appear (normalised) in the cited article's steps."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()  # noqa: E731
    allowed = {norm(s) for s in article["steps"]}
    sents = [norm(re.sub(r"^\d+\.\s*", "", line)) for line in reply.splitlines() if line.strip()]
    return bool(sents) and all(s in allowed for s in sents)


@dataclass
class TriageResult:
    category: str
    priority: str
    confidence: float
    kb_id: str
    kb_score: float
    route: str
    draft_reply: str
    citations: list[str] = field(default_factory=list)


class Copilot:
    def __init__(self, train_rows: list[dict], kb: list[dict] | None = None,
                 llm: Callable[[str], str] = offline_llm, min_conf: float = 0.55, min_kb: float = 0.12):
        self.kb = kb or json.loads((DATA / "kb.json").read_text())
        self.clf = Classifier().fit([r["text"] for r in train_rows], [r["category"] for r in train_rows])
        self.ret = KBRetriever(self.kb)
        self.llm, self.min_conf, self.min_kb = llm, min_conf, min_kb

    def triage(self, text: str) -> TriageResult:
        cats, conf = self.clf.predict([text])
        cat, conf = cats[0], float(conf[0])
        hits = self.ret.search(text, k=3)
        # prefer the best article whose category agrees with the classifier
        art, score = next(((a, s) for a, s in hits if a["category"] == cat), hits[0])
        agree = art["category"] == cat
        auto = agree and conf >= self.min_conf and score >= self.min_kb
        reply, cites = "", []
        if auto:
            prompt = ("Write the fix steps for the user. Use ONLY the context.\nCONTEXT:\n"
                      + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(art["steps"])))
            reply = self.llm(prompt)
            if grounded(reply, art):
                cites = [art["id"]]
            else:  # an LLM added unsupported text -> do not auto-send
                auto, reply = False, ""
        return TriageResult(cat, priority(text), round(conf, 3), art["id"], round(score, 3),
                            "auto_suggest" if auto else "agent_queue", reply, cites)
