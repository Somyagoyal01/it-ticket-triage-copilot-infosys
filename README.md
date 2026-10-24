# IT Ticket Triage Copilot (classify → retrieve → grounded reply → route)

> Personal project by Somya Goyal, built while applying to Infosys. Not affiliated with or endorsed by Infosys; uses public/synthetic data.

## Why this matters for Infosys

**Problem:** service desks get large numbers of repetitive L1 tickets (VPN, passwords, MFA, Outlook, access requests). Agents spend time reading, categorising and pasting the same knowledge-base steps.
**Who feels it:** managed-services and application-support teams, and the end users waiting for a fix.
**What this shows:** a small GenAI copilot. It classifies the ticket, finds the right KB article, drafts a reply that may only use that article's steps, and **auto-suggests the reply only when it is confident**. Everything else goes to a human agent.

Public context:
- In Nov 2025 Infosys launched **Topaz Fabric** for IT operations, with 50+ AI agents and a human-in-the-loop approach (Voice&Data, 3 Nov 2025). This repo is a tiny, inspectable version of that pattern: agent output is checked and only released when grounded.
- The Python ML/GenAI JD (INFSYS-EXTERNAL-255682) asks for chatbots/virtual assistants, RAG with vector search, prompt design, REST APIs (FastAPI/Flask), testing and performance work. Each piece is here.

## Results (real numbers from `python -m triage.evaluate`)

| Test set | Category acc. | KB hit@1 | KB hit@3 | Auto-suggested | Precision of auto-suggestions |
|---|---|---|---|---|---|
| Generated tickets (same templates as training, 400) | 1.00 | 0.953 | 0.953 | 94% | 1.00 |
| **Hand-written held-out tickets** (20, messy real-world wording, never used for tuning) | **0.85** | **0.80** | 0.85 | 75% | **0.933** (14 of 15 correct) |

The gap between the two rows is the honest finding: template-generated tests overstate quality. On real-looking tickets ("C drive is red, only 300 MB left", "the globalprotect thing just spins") the lexical retriever misses 4 of 20. The confidence gate still routes most of those misses to a human: 1 of 15 auto-suggestions was wrong.

## How it works

- `data.py`: synthetic tickets from templates, plus a simple, explainable priority rule (P2 for urgent/blocking words).
- `core.py`:
  - **Classifier**: word plus character n-gram TF-IDF with logistic regression.
  - **KBRetriever**: hybrid word and char-n-gram similarity over `data/kb.json`.
  - **Draft**: produced through an `llm` callable. Offline by default; any OpenAI/Azure OpenAI wrapper can be plugged in.
  - **Groundedness check**: every sentence must come from the cited article, or the draft is dropped.
  - **Routing**: auto-suggest only if the classifier and retriever agree and both clear their thresholds.
- `api.py`: FastAPI service, `POST /triage` with `{"text": "..."}`. It returns category, priority, confidence, KB id, route, draft and citations.
- `tests/`: 6 pytest tests, including one where a "hallucinating" LLM gets blocked from auto-sending.

## Run it

```bash
pip install -r requirements.txt
cd src && python -m triage.evaluate            # writes results/metrics.json
uvicorn triage.api:app --reload                # then POST to http://127.0.0.1:8000/triage
cd .. && pytest -q
```

No API key is needed. For a real LLM, pass `llm=` to `Copilot`, for example a function that calls Azure OpenAI with the prompt it receives. Keep the groundedness check on.

## Limitations / next steps

- The KB (12 articles) and training tickets are synthetic and small. A real desk has thousands of articles and needs a vector index (FAISS/ChromaDB/Azure AI Search) with embeddings, which should help with paraphrases like "globalprotect".
- The groundedness check is strict exact-sentence matching. A real LLM that rephrases would need an entailment or overlap-based check instead.
- Next: embeddings plus re-ranking, measured on the same hand-written set, and per-category thresholds tuned on a separate validation set (not the held-out one).
