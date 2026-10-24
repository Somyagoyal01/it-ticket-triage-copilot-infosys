"""FastAPI service:  uvicorn triage.api:app --reload  (run from src/)."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .core import Copilot
from .data import make_tickets

app = FastAPI(title="IT ticket triage copilot (demo, synthetic data)")
copilot = Copilot(make_tickets())


class Ticket(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "kb_articles": len(copilot.kb)}


@app.post("/triage")
def triage(ticket: Ticket) -> dict:
    return asdict(copilot.triage(ticket.text))
