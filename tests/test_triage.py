import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from triage.core import Copilot, grounded  # noqa: E402
from triage.data import make_tickets, priority  # noqa: E402

COPILOT = Copilot(make_tickets(600, seed=1))


def test_known_issue_gets_grounded_auto_suggestion():
    r = COPILOT.triage("vpn not connecting from home")
    assert r.category == "network" and r.kb_id == "KB-VPN"
    assert r.route == "auto_suggest" and r.citations == ["KB-VPN"]


def test_out_of_scope_ticket_goes_to_agent():
    r = COPILOT.triage("the coffee machine on floor 2 is leaking water")
    assert r.route == "agent_queue" and r.draft_reply == ""


def test_hallucinating_llm_is_blocked():
    bad_llm = lambda prompt: "1. Format your hard disk and reinstall Windows."  # noqa: E731
    cp = Copilot(make_tickets(600, seed=1), llm=bad_llm)
    r = cp.triage("vpn not connecting from home")
    assert r.route == "agent_queue" and r.citations == []


def test_grounded_check():
    art = {"steps": ["Restart the app.", "Sign in again."]}
    assert grounded("1. Restart the app.\n2. Sign in again.", art)
    assert not grounded("1. Restart the app.\n2. Call your bank.", art)


def test_priority_rule():
    assert priority("Production outage, orders down") == "P2"
    assert priority("printer not printing") == "P3"


def test_api_triage_endpoint():
    from triage.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/triage", json={"text": "account locked out after password expired"}).json()
    assert body["category"] == "access" and body["kb_id"] == "KB-PWD"
