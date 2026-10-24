"""Synthetic service-desk tickets built from templates (clearly labelled synthetic).

The hand-written held-out set in data/heldout_tickets.jsonl uses different,
messier wording and is never used for training or tuning.
"""
from __future__ import annotations

import random

TEMPLATES = {
    "KB-VPN": ["vpn not connecting", "unable to connect to vpn from home", "vpn client error when connecting",
               "remote access vpn keeps failing", "cannot reach intranet over vpn"],
    "KB-PWD": ["forgot my password", "account locked out", "need password reset",
               "password expired cannot login", "my login is locked"],
    "KB-MFA": ["mfa prompt not received", "not getting multi factor authentication notification",
               "authenticator app not working", "two factor code not accepted", "need to re-register mfa device"],
    "KB-MAIL": ["outlook not syncing", "emails not sending", "outlook stuck on updating folder",
                "mailbox full cannot send", "not receiving emails in outlook"],
    "KB-DISK": ["disk space low on laptop", "hard drive full", "no storage space left",
                "low disk space warning", "c drive almost full"],
    "KB-SLOW": ["laptop very slow", "computer freezing", "system performance is poor",
                "pc hangs frequently", "laptop takes long to start"],
    "KB-PRINT": ["printer not working", "cannot print document", "print job stuck in queue",
                 "network printer offline", "printer not printing"],
    "KB-SAP": ["need access to sap role", "request sap authorization", "access request for business application",
               "please grant role in sap", "need permission for erp module"],
    "KB-SW": ["install software on laptop", "need application installed", "software installation request",
              "please install tool", "cannot find software in software centre"],
    "KB-TEAMS": ["teams call audio not working", "teams meeting camera issue", "microphone not working in teams",
                 "teams not joining meeting", "teams video freezing"],
    "KB-DB": ["application cannot connect to database", "database connection error", "db timeout in application",
              "connection refused from database", "service account cannot reach database"],
    "KB-BUILD": ["ci build failing", "deployment pipeline failed", "build job error",
                 "pipeline cannot download dependency", "release pipeline broken"],
}
KB_CATEGORY = {"KB-VPN": "network", "KB-PWD": "access", "KB-MFA": "access", "KB-MAIL": "email",
               "KB-DISK": "hardware", "KB-SLOW": "hardware", "KB-PRINT": "hardware", "KB-SAP": "access",
               "KB-SW": "software", "KB-TEAMS": "software", "KB-DB": "application", "KB-BUILD": "application"}
PREFIX = ["", "hi team, ", "hello, ", "urgent: ", "please help, ", "issue: "]
SUFFIX = ["", " since today", " for the last hour", ", please check", " - blocking my work", " again"]
URGENT = ("urgent", "blocking", "production", "outage", "down")


def make_tickets(n: int = 1200, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    kbs = list(TEMPLATES)
    rows = []
    for _ in range(n):
        kb = rng.choice(kbs)
        text = rng.choice(PREFIX) + rng.choice(TEMPLATES[kb]) + rng.choice(SUFFIX)
        rows.append({"text": text, "category": KB_CATEGORY[kb], "kb": kb})
    return rows


def priority(text: str) -> str:
    """Simple, explainable priority rule (P2 for urgent/blocking words, else P3)."""
    t = text.lower()
    return "P2" if any(w in t for w in URGENT) else "P3"
