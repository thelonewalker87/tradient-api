# ai_router.py
# ------------
# The only file your frontend talks to.
# Exposes one endpoint: POST /ai/query
# Routes each request type to the right handler.
# Each handler builds a prompt, calls Claude, and returns structured JSON.
#
# To add a new AI capability: write a handle_X function, add it to HANDLERS.
# Nothing else needs to change.
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Any
import json
from openai import OpenAI

from grader import grade_trade
from models import TradeInput

router = APIRouter(prefix="/ai")

MODEL  = "meta-llama/llama-3.3-70b-instruct:free"

client = OpenAI(                               
    base_url = "https://openrouter.ai/api/v1",
    api_key  = os.environ.get("OPENROUTER_API_KEY"),
)


# ── Single request envelope ────────────────────────────────────────────────────
# Frontend always sends: { "type": "...", "payload": { ... } }

class AIRequest(BaseModel):
    type: Literal[
        "grade_trade",
        "analyse_performance",
        "pre_trade_check",
        "coaching_chat",
        "journal_reflection",
        "explain_grade",
    ]
    payload: dict[str, Any]


# ── Handlers ───────────────────────────────────────────────────────────────────

def handle_grade_trade(payload: dict) -> dict:
    """
    Grades a single trade.
    Calls grader.py which runs metrics.py then Claude.
    payload: { trade: TradeInput fields }
    returns: GradeResult
    """
    trade = TradeInput(**payload["trade"])
    return grade_trade(trade).dict()


def handle_analyse_performance(payload: dict) -> dict:
    """
    Analyses a batch of already-graded trades and answers a question about them.
    payload: { trades: [GradeResult, ...], question: str }
    returns: { answer, top_weakness, recommendations, positive_patterns }
    """
    trades   = payload["trades"]
    question = payload.get("question", "What is my biggest weakness?")

    # Compress trades to fit context window — only send what Claude needs
    summary = [
        {
            "ticker":    t["trade"]["ticker"] if "trade" in t else t.get("ticker", "?"),
            "grade":     t["letter_grade"],
            "score":     t["overall_score"],
            "patterns":  t["patterns"],
            "pnl":       t["metrics"]["pnl"],
            "actual_rr": t["metrics"]["actual_rr"],
        }
        for t in trades
    ]

    response = client.messages.create(
        model      = MODEL,
        max_tokens = 1024,
        system     = "You are an expert trading coach. Respond with valid JSON only, no markdown.",
        messages   = [{
            "role": "user",
            "content": f"""
Here are {len(summary)} graded trades:
{json.dumps(summary, indent=2)}

Question: {question}

Respond in JSON:
{{
  "answer": "<direct answer>",
  "top_weakness": "<single biggest weakness>",
  "recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"],
  "positive_patterns": ["<strength 1>", "<strength 2>"]
}}"""
        }]
    )
    return json.loads(response.content[0].text)


def handle_pre_trade_check(payload: dict) -> dict:
    """
    Scores a setup the trader is considering BEFORE entering.
    payload: { description: str, account_size: float, risk_percent: float }
    returns: { score, take_trade, reasons_for, reasons_against, what_to_watch }
    """
    response = client.messages.create(
        model      = MODEL,
        max_tokens = 512,
        system     = "You are a trading coach. Respond with valid JSON only, no markdown.",
        messages   = [{
            "role": "user",
            "content": f"""
A trader is considering this setup:
"{payload['description']}"
Account: ${payload.get('account_size', 10000)} | Risk: {payload.get('risk_percent', 1)}%

Score 0-100 and respond in JSON:
{{
  "score": <0-100>,
  "take_trade": <true|false>,
  "reasons_for": ["<reason>"],
  "reasons_against": ["<reason>"],
  "what_to_watch": "<one thing to monitor once in the trade>"
}}"""
        }]
    )
    return json.loads(response.content[0].text)


def handle_coaching_chat(payload: dict) -> dict:
    """
    Multi-turn coaching Q&A. Pass history to maintain conversation context.
    payload: { message: str, history: [{ role, content }, ...] }
    returns: { reply: str }
    """
    history  = payload.get("history", [])
    message  = payload["message"]
    messages = history + [{"role": "user", "content": message}]

    response = client.messages.create(
        model      = MODEL,
        max_tokens = 1024,
        system     = "You are an expert trading coach. Be direct, specific, and actionable.",
        messages   = messages
    )
    return {"reply": response.content[0].text}


def handle_journal_reflection(payload: dict) -> dict:
    """
    Extracts structured lessons from a free-text end-of-day journal entry.
    payload: { entry: str }
    returns: { mood_score, key_lesson, mistakes, what_went_well, tomorrow_focus }
    """
    response = client.messages.create(
        model      = MODEL,
        max_tokens = 512,
        system     = "You are a trading psychologist. Respond with valid JSON only, no markdown.",
        messages   = [{
            "role": "user",
            "content": f"""
Trader's journal entry:
"{payload['entry']}"

Extract insights in JSON:
{{
  "mood_score": <1-10>,
  "key_lesson": "<most important lesson>",
  "mistakes": ["<mistake 1>"],
  "what_went_well": ["<positive 1>"],
  "tomorrow_focus": "<one specific thing to focus on tomorrow>"
}}"""
        }]
    )
    return json.loads(response.content[0].text)


def handle_explain_grade(payload: dict) -> dict:
    """
    Rewrites an existing GradeResult in plain English for the trader.
    payload: { grade: GradeResult dict }
    returns: { explanation: str }
    """
    g = payload["grade"]
    response = client.messages.create(
        model      = MODEL,
        max_tokens = 256,
        system     = "You are a friendly trading coach. Use plain language, no jargon.",
        messages   = [{
            "role": "user",
            "content": f"""
Explain this grade in plain English. Under 100 words. End with one action for tomorrow.

Grade: {g['letter_grade']} ({g['overall_score']}/100)
Entry:  {g['entry_quality']['score']}/25 — {g['entry_quality']['feedback']}
Risk:   {g['risk_management']['score']}/25 — {g['risk_management']['feedback']}
Thesis: {g['trade_thesis']['score']}/25 — {g['trade_thesis']['feedback']}
Exit:   {g['exit_quality']['score']}/25 — {g['exit_quality']['feedback']}
Patterns: {g['patterns']}"""
        }]
    )
    return {"explanation": response.content[0].text}


# ── Handler registry ───────────────────────────────────────────────────────────

HANDLERS = {
    "grade_trade":         handle_grade_trade,
    "analyse_performance": handle_analyse_performance,
    "pre_trade_check":     handle_pre_trade_check,
    "coaching_chat":       handle_coaching_chat,
    "journal_reflection":  handle_journal_reflection,
    "explain_grade":       handle_explain_grade,
}


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post("/query")
async def ai_query(request: AIRequest):
    handler = HANDLERS.get(request.type)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unknown type: {request.type}")
    try:
        return handler(request.payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── App entry point ────────────────────────────────────────────────────────────
# Run with: uvicorn ai_router:app --reload --port 8000

app = FastAPI(title="Trade Grader AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # Tighten this to your domain in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)
app.include_router(router)
