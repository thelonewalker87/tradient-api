# ai_router.py
import os
import json
from typing import Literal, Any
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from grader import grade_trade
from models import TradeInput

load_dotenv()

router = APIRouter(prefix="/ai")

MODEL = "openrouter/free"

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key  = os.environ.get("OPENROUTER_API_KEY"),
)


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


def handle_grade_trade(payload: dict) -> dict:
    trade = TradeInput(**payload["trade"])
    return grade_trade(trade).dict()


def handle_analyse_performance(payload: dict) -> dict:
    trades   = payload["trades"]
    question = payload.get("question", "What is my biggest weakness?")

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

    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role": "system",
                "content": "You are an expert trading coach. Respond with valid JSON only, no markdown."
            },
            {
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
            }
        ]
    )
    return json.loads(response.choices[0].message.content)


def handle_pre_trade_check(payload: dict) -> dict:
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role": "system",
                "content": "You are a trading coach. Respond with valid JSON only, no markdown."
            },
            {
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
            }
        ]
    )
    return json.loads(response.choices[0].message.content)


def handle_coaching_chat(payload: dict) -> dict:
    history  = payload.get("history", [])
    message  = payload["message"]

    messages = [
        {
            "role": "system",
            "content": "You are an expert trading coach. Be direct, specific, and actionable."
        }
    ]
    messages += history
    messages += [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model    = MODEL,
        messages = messages
    )
    return {"reply": response.choices[0].message.content}


def handle_journal_reflection(payload: dict) -> dict:
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role": "system",
                "content": "You are a trading psychologist. Respond with valid JSON only, no markdown."
            },
            {
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
            }
        ]
    )
    return json.loads(response.choices[0].message.content)


def handle_explain_grade(payload: dict) -> dict:
    g = payload["grade"]
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role": "system",
                "content": "You are a friendly trading coach. Use plain language, no jargon."
            },
            {
                "role": "user",
                "content": f"""
Explain this grade in plain English. Under 100 words. End with one action for tomorrow.

Grade: {g['letter_grade']} ({g['overall_score']}/100)
Entry:  {g['entry_quality']['score']}/25 — {g['entry_quality']['feedback']}
Risk:   {g['risk_management']['score']}/25 — {g['risk_management']['feedback']}
Thesis: {g['trade_thesis']['score']}/25 — {g['trade_thesis']['feedback']}
Exit:   {g['exit_quality']['score']}/25 — {g['exit_quality']['feedback']}
Patterns: {g['patterns']}"""
            }
        ]
    )
    return {"explanation": response.choices[0].message.content}


HANDLERS = {
    "grade_trade":         handle_grade_trade,
    "analyse_performance": handle_analyse_performance,
    "pre_trade_check":     handle_pre_trade_check,
    "coaching_chat":       handle_coaching_chat,
    "journal_reflection":  handle_journal_reflection,
    "explain_grade":       handle_explain_grade,
}


@router.post("/query")
async def ai_query(request: AIRequest):
    handler = HANDLERS.get(request.type)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unknown type: {request.type}")
    try:
        return handler(request.payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app = FastAPI(title="Tradient")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)
app.include_router(router)