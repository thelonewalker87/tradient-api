# grader.py
# ---------
# The AI brain. This is the only file that talks to Claude.
# Orchestrates the full grading pipeline:
#   TradeInput → metrics.py → Claude → GradeResult
#
# Called by ai_router.py for "grade_trade" requests.
# All other AI request types (coaching, analysis etc.) call Claude directly
# in ai_router.py without going through this file.

import os
from openai import OpenAI
import json
from models import TradeInput, TradeMetrics, GradeResult, DimensionScore
from metrics import calculate_metrics, detect_patterns

MODEL = "openrouter/free"

client = OpenAI(                                   
    base_url = "https://openrouter.ai/api/v1",
    api_key  = os.environ.get("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """
You are an expert trading coach and performance analyst grading individual trades.

You receive:
- Raw trade data (entry, exit, stop, size, direction)
- Pre-calculated metrics (R:R, risk %, P&L) — trust these numbers exactly
- Rule-based pattern flags detected before you were called
- The trader's own notes explaining their reasoning

Return ONLY a JSON object with this exact structure, no markdown, no extra text:
{
  "entry_quality":   { "score": <0-25>, "feedback": "<specific feedback>" },
  "risk_management": { "score": <0-25>, "feedback": "<specific feedback>" },
  "trade_thesis":    { "score": <0-25>, "feedback": "<specific feedback>" },
  "exit_quality":    { "score": <0-25>, "feedback": "<specific feedback>" },
  "summary":         "<2-3 sentence overall narrative>"
}

Grading rules:
- Grade the PROCESS not the outcome. A losing trade with good process scores well.
- Be specific. "Good entry" is useless. Reference actual prices, levels, and reasoning.
- No trade notes = trade_thesis cannot exceed 10/25. A trade with no thesis is a gamble.
- Risk over 2% of account always reduces risk_management score significantly.
- No take profit set = exit_quality cannot exceed 15/25.
- Planned R:R below 1:1 = heavy penalty on entry_quality or risk_management.
- A losing trade that was well-executed should still score 60+. Say so explicitly.
"""


def build_prompt(trade: TradeInput, metrics: TradeMetrics, patterns: list[str]) -> str:
    return f"""
## Trade
Ticker: {trade.ticker} | Direction: {trade.direction.value.upper()}
Entry: {trade.entry_price} | Exit: {trade.exit_price}
Stop loss: {trade.stop_loss} | Take profit: {trade.take_profit or 'NOT SET'}
Session: {trade.session or 'Not specified'} | Strategy: {trade.strategy_tag or 'Not specified'}

## Pre-Calculated Metrics
Planned R:R: {metrics.risk_reward_ratio}:1
Actual R:R:  {metrics.actual_rr}:1
Risk %:      {metrics.risk_percent}%
Hit TP:      {metrics.hit_target}
Hit SL:      {metrics.hit_stop}
P&L:         ${metrics.pnl}

## Pattern Flags
{json.dumps(patterns) if patterns else 'None'}

## Trader Notes
{trade.trade_notes or 'No notes provided.'}

Grade this trade. Return only the JSON object.
"""


def score_to_letter(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def grade_trade(trade: TradeInput) -> GradeResult:
    """
    Full pipeline:
    1. Calculate deterministic metrics (metrics.py)
    2. Detect rule-based patterns (metrics.py)
    3. Send everything to Claude (this file)
    4. Parse Claude's JSON response
    5. Return a typed GradeResult
    """
    metrics  = calculate_metrics(trade)
    patterns = detect_patterns(trade, metrics)


    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(trade, metrics, patterns)}
        ]
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())

    entry   = DimensionScore(**parsed["entry_quality"])
    risk    = DimensionScore(**parsed["risk_management"])
    thesis  = DimensionScore(**parsed["trade_thesis"])
    exit_q  = DimensionScore(**parsed["exit_quality"])
    overall = entry.score + risk.score + thesis.score + exit_q.score

    return GradeResult(
        overall_score   = overall,
        letter_grade    = score_to_letter(overall),
        metrics         = metrics,
        entry_quality   = entry,
        risk_management = risk,
        trade_thesis    = thesis,
        exit_quality    = exit_q,
        summary         = parsed["summary"],
        patterns        = patterns,
    )
