# models.py
# ---------
# Defines the shape of every object that flows through the system.
# Think of this as the contract between all other files.
# models.py knows nothing about Claude, math, or HTTP — it only defines structure.

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class Direction(str, Enum):
    LONG  = "long"
    SHORT = "short"


class TradeInput(BaseModel):
    """
    The raw trade data your frontend sends.
    Every other file receives a TradeInput as its starting point.
    """
    ticker:         str
    direction:      Direction
    entry_price:    float
    exit_price:     float
    stop_loss:      float
    take_profit:    Optional[float] = None
    account_size:   float
    position_size:  float
    trade_notes:    Optional[str]   = None
    session:        Optional[str]   = None
    strategy_tag:   Optional[str]   = None

    @validator("stop_loss")
    def stop_loss_must_be_logical(cls, sl, values):
        entry     = values.get("entry_price")
        direction = values.get("direction")
        if entry and direction:
            if direction == Direction.LONG and sl >= entry:
                raise ValueError("Stop loss must be below entry for a long trade")
            if direction == Direction.SHORT and sl <= entry:
                raise ValueError("Stop loss must be above entry for a short trade")
        return sl


class TradeMetrics(BaseModel):
    """
    Computed by metrics.py before Claude is called.
    Passed into grader.py so Claude receives pre-calculated numbers, not raw prices.
    """
    risk_reward_ratio:  float
    actual_rr:          float
    risk_percent:       float
    hit_target:         bool
    hit_stop:           bool
    pnl:                float


class DimensionScore(BaseModel):
    """One scored dimension returned by Claude."""
    score:    int       # 0–25
    max:      int = 25
    feedback: str


class GradeResult(BaseModel):
    """
    The final output returned to your frontend.
    Assembled in grader.py from Claude's response + calculated metrics.
    """
    overall_score:   int
    letter_grade:    str
    metrics:         TradeMetrics
    entry_quality:   DimensionScore
    risk_management: DimensionScore
    trade_thesis:    DimensionScore
    exit_quality:    DimensionScore
    summary:         str
    patterns:        list[str] = []
