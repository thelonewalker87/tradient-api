# metrics.py
# ----------
# Pure deterministic math. No Claude, no HTTP, no database.
# Called by grader.py BEFORE Claude is involved.
#
# Why separate from grader.py?
# - Math should never be delegated to an LLM (it hallucinates numbers)
# - These results are fully testable and reproducible
# - Claude receives the computed numbers as facts, not raw prices to calculate

from models import TradeInput, TradeMetrics, Direction


def calculate_metrics(trade: TradeInput) -> TradeMetrics:
    is_long    = trade.direction == Direction.LONG
    dollar_risk = abs(trade.entry_price - trade.stop_loss)

    # Planned R:R — only calculable if take profit was set
    if trade.take_profit:
        reward     = abs(trade.take_profit - trade.entry_price)
        planned_rr = round(reward / dollar_risk, 2) if dollar_risk > 0 else 0.0
    else:
        planned_rr = 0.0

    # Actual R:R — based on where the trade actually exited
    actual_gain = (trade.exit_price - trade.entry_price) if is_long else (trade.entry_price - trade.exit_price)
    actual_rr   = round(actual_gain / dollar_risk, 2) if dollar_risk > 0 else 0.0

    # Did price reach take profit or stop loss?
    if is_long:
        hit_target = trade.take_profit is not None and trade.exit_price >= trade.take_profit
        hit_stop   = trade.exit_price <= trade.stop_loss
    else:
        hit_target = trade.take_profit is not None and trade.exit_price <= trade.take_profit
        hit_stop   = trade.exit_price >= trade.stop_loss

    risk_percent = round((trade.position_size / trade.account_size) * 100, 2)
    pnl          = round(actual_gain * (trade.position_size / trade.entry_price), 2)

    return TradeMetrics(
        risk_reward_ratio = planned_rr,
        actual_rr         = actual_rr,
        risk_percent      = risk_percent,
        hit_target        = hit_target,
        hit_stop          = hit_stop,
        pnl               = pnl,
    )


def detect_patterns(trade: TradeInput, metrics: TradeMetrics) -> list[str]:
    """
    Rule-based flags passed to Claude as context.
    Each flag is a string that Claude uses when writing feedback.
    Stored in GradeResult.patterns for long-term pattern tracking.
    """
    flags = []

    if metrics.actual_rr > 0 and metrics.actual_rr < metrics.risk_reward_ratio:
        flags.append("early_exit_winner")

    if metrics.hit_stop and metrics.actual_rr >= 0:
        flags.append("stopped_out_at_breakeven_or_above")

    if metrics.risk_percent > 2.0:
        flags.append("oversized_position")

    if trade.take_profit is None:
        flags.append("no_take_profit_defined")

    if not trade.trade_notes or len(trade.trade_notes.strip()) < 20:
        flags.append("insufficient_trade_notes")

    if 0 < metrics.risk_reward_ratio < 1.0:
        flags.append("poor_planned_rr")

    return flags
