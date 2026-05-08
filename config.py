"""
ra — shared constants

Single source of truth for values that are referenced across multiple modules.
Import from here; never redefine locally.
"""

HOLD_DAYS       = 5       # holding period in trading days (one week)
TOP_N           = 5       # long signals + short signals to select per day
INITIAL_CAPITAL = 10_000  # backtest starting equity (€)

# 8 bps commission + 3 bps slippage per side, round-trip
ROUND_TRIP_COST = 0.0016

# kept for any code that still references it; no longer used as a signal gate
MIN_PROB        = 0.52
