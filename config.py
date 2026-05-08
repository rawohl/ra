"""
ra — shared constants

Single source of truth for values that are referenced across multiple modules.
Import from here; never redefine locally.
"""

HOLD_DAYS       = 5       # holding period in trading days (one week)
TOP_N           = 5       # long signals + short signals to select per day
MIN_SPREAD      = 0.04    # minimum (max_prob - min_prob) across all stocks on a given day;
                          # below this the model is too uncertain to trade
INITIAL_CAPITAL = 10_000  # backtest starting equity (€)
ROUND_TRIP_COST = 0.0016  # 8 bps commission + 3 bps slippage per side, round-trip

# walk-forward structure
TRAIN_YEARS     = 2       # minimum training history required before first test fold
TEST_MONTHS     = 6       # each fold's test window (semi-annual — ~18 folds over 9yr evaluation)
HPO_YEARS       = 2       # dedicated HPO zone; covers 2015-2017 for regime diversity
                          # walk-forward test sets never overlap this period

# kept for any code that still references it; no longer used as a signal gate
MIN_PROB        = 0.52
