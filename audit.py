"""
Backtest Audit Script v3

Checks for common sources of inflated backtest results:
1. Data overview
2. Lookahead bias (prob_up vs fwd_ret_5d correlation)
3. Walk-forward structure (fold ordering and continuity)
4. Return distribution sanity check
5. Per-fold performance (degradation over time)
6. VIX regime consistency (training range vs signal range)
7. VIX regime performance breakdown
8. Survivorship bias warning
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("  BACKTEST AUDIT v3")
print("=" * 60)


# ── 1. Load data ───────────────────────────────────────────────────────────────

pred_path     = Path("data/clean/predictions.parquet")
featured_path = Path("data/clean/featured.parquet")

if not pred_path.exists() or not featured_path.exists():
    print("ERROR: Run training first (data/clean/predictions.parquet missing).")
    exit(1)

preds    = pd.read_parquet(pred_path)
featured = pd.read_parquet(featured_path)

preds["date"]    = pd.to_datetime(preds["date"])
featured["date"] = pd.to_datetime(featured["date"])
for df in (preds, featured):
    col = df["date"]
    if col.dt.tz is not None:
        df["date"] = col.dt.tz_convert(None)

print(f"\n1. DATA OVERVIEW")
print(f"   Predictions : {len(preds):,} rows")
print(f"   Date range  : {preds['date'].min().date()} → {preds['date'].max().date()}")
print(f"   Tickers     : {preds['ticker'].nunique()}")
print(f"   Folds       : {preds['fold'].nunique()}")
print(f"   Signal rate : {(preds['signal'] != 0).mean():.1%}")


# ── 2. Lookahead bias ─────────────────────────────────────────────────────────
# A suspiciously high correlation between model confidence and future returns
# would suggest the model has access to information it shouldn't.

print(f"\n2. LOOKAHEAD BIAS CHECK  (5d hold horizon)")
if "fwd_ret_5d" in preds.columns:
    corr = preds["prob_up"].corr(preds["fwd_ret_5d"])
    print(f"   Corr(prob_up, fwd_ret_5d): {corr:.4f}")
    if abs(corr) > 0.15:
        print("   ⚠ HIGH — possible lookahead; investigate feature construction")
    elif abs(corr) > 0.05:
        print("   ✓ Moderate — expected for a working model")
    else:
        print("   ~ Low — model may have weak signal (or features are too smooth)")
else:
    print("   fwd_ret_5d not in predictions — re-run training.")


# ── 3. Walk-forward structure ─────────────────────────────────────────────────
# Verify folds are in chronological order with no gaps and no date overlap.

print(f"\n3. WALK-FORWARD STRUCTURE")
folds = sorted(preds["fold"].unique())
fold_ranges = {}
for fold in folds:
    fd = preds[preds["fold"] == fold]
    fold_ranges[fold] = (fd["date"].min(), fd["date"].max())
    n = len(fd)
    print(f"   Fold {fold}: {fd['date'].min().date()} → {fd['date'].max().date()} "
          f"| n={n:,} | signals={(fd['signal'] != 0).sum():,}")

gap_or_overlap = False
for i in range(1, len(folds)):
    prev_end   = fold_ranges[folds[i - 1]][1]
    curr_start = fold_ranges[folds[i]][0]
    gap_days   = (curr_start - prev_end).days
    if gap_days > 7:
        print(f"   ⚠ Gap between fold {folds[i-1]} and {folds[i]}: {gap_days} calendar days")
        gap_or_overlap = True
    elif gap_days < 0:
        print(f"   ⚠ Overlap between fold {folds[i-1]} and {folds[i]}: {-gap_days} days")
        gap_or_overlap = True

if not gap_or_overlap:
    print("   ✓ Folds are contiguous and non-overlapping")


# ── 4. Return sanity check ────────────────────────────────────────────────────

print(f"\n4. RETURN SANITY CHECK  (5d forward returns)")
if "fwd_ret_5d" in preds.columns:
    extreme = preds[preds["fwd_ret_5d"].abs() > 0.5]
    print(f"   Returns > ±50% in 5 days: {len(extreme)} ({len(extreme)/len(preds):.2%})")
    if len(extreme) > 0:
        print(extreme[["date", "ticker", "fwd_ret_5d"]].head(5).to_string(index=False))

    print(f"\n   fwd_ret_5d distribution:")
    d = preds["fwd_ret_5d"]
    print(f"   Mean  : {d.mean():.4f}   Std: {d.std():.4f}")
    print(f"   Min   : {d.min():.4f}   Max: {d.max():.4f}")
    print(f"   Skew  : {d.skew():.4f}   (negative skew expected for mean-reversion)")
else:
    print("   fwd_ret_5d not in predictions — re-run training.")


# ── 5. Per-fold performance ───────────────────────────────────────────────────
# Performance decay over time signals regime change or overfitting.

print(f"\n5. PERFORMANCE BY FOLD  (long signals, 5d return)")
has_spy21 = "spy_ret_5d" in preds.columns
long_signals = preds[preds["signal"] == 1]
for fold in folds:
    fs = long_signals[long_signals["fold"] == fold]
    if len(fs) == 0:
        print(f"   Fold {fold}: no long signals")
        continue
    if "fwd_ret_5d" not in fs.columns:
        print(f"   Fold {fold}: fwd_ret_5d missing")
        continue
    mean_ret = fs["fwd_ret_5d"].mean()
    win_rate = (fs["fwd_ret_5d"] > 0).mean()
    beat_str = ""
    if has_spy21:
        beat_spy = (fs["fwd_ret_5d"] > fs["spy_ret_5d"]).mean()
        beat_str = f" | beat_spy={beat_spy:.1%}"
    print(f"   Fold {fold}: n={len(fs):>4,} | ret={mean_ret:.3%} | wr={win_rate:.1%}{beat_str}")


# ── 6. VIX regime consistency ─────────────────────────────────────────────────
# Check that the VIX range seen during training covers the range at signal time.

print(f"\n6. VIX REGIME CONSISTENCY")
if "vix" in preds.columns:
    all_vix = preds["vix"].dropna()
    sig_vix = preds[preds["signal"] != 0]["vix"].dropna()
    print(f"   Training universe  — VIX mean: {all_vix.mean():.1f}  "
          f"range [{all_vix.min():.0f}, {all_vix.max():.0f}]")
    print(f"   Signals fired      — VIX mean: {sig_vix.mean():.1f}  "
          f"range [{sig_vix.min():.0f}, {sig_vix.max():.0f}]")
    pct_high     = (all_vix >= 25).mean()
    pct_sig_high = (sig_vix >= 25).mean()
    print(f"   VIX ≥ 25 in training: {pct_high:.1%}  |  in signals: {pct_sig_high:.1%}")
    if abs(pct_high - pct_sig_high) > 0.10:
        print("   ⚠ Signals skewed toward different VIX regime than training data")
    else:
        print("   ✓ VIX distribution of signals consistent with training data")
else:
    print("   VIX data not available in predictions.")


# ── 7. VIX regime performance ─────────────────────────────────────────────────

print(f"\n7. VIX REGIME PERFORMANCE  (all signals, 5d return)")
if "vix" in preds.columns and "fwd_ret_5d" in preds.columns:
    sig = preds[preds["signal"] != 0].copy()
    sig["direction"] = sig["signal"].astype(float)
    sig["adj_ret"]   = sig["fwd_ret_5d"] * sig["direction"]
    for label, mask in [
        ("VIX < 15  (calm)",     sig["vix"] < 15),
        ("VIX 15-20 (normal)",  (sig["vix"] >= 15) & (sig["vix"] < 20)),
        ("VIX 20-30 (elevated)", (sig["vix"] >= 20) & (sig["vix"] < 30)),
        ("VIX > 30  (fear)",     sig["vix"] >= 30),
    ]:
        subset = sig[mask]
        if len(subset) == 0:
            continue
        ret = subset["adj_ret"].mean()
        wr  = (subset["adj_ret"] > 0).mean()
        print(f"   {label}: n={len(subset):>5,} | dir-adj ret={ret:.3%} | wr={wr:.1%}")
else:
    print("   VIX or fwd_ret_5d data not available.")


# ── 8. Survivorship bias ──────────────────────────────────────────────────────

print(f"\n8. SURVIVORSHIP BIAS")
print(f"   ⚠ Universe = current S&P 500 constituents only.")
print(f"   Delisted stocks (often poor performers) are absent from training data.")
print(f"   Academic estimate: ~1-3% annualized return inflation.")
print(f"   Discount your annualized return estimate by ~2% to compensate.")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  AUDIT COMPLETE")
print(f"{'=' * 60}\n")
