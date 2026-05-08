# ra: s&p 500 mean reversion signals

a market-neutral signal system for the s&p 500. each day it scores every stock in the index, selects the top-5 long and top-5 short candidates by model confidence, and holds for 5 trading days. trained and validated using walk-forward cross-validation on ~3 years of daily data across ~500 stocks.

---

## what it does

every morning before market open, ra scans the s&p 500 and outputs a ranked list of the 5 highest-conviction long candidates and 5 highest-conviction short candidates. the model predicts which stocks will land in the top or bottom 30% of 21-day relative performers vs spy — then the live system selects the top-5 per direction and holds for one week.

the strategy is market-neutral: it doesn't bet on the market going up or down, only on which stocks will do better or worse than their peers.

---

## approach

### signal selection

the model ranks all ~500 stocks each day by predicted probability of outperformance. the top-5 by `prob_up` become longs; the bottom-5 (highest `1 - prob_up`) become shorts. this is how quant funds actually operate — rank the universe and trade the extremes — rather than firing every stock that crosses an arbitrary confidence threshold.

10 positions entered per day × 5-day hold = ~50 simultaneous positions at any time. manageable for a real portfolio.

### training target

the model predicts whether a stock will rank in the **top or bottom 30%** of the s&p 500 by excess return over spy in the next 21 days. the middle 40% is dropped from training — those stocks are too close to the median to give the model a clean signal, so including them would add noise without adding label quality.

a 21-day prediction horizon is used because mean reversion is empirically strongest at a one-month window. the live hold period (5 days) is deliberately shorter — we capture the initial move before the position has time to reverse.

### model

lightgbm binary classifier trained on labeled top/bottom 30% rows, but scored against the full universe at prediction time. optional optuna hyperparameter tuning (`--trials N`) runs before the walk-forward folds using only the first fold's training data to keep the search time-aware.

### validation

8-fold walk-forward cross-validation: each fold trains on 2 years of history and tests on the next 3 months, stepping forward in time. random splits leak future data into training — walk-forward is the only valid approach for time-series.

### features (~55 total)

**per-stock technicals:** rsi-7/14/21, 10/20/60-day z-score, distance from 20/50/200-day ma, bollinger band position (10/20-day), normalized atr, volume ratio, volume regime, 52-week high/low distance, 1/2/5/10/20-day returns, consecutive up/down days, intraday range, overnight gap.

**cross-sectional rank features:** for each core indicator, the stock's percentile rank within the full s&p 500 on that date, and its rank within its gics sector. "most oversold stock in xlk today" is a much stronger signal than an rsi reading in isolation.

**market context:** vix, vix regime flags (vix_low/normal/high/fear — four buckets, not three), cross-sectional return dispersion (how much stocks are moving independently vs. in lockstep).

**sector identity (one-hot):** `sector_XLK`, `sector_XLF`, ..., `sector_XLC` — 11 binary flags, one per gics sector etf. lets the model learn sector-specific biases vs spy directly (e.g. xlre longs consistently underperform in high-rate environments even when technically oversold). without these flags the model treats an oversold xlre stock identically to an oversold xlk stock.

---

## results

### walk-forward fold summary (latest run)

```
 fold  train_size  test_size      auc  long_prec  short_prec  long_rate  short_rate
    1      148981      18837 0.476333   0.526882    0.555556   0.004937    0.000956
    2      167818      19435 0.545009   0.563107    0.218182   0.037098    0.002830
    3      187253      18239 0.490742   0.488003    0.478506   0.317616    0.155601
    4      205492      18293 0.536695   0.553287    0.575864   0.195430    0.202482
    5      223785      18963 0.589494   0.634349    0.597504   0.038074    0.033803
    6      242748      19565 0.564028   0.568855    0.552635   0.334781    0.383082
    7      262313      18662 0.512290   0.522795    0.523179   0.125764    0.016183
    8      280975       8127 0.542788   0.533409    0.478125   0.108650    0.078750

  mean auc            0.5322
  mean long prec      54.9%
  mean short prec     49.7%
```

### vix regime breakdown

| regime | signals | long prec | short prec |
|---|---|---|---|
| calm (<15) | 7,235 | 53.3% | 54.4% |
| normal (15–20) | 19,434 | 53.8% | 53.2% |
| elevated (20–30) | 7,553 | 50.4% | 48.6% |
| **fear (>30)** | 2,155 | **69.5%** | **80.4%** |

the fear regime is where most of the real edge lives. extreme vix creates the strongest mean reversion setups. the elevated (20–30) regime is effectively noise for the short side.

### sector precision (long)

| sector | prec | edge |
|---|---|---|
| xlu | 65.5% | +15.5% |
| xlk | 58.1% | +8.1% |
| xlp | 55.9% | +5.9% |
| xlb | 46.2% | -3.8% |
| xlre | 32.4% | -17.6% |

### sector precision (short)

| sector | prec | edge |
|---|---|---|
| xlp | 72.4% | +22.4% |
| xlc | 67.9% | +17.9% |
| xlre | 64.9% | +14.9% |
| xle | 43.5% | -6.5% |
| xlk | 46.4% | -3.6% |

### backtest (top-5 × 5-day hold)

```
  total return          23.54%
  annualized            12.60%
  sharpe ratio           4.505
  calmar ratio           1.770
  max drawdown          -7.12%
  win rate              51.65%
  profit factor          1.048
  total trades          36,377
    long / short    20,684 / 15,693
  long win rate         50.83%
  short win rate        52.72%
  signals / day            86.4
  final equity       EUR 12,354
```

### ml vs naive baseline

| metric | naive reversal | ml model |
|---|---|---|
| total return | -12.39% | +23.54% |
| annualized | -6.89% | +12.60% |
| sharpe | -4.681 | +4.505 |
| win rate | 47.92% | 51.65% |

naive reversal (buy bottom-30% losers, short top-30% winners) loses badly over 2022–2026 because sustained momentum runs (ai/tech) crush simple mean-reversion shorts. the ml model learns when not to fight momentum.

---

## known limitations

- **xlre longs:** 32.4% precision — well below chance. real estate performance vs spy is driven by interest rate expectations, not technical indicators. the sector one-hot flag teaches the model about this, but more training data across different rate regimes is needed.
- **short side in elevated vix (20–30):** 48.6% precision — slightly net wrong. the model's short edge lives almost entirely in the fear (>30) regime. the model is still learning to use the split vix_high/vix_fear features.
- **survivorship bias:** the universe is current s&p 500 constituents. delisted stocks (historically poor performers) are absent. academic estimate: ~1–3% annualized return inflation. discount accordingly.
- **market impact:** not modeled. 10 positions/day at reasonable size avoids impact, but revisit at larger scale.
- **data source:** yfinance (unofficial scraper). occasional breaks. migration to a proper data vendor is the right long-term path.

---

## how to run

```bash
pip install -r requirements.txt

# gui
python main.py

# cli — interactive repl
python main.py --no-gui

# cli — one-shot commands
python main.py --no-gui download              # step 01: data + features
python main.py --no-gui train                # step 02: walk-forward training
python main.py --no-gui train --trials 50    # step 02: with optuna hpo
python main.py --no-gui backtest             # step 03: backtest
python main.py --no-gui signals              # step 04: today's top-5 picks
python main.py --no-gui signals --top-n 10  # top-10 per side instead

# taha tools (research & diagnostics)
python main.py --no-gui baseline             # naive reversal vs ml
python main.py --no-gui regime               # vix + dispersion breakdown
python main.py --no-gui sector               # precision by sector

# add --debug to any command for compact machine-readable output
python main.py --no-gui backtest --debug
```

---

## files

```
config.py              shared constants (HOLD_DAYS, TOP_N, ROUND_TRIP_COST, etc.)
data_pipeline.py       s&p 500 universe + ohlcv download
feature_engineering.py technical indicators + cross-sectional + sector one-hot features
model_training.py      walk-forward training: trains on top/bottom 30%, predicts on full universe
backtesting.py         top-n portfolio simulation with overlapping 5-day positions
signal_generator.py    live signal generation — top-n selection from full universe
main.py                tkinter gui
cli.py                 command-line interface (repl + one-liner mode)
audit.py               backtest sanity checks (lookahead, fold structure, return distribution)
models/model.pkl       trained lightgbm model + feature list
models/shap_analysis.png  feature importance chart (last fold)
```
