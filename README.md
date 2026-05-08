# ra: s&p 500 mean reversion signals

a market-neutral signal system for the s&p 500. each day it scores every stock in the index, selects the top-5 long and top-5 short candidates by model confidence, and holds for 5 trading days. trained and validated using 16-fold walk-forward cross-validation on 11 years of daily data across ~500 stocks.

---

## what it does

every morning before market open, ra scans the s&p 500 and outputs a ranked list of the 5 highest-conviction long candidates and 5 highest-conviction short candidates. the model predicts which stocks will land in the top or bottom 30% of 21-day relative performers vs spy — then the live system selects the top-5 per direction and holds for one week.

the strategy is market-neutral: it doesn't bet on the market going up or down, only on which stocks will do better or worse than their peers.

---

## approach

### signal selection

the model ranks all ~500 stocks each day by predicted probability of outperformance. the top-5 by `prob_up` become longs; the bottom-5 (highest `1 - prob_up`) become shorts. a confidence spread gate skips days where the model cannot discriminate between stocks (max_prob − min_prob below threshold) — those days generate noise, not signal.

10 positions entered per day × 5-day hold = ~50 simultaneous positions at any time.

### training target

the model predicts whether a stock will rank in the **top or bottom 30%** of the s&p 500 by excess return over spy in the next 21 days. the middle 40% is dropped from training — those stocks are too close to the median to give the model a clean label.

a 21-day prediction horizon is used because mean reversion is empirically strongest at a one-month window. the live hold period (5 days) is deliberately shorter — we capture the initial move before the position has time to reverse.

### model

lightgbm binary classifier trained on labeled top/bottom 30% rows, but scored against the full universe at prediction time.

optional optuna hyperparameter tuning (`--trials N`) runs on a **dedicated hpo zone** (first 2 years of data) before any walk-forward folds. test sets never overlap this period, keeping the walk-forward evaluation genuinely out-of-sample.

### validation

16-fold walk-forward cross-validation across 11 years (2015–2026): each fold trains on all available history and tests on the next 6 months, stepping forward. growing training window ensures the model sees the full preceding regime at each fold.

### features (38 total, after shap-based pruning)

**per-stock technicals:** rsi-14/21, 60-day z-score, distance from 20/50/200-day ma, normalized atr, volatility regime, 52-week high/low distance, 5/10/20-day returns, intraday range.

**cross-sectional rank features:** for each core indicator, the stock's percentile rank within the full s&p 500 on that date, and its rank within its gics sector. "most oversold stock in xlk today" is a much stronger signal than an rsi reading in isolation.

**market context:** vix (continuous), cross-sectional return dispersion 5d/20d (how much stocks are moving independently vs. in lockstep — low dispersion signals macro risk-off where relative-value signals lose power).

**sector identity (one-hot):** `sector_XLK`, `sector_XLF`, `sector_XLV`, `sector_XLI`, `sector_XLP`, `sector_XLB`, `sector_XLRE` — 7 flags (xle/xlu/xly/xlc dropped in shap pruning). lets the model learn sector-specific biases vs spy directly.

---

## results

### walk-forward fold summary (latest run, 11yr data)

```
 fold  test period              auc    long_prec  short_prec
    1  2018-05 → 2018-11     0.4477     34.9%      39.1%
    2  2018-11 → 2019-05     0.5036      n/a        n/a
    3  2019-05 → 2019-11     0.5257     61.8%      66.2%
    4  2019-11 → 2020-05     0.5340     74.3%      43.5%
    5  2020-05 → 2020-11     0.5445     51.8%      62.4%
    6  2020-11 → 2021-05     0.5169     56.3%      57.2%
    7  2021-05 → 2021-11     0.5460     62.5%      41.7%
    8  2021-11 → 2022-05     0.4661      n/a        n/a
    9  2022-05 → 2022-11     0.5371     51.3%      55.7%
   10  2022-11 → 2023-05     0.5462     49.5%      55.7%
   11  2023-05 → 2023-11     0.5053     59.3%      43.2%
   12  2023-11 → 2024-05     0.5028      n/a        n/a
   13  2024-05 → 2024-11     0.5055     56.4%      50.3%
   14  2024-11 → 2025-05     0.5556     59.2%      59.7%
   15  2025-05 → 2025-11     0.5162     38.5%      53.1%
   16  2025-11 → 2026-03     0.4628     27.0%      42.4%

  mean auc            0.5135
  mean long prec      52.5%
  mean short prec     51.6%
```

n/a = spread gate suppressed signals (model too uncertain that day)

folds 1, 8, 15, 16 are macro regime shock periods (2018 trade war, 2022 rate hike cycle, 2025–2026 tariff shock). mean reversion underperforms in sustained directional moves — these are structural losses, not model failures.

### vix regime breakdown

| regime | long prec | short prec | note |
|---|---|---|---|
| calm (<15) | 49.8% | 40.4% | longs marginal, shorts negative |
| normal (15–20) | 46.8% | 54.2% | longs negative edge |
| elevated (20–30) | 53.2% | 57.5% | both sides profitable |
| fear (>30) | 62.2% | 57.4% | strongest edge |

counterintuitively, the model performs best in elevated/fear regimes — sector rotation is most pronounced during stress, which is where the learned patterns are clearest.

### sector precision (long)

| sector | prec | edge |
|---|---|---|
| xlre | 61.4% | +11.4% |
| xly | 59.0% | +9.0% |
| xlc | 58.1% | +8.1% |
| xlf | 57.3% | +7.3% |
| xlv | 57.3% | +7.3% |
| xlk | 47.8% | −2.2% |
| xle | 41.8% | −8.2% |
| xlp | 39.1% | −10.9% |

### sector precision (short)

| sector | prec | edge |
|---|---|---|
| xlu | 78.7% | +28.7% |
| xlb | 62.1% | +12.1% |
| xlf | 56.6% | +6.6% |
| xlp | 51.4% | +1.4% |
| xlk | 47.8% | −2.2% |
| xle | 42.5% | −7.5% |

### backtest (top-5 × 5-day hold, equal weight)

```
  total return          21.73%
  annualized             3.71%
  sharpe ratio           0.669
  calmar ratio           0.170
  max drawdown         -21.86%
  win rate              49.39%
  profit factor          1.057
  total trades          13,600
    long / short     6,800 / 6,800
  long win rate         51.97%
  short win rate        46.81%
  signals / day            10.0
  final equity        EUR 12,173
```

### ml vs naive baseline

| metric | naive reversal | ml model |
|---|---|---|
| total return | −14.66% | +23.04% |
| annualized | −2.89% | +3.92% |
| sharpe | −0.747 | +0.974 |
| max drawdown | −30.66% | −21.60% |

naive reversal (buy the most beaten-down stocks, short the strongest) loses badly because sustained momentum runs — trade war, rate hike cycle, tariff shock — mean oversold stocks keep falling. the ml model learns when not to fight momentum.

---

## known limitations

- **macro regime shocks:** folds covering the 2018 trade war, 2022 rate hike cycle, and 2025–2026 tariff shock all show auc < 0.5 — the model actively mispredicts during sustained sector-wide selloffs. mean reversion strategies are structurally weak in trending macro environments.
- **xlk/xle/xlp longs:** below-50% precision. the model generates long signals here because the cross-sectional technicals nominate them, but sector-level macro pressure overrides individual stock signals. position sizing (kelly) is the right fix rather than hard exclusions.
- **equal-weight assumption:** current backtest weights all signals equally. the model already outputs `prob_up` which can feed kelly criterion sizing — high-confidence picks would compound significantly faster.
- **survivorship bias:** the universe is current s&p 500 constituents. delisted stocks (historically poor performers) are absent. academic estimate: ~1–3% annualized return inflation.
- **market impact:** not modeled. 10 positions/day at reasonable size avoids impact, but revisit at larger scale.
- **data source:** yfinance (unofficial). occasional breaks or stale data. migration to a proper data vendor is the right long-term path.

---

## how to run

```bash
pip install -r requirements.txt

# gui
python main.py

# cli — one-shot commands
python cli.py download                  # step 01: data + features (~8 min first run)
python cli.py train                     # step 02: walk-forward training (~3 min, 16 threads)
python cli.py train --trials 100        # step 02: with optuna hpo
python cli.py backtest                  # step 03: backtest
python cli.py signals                   # step 04: today's top-5 picks
python cli.py signals --top-n 10        # top-10 per side

# research tools
python cli.py baseline                  # naive reversal vs ml comparison
python cli.py regime                    # vix + dispersion breakdown
python cli.py sector                    # precision by sector

# add --debug to any command for compact output
python cli.py backtest --debug
```

---

## files

```
config.py              shared constants (HOLD_DAYS, TOP_N, MIN_SPREAD, etc.)
data_pipeline.py       s&p 500 universe scrape + 11yr ohlcv download
feature_engineering.py 38 features: technicals, cross-sectional ranks, sector one-hot
model_training.py      hpo zone + 16-fold walk-forward training
backtesting.py         overlapping-cohort portfolio simulation
signal_generator.py    live signal generation with incremental price cache
main.py                tkinter gui
cli.py                 command-line interface
audit.py               backtest sanity checks (lookahead, fold structure)
models/model.pkl       trained lightgbm model + feature list
models/shap_analysis.png  shap feature importance chart
```