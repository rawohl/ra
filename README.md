# ra: s&p 500 mean reversion signals

a market-neutral signal system for the s&p 500. it predicts which stocks are likely to outperform or underperform their peers over the next 21 trading days and takes long and short positions accordingly. trained and validated using walk-forward cross-validation on roughly 2.5 years of daily data across ~500 stocks.

---

## what it does

every morning before market open, ra scans the s&p 500 and outputs a ranked list of long and short candidates. longs are stocks the model expects to land in the top 30% of 21-day relative performers. shorts are expected to land in the bottom 30%. positions are sized by model confidence and held for approximately one month.

the strategy is market-neutral: it doesn't bet on the market going up or down, only on which stocks will do better or worse than their peers.

---

## approach

### target

the model predicts whether a stock will rank in the **top or bottom 30%** of the s&p 500 by excess return over spy in the next 21 days. the middle 40% is dropped from training because those stocks are too close to the median to give the model a clean signal. this gives roughly balanced classes without artificially reweighting.

a 21-day horizon was chosen deliberately. 5-day targets produce near-random labels because short-term momentum drowns out mean reversion. at one month, the reversion effect becomes detectable.

### model

lightgbm binary classifier. no neural networks; tree models handle tabular financial data well and are much easier to interpret. key hyperparameters were set conservatively (`min_child_samples=50`, `learning_rate=0.02`) to limit overfitting on the relatively small per-fold training sets.

### validation

8-fold walk-forward cross-validation: each fold trains on 2 years of history and tests on the next 3 months, stepping forward in time. random splits leak future data into training, so this is the only valid approach for time-series models.

signal threshold of 0.52: stocks with predicted `prob_up >= 0.52` are longs, stocks with `prob_up <= 0.48` are shorts. narrow by design (see results).

### features (~45 total)

per-stock technical indicators: rsi-14, 20/60-day z-score, distance from 20/50/200-day moving averages, bollinger band position, normalized atr, volume ratio, 52-week high/low distance, 5/20-day returns, consecutive up/down days, overnight gap.

cross-sectional rank features: for each indicator, the stock's rank within the full s&p 500 on that date. the most oversold stock today is a stronger signal than a raw rsi value alone.

market context: vix, cross-sectional return dispersion (how much stocks are moving independently vs. in lockstep), sector-relative z-score.

---

## what the model learned

shap values were computed on the final fold's test set after training to see which features actually drove predictions.

![shap feature importance](models/shap_analysis.png)

top features by mean absolute shap value:

| feature | what it measures | direction |
|---------|-----------------|-----------|
| `dist_52w_low` | how far above the 52-week low | stocks near 52w low predicted to outperform (mean reversion) |
| `xs_disp_20d` | 20-day avg cross-sectional dispersion | high dispersion = model more willing to signal; low = macro-driven market |
| `dist_52w_high_xs_rank` | cross-sectional rank of distance from 52w high | stocks far from their highs tend to get shorted |
| `dist_ma200` | distance from 200-day moving average | below 200ma predicted to recover |
| `vix` | fear index | high vix periods produce stronger signals |

the 52-week low distance being the top feature is the mean reversion thesis showing up directly in the data. the dispersion feature makes sense too: when everything is moving in lockstep (a macro shock), stock-specific signals stop working and the model picks up on that.

rsi ranked near the bottom despite being one of the most commonly used indicators. it's overcrowded and largely priced in.

---

## results

### walk-forward

```
 fold  period                  auc     long prec  short prec  long rate  short rate
    1  apr-jul 2024          0.540       53.3%      53.2%       38.4%      40.7%
    2  jul-oct 2024          0.541       53.8%      53.7%       46.7%      32.0%
    3  oct 2024-jan 2025     0.506       50.4%      51.7%       42.5%      28.6%
    4  jan-apr 2025          0.487       48.2%      50.9%       41.2%      20.5%
    5  apr-jul 2025          0.590       46.0%      67.8%        2.0%       1.2%
    6  jul-oct 2025          0.552        n/a        n/a         0.0%       0.0%
    7  oct 2025-jan 2026     0.521       52.5%      50.9%       39.6%      27.4%
    8  jan-feb 2026          0.533       53.7%      53.7%       28.5%      22.6%

  mean auc           0.534
  mean long prec     51.1%
  mean short prec    54.6%
```

**fold 4 (jan-apr 2025):** long precision dropped to 48.2%, below chance. this period covers the trump tariff shock. mean reversion fails in strong downtrends: stocks near 52-week lows kept falling rather than recovering. the short leg held (50.9%) because downward momentum is more persistent.

**fold 5 (apr-jul 2025):** best auc at 0.590 but signal rates collapsed to ~2%. the model got very selective and short precision hit 67.8% on that small sample. the post-tariff environment had genuine asymmetric setups but the model was cautious about entering broadly.

**fold 6:** zero signals. cross-sectional dispersion was in the bottom quintile of its history, stocks were all moving together, and the model correctly produced no output rather than trading noise.

**takeaway:** shorts are consistently more reliable than longs. mean precision 54.6% vs 51.1% across 8 folds. underperformance tends to be stickier than outperformance and the model gets fewer false signals on the short side.

### backtest

market-neutral long/short strategy. positions are pro-rated daily (1/21 of the 21-day return per day) and sized by confidence. 0.1% round-trip transaction cost applied per trade.

```
  total return          5.58%
  annualized            3.74%
  sharpe ratio          2.57
  calmar ratio          0.79
  max drawdown         -4.73%
  win rate             50.81%
  profit factor         1.046
  total trades         71,782
    long / short     41,756 / 30,026
  long win rate        50.87%
  short win rate       50.73%
  signals / day          192
```

sharpe of 2.57 is the headline number. most actively managed funds target above 1. the max drawdown of -4.73% is small given the number of positions.

3.74% annualized sounds low but this is pure alpha with no net market exposure. it wouldn't be affected by a 20% market crash unless correlations break down the way they did in fold 4. adding leverage to a strategy with sharpe above 2 and controlled drawdown is how the return becomes meaningful in practice.

the backtest does not model market impact. at the scale tested (10,000 euros across ~500 stocks) individual positions are small enough that impact is negligible, but this would need revisiting at larger scale.

---

## known limitations

- **downtrend longs:** the long leg underperforms in sustained sell-offs (fold 4). suppressing longs when spy is below its 200-day moving average would help. the short leg works fine in those conditions.
- **data source:** uses yfinance, which is an unofficial scraper and occasionally breaks. migration to polygon.io is planned.
- **sample size:** 8 folds over roughly 2 years is meaningful but not conclusive. the tariff shock was one macro event and it weighted the long leg results heavily. more regime diversity in the data would give a cleaner picture.
- **no earnings filter:** stocks approaching earnings are driven by event risk, not mean reversion. filtering the 3 days before announcements would reduce noise.

---

## how to run

```bash
pip install -r requirements.txt

# step 1: download data and build features (~10 min first run)
python feature_engineering.py

# step 2: train model + shap analysis
python model_training.py

# step 3: backtest on walk-forward predictions
python backtesting.py

# step 4: open the gui
python main.py

# or use the cli
python main.py --no-gui          # interactive repl
python main.py --no-gui signals  # one-shot signal output
```

---

## files

```
data_pipeline.py       s&p 500 universe + ohlcv download
feature_engineering.py technical indicator computation + cross-sectional features
model_training.py      walk-forward training + shap analysis
backtesting.py         position sizing + p&l simulation
signal_generator.py    live signal generation (runs daily)
main.py                tkinter gui
cli.py                 command-line interface
models/model.pkl       trained model (last fold)
models/shap_analysis.png  feature importance chart
```
