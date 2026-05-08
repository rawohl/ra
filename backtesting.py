"""
S&P 500 Mean Reversion Signal System v3
Phase 4: Backtesting

Key design choices:
  - Overlapping-cohort equity curve: a signal entered on day T contributes
    net/HOLD_DAYS per day to the portfolio for each of the next HOLD_DAYS
    trading days, correctly modelling simultaneous open positions.
  - Position sizing via confidence weight applied once (in the weighted
    average across open positions), not baked into per-trade net return.
  - Trade-level win/loss uses the full 21d net return (gross - round-trip
    cost), which is the natural unit for a 21d holding strategy.
  - SPY benchmark and rolling Sharpe subplot for visual context.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging

from config import MIN_PROB, HOLD_DAYS, INITIAL_CAPITAL, ROUND_TRIP_COST

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _confidence_weight(conf: float) -> float:
    """
    Linear position-size scaler in [0.5, 1.5].
    conf = MIN_PROB  → weight 0.5 (half size)
    conf = 1.0       → weight 1.5 (one-and-a-half size)
    Expected mean ≈ 1.0 across typical signal distributions.
    """
    return max(0.5, 0.5 + (conf - MIN_PROB) / (1.0 - MIN_PROB))


def run_backtest(predictions: pd.DataFrame) -> dict:
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    if predictions["date"].dt.tz is not None:
        predictions["date"] = predictions["date"].dt.tz_convert(None)
    predictions = predictions.sort_values("date")

    signals = predictions[predictions["signal"] != 0].copy()
    if len(signals) == 0:
        log.error("No signals to backtest.")
        return {}

    n_long  = (signals["signal"] ==  1).sum()
    n_short = (signals["signal"] == -1).sum()
    log.info(f"Signals: {len(signals):,} across {signals['date'].nunique()} days  "
             f"({n_long:,} long  ·  {n_short:,} short)")

    # ── Per-trade records ─────────────────────────────────────────────────────
    # Net P&L for a trade = full 21d gross return (direction-adjusted) minus
    # the one-time round-trip transaction cost.  Weight is kept separate so it
    # only enters position sizing, not the win/loss determination.
    trade_rows = []
    for row in signals.itertuples(index=False):
        is_long   = row.signal == 1
        conf      = row.prob_up if is_long else 1.0 - row.prob_up
        weight    = _confidence_weight(conf)
        direction = 1 if is_long else -1
        net       = row.fwd_ret_5d * direction - ROUND_TRIP_COST
        trade_rows.append({
            "date":    row.date,
            "ticker":  row.ticker,
            "side":    "long" if is_long else "short",
            "prob_up": row.prob_up,
            "weight":  weight,
            "net":     net,
            "win":     net > 0,
        })

    trades_df = pd.DataFrame(trade_rows)

    # ── Overlapping-cohort equity curve ───────────────────────────────────────
    # A position entered on trading day T earns net/HOLD_DAYS per day for each
    # of the next HOLD_DAYS trading days.  Confidence weight controls how much
    # that position contributes to the portfolio's weighted daily return.
    # Accumulate into fixed-size arrays indexed by position in the date list.
    all_dates = np.array(sorted(predictions["date"].unique()))
    date_pos  = {d: i for i, d in enumerate(all_dates)}
    n_total   = len(all_dates)
    net_sum    = np.zeros(n_total)
    weight_sum = np.zeros(n_total)

    for row in trades_df.itertuples(index=False):
        start = date_pos.get(row.date, -1)
        if start < 0:
            continue
        end        = min(start + HOLD_DAYS, n_total)
        daily_net  = row.net / HOLD_DAYS
        net_sum[start:end]    += daily_net * row.weight
        weight_sum[start:end] += row.weight

    active = weight_sum > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_return = np.where(active, net_sum / weight_sum, np.nan)

    daily = pd.DataFrame({"date": all_dates, "daily_return": daily_return})
    daily = daily.dropna(subset=["daily_return"]).reset_index(drop=True)
    daily["equity"] = INITIAL_CAPITAL * (1 + daily["daily_return"]).cumprod()

    # Rolling 63-day (≈ 1 quarter) Sharpe ratio
    daily["rolling_sharpe"] = (
        daily["daily_return"].rolling(63).mean() * 252 /
        (daily["daily_return"].rolling(63).std() * np.sqrt(252) + 1e-9)
    )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    returns      = daily["daily_return"]
    total_return = daily["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    n_years      = len(daily) / 252
    ann_return   = (1 + total_return) ** (1 / max(n_years, 1e-9)) - 1
    sharpe       = returns.mean() * 252 / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    equity      = daily["equity"]
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())
    calmar      = ann_return / abs(max_dd) if max_dd < 0 else np.inf

    long_df  = trades_df[trades_df["side"] == "long"]
    short_df = trades_df[trades_df["side"] == "short"]
    wins     = trades_df[trades_df["win"]]
    losses   = trades_df[~trades_df["win"]]
    avg_win  = float(wins["net"].mean())  if len(wins)   > 0 else np.nan
    avg_loss = float(losses["net"].mean()) if len(losses) > 0 else np.nan
    pf       = abs(avg_win / avg_loss) if (avg_loss and avg_loss != 0) else np.inf

    metrics = {
        "total_return":      total_return,
        "annualized_return": ann_return,
        "sharpe_ratio":      sharpe,
        "calmar_ratio":      calmar,
        "max_drawdown":      max_dd,
        "win_rate":          trades_df["win"].mean(),
        "avg_win":           avg_win,
        "avg_loss":          avg_loss,
        "profit_factor":     pf,
        "total_trades":      len(trades_df),
        "long_trades":       len(long_df),
        "short_trades":      len(short_df),
        "long_win_rate":     long_df["win"].mean()  if len(long_df)  > 0 else np.nan,
        "short_win_rate":    short_df["win"].mean() if len(short_df) > 0 else np.nan,
        "trading_days":      len(daily),
        "signals_per_day":   len(trades_df) / max(trades_df["date"].nunique(), 1),
        "final_equity":      float(daily["equity"].iloc[-1]),
    }

    return {"metrics": metrics, "trades": trades_df, "daily": daily, "drawdown": drawdown}


def _fetch_spy_benchmark(start, end) -> pd.DataFrame | None:
    """Download SPY and return a daily equity curve scaled to INITIAL_CAPITAL."""
    try:
        spy = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )
        spy.index = pd.to_datetime(spy.index)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_convert(None)
        spy_ret = spy["Close"].pct_change().dropna()
        equity  = (INITIAL_CAPITAL * (1 + spy_ret).cumprod()).reset_index()
        equity.columns = ["date", "spy_equity"]
        return equity
    except Exception as e:
        log.warning(f"SPY benchmark fetch failed: {e}")
        return None


def plot_results(results: dict, save_path: Path = Path("backtest_results.png")) -> None:
    daily    = results["daily"]
    trades   = results["trades"]
    drawdown = results["drawdown"]

    spy_bench = _fetch_spy_benchmark(daily["date"].min(), daily["date"].max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes.flatten():
        ax.set_facecolor("#313244")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#89b4fa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    fig.suptitle("Mean Reversion Strategy — Backtest v3", color="#cdd6f4", fontsize=14)

    # ── Equity curve + SPY benchmark ─────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(daily["date"], daily["equity"], color="#89b4fa", linewidth=1.5, label="Strategy")
    if spy_bench is not None:
        ax1.plot(spy_bench["date"], spy_bench["spy_equity"],
                 color="#a6adc8", linewidth=1.0, linestyle="--", label="SPY buy-&-hold", alpha=0.8)
    ax1.axhline(INITIAL_CAPITAL, color="#45475a", linestyle=":", linewidth=0.8)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Value (€)", color="#cdd6f4")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(labelcolor="#cdd6f4", facecolor="#313244", framealpha=0.5, fontsize=9)
    ax1.grid(True, alpha=0.2, color="#585b70")

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.fill_between(daily["date"], drawdown * 100, 0, alpha=0.7, color="#f38ba8")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown %", color="#cdd6f4")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, alpha=0.2, color="#585b70")

    # ── Rolling 63-day Sharpe ─────────────────────────────────────────────────
    ax3 = axes[1, 0]
    rs = daily["rolling_sharpe"].dropna()
    rs_dates = daily.loc[rs.index, "date"]
    ax3.plot(rs_dates, rs, color="#a6e3a1", linewidth=1.2)
    ax3.fill_between(rs_dates, rs, 0,
                     where=(rs >= 0), alpha=0.15, color="#a6e3a1")
    ax3.fill_between(rs_dates, rs, 0,
                     where=(rs < 0),  alpha=0.15, color="#f38ba8")
    ax3.axhline(0, color="#585b70", linewidth=0.8, linestyle="--")
    ax3.axhline(1, color="#f9e2af", linewidth=0.8, linestyle=":", alpha=0.7)
    ax3.set_title("Rolling 63-Day Sharpe Ratio")
    ax3.set_ylabel("Sharpe", color="#cdd6f4")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.grid(True, alpha=0.2, color="#585b70")

    # ── Monthly returns heatmap ───────────────────────────────────────────────
    ax4 = axes[1, 1]
    dc = daily.copy()
    dc["year"]  = dc["date"].dt.year
    dc["month"] = dc["date"].dt.month
    monthly = dc.groupby(["year", "month"])["daily_return"].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()

    im = ax4.imshow(monthly.values * 100, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
                        fontsize=8, color="#cdd6f4")
    ax4.set_yticks(range(len(monthly.index)))
    ax4.set_yticklabels(monthly.index, fontsize=8, color="#cdd6f4")
    ax4.set_title("Monthly Returns (%)")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1e1e2e")
    log.info(f"Chart saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    pred_path = Path("data/clean/predictions.parquet")
    if not pred_path.exists():
        log.error("Run model_training.py first.")
        exit(1)

    predictions = pd.read_parquet(pred_path)
    results = run_backtest(predictions)

    if results:
        m = results["metrics"]
        print(f"\nTotal Return:      {m['total_return']:.2%}")
        print(f"Annualized:        {m['annualized_return']:.2%}")
        print(f"Sharpe:            {m['sharpe_ratio']:.3f}")
        print(f"Calmar:            {m['calmar_ratio']:.3f}")
        print(f"Max Drawdown:      {m['max_drawdown']:.2%}")
        print(f"Win Rate:          {m['win_rate']:.2%}")
        print(f"Profit Factor:     {m['profit_factor']:.3f}")
        print(f"Total Trades:      {m['total_trades']:,}")
        print(f"Final Equity:      €{m['final_equity']:,.2f}")
        plot_results(results)
