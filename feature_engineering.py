"""
S&P 500 Mean Reversion Signal System v3
Phase 2: Feature Engineering

v3 changes:
  - GICS sector mapping replaces correlation-based sector assignment:
    each stock is matched to its actual sector ETF (XLK, XLF, etc.)
  - Fallback to best-correlated ETF if sector_etf column is missing
  - Consistent tz_convert(None) for all date handling
  - VIX sourced without vintage_date (returns current published values)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import logging
import requests
from io import StringIO

log = logging.getLogger(__name__)

ALL_SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def _to_naive(s: pd.Series) -> pd.Series:
    """
    Strip timezone and normalize to midnight datetime64[ns].

    tz_localize(None) strips the tz label WITHOUT shifting the time value.
    tz_convert(None) would convert to UTC first — yfinance America/New_York
    midnight (00:00-05:00) would become 05:00 UTC, mismatching VIX/stock dates
    which are stored as midnight. We want the date, not the UTC equivalent.
    """
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)   # strip tz, keep local midnight value
    return s.dt.normalize().astype("datetime64[ns]")


def get_market_context(start: str, end: str) -> pd.DataFrame:
    ctx = pd.DataFrame()

    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", timeout=10)
        vix = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"],
                          index_col="observation_date")
        vix.index = _to_naive(vix.index.to_series()).values
        vix.index = pd.to_datetime(vix.index)
        vix = vix[(vix.index >= pd.Timestamp(start)) & (vix.index <= pd.Timestamp(end))]
        vix["VIXCLS"] = pd.to_numeric(vix["VIXCLS"], errors="coerce")
        ctx["vix"] = vix["VIXCLS"]
        log.info(f"VIX loaded: {len(vix)} days, range {vix['VIXCLS'].min():.1f}–{vix['VIXCLS'].max():.1f}")
    except Exception as e:
        log.warning(f"VIX FRED failed: {e}. Using default 20.")
        ctx["vix"] = 20.0

    try:
        spy = yf.Ticker("SPY").history(start=start, end=end)
        spy.index = _to_naive(spy.index.to_series()).values
        spy.index = pd.to_datetime(spy.index)
        ctx["spy_ret"]     = spy["Close"].pct_change()
        ctx["spy_ret_5d"]  = spy["Close"].pct_change(5).shift(-5)
        ctx["spy_ret_21d"] = spy["Close"].pct_change(21).shift(-21)
        log.info(f"SPY loaded: {len(spy)} days")
    except Exception as e:
        log.error(
            f"SPY download failed: {e}  —  spy_ret_21d will be NaN for all dates. "
            "Training will fail. Check network and re-run step 01."
        )
        ctx["spy_ret"]     = 0.0
        ctx["spy_ret_5d"]  = np.nan
        ctx["spy_ret_21d"] = np.nan

    return ctx.dropna(how="all")


def get_sector_returns(start: str, end: str) -> pd.DataFrame:
    """Download all sector ETF daily returns."""
    log.info("Downloading sector ETFs...")
    raw = yf.download(ALL_SECTOR_ETFS, start=start, end=end,
                      auto_adjust=True, progress=False, group_by="ticker")
    out = pd.DataFrame()
    for etf in ALL_SECTOR_ETFS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw[etf]["Close"]
            else:
                prices = raw["Close"]
            prices.index = _to_naive(prices.index.to_series()).values
            prices.index = pd.to_datetime(prices.index)
            out[etf] = prices.pct_change()
        except Exception:
            continue
    return out


# ── Technical indicator functions ─────────────────────────────────────────────

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))


def bollinger_position(series, period=20, num_std=2.0):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - (ma - num_std * std)) / (2 * num_std * std).replace(0, np.nan)


def zscore(series, period):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - ma) / std.replace(0, np.nan)


def dist_from_ma(series, period):
    ma = series.rolling(period).mean()
    return (series - ma) / ma.replace(0, np.nan)


def volume_ratio(volume, period=20):
    avg = volume.rolling(period).mean()
    return volume / avg.replace(0, np.nan)


def atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def vol_regime(close, short=10, long=60):
    ret = close.pct_change()
    return ret.rolling(short).std() / ret.rolling(long).std().replace(0, np.nan)


def consec_days(close, direction="down"):
    cond = close.pct_change().lt(0) if direction == "down" else close.pct_change().gt(0)
    groups = (cond != cond.shift()).cumsum()
    return (cond.groupby(groups).cumcount() + 1) * cond.astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    c = df["Close"]
    h, l = df["High"], df["Low"]
    v = df["Volume"]
    o = df["Open"]

    df["rsi_7"]  = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)
    df["rsi_21"] = rsi(c, 21)

    df["bb_pos_10"] = bollinger_position(c, 10)
    df["bb_pos_20"] = bollinger_position(c, 20)

    df["zscore_10"] = zscore(c, 10)
    df["zscore_20"] = zscore(c, 20)
    df["zscore_60"] = zscore(c, 60)

    df["dist_ma20"]  = dist_from_ma(c, 20)
    df["dist_ma50"]  = dist_from_ma(c, 50)
    df["dist_ma200"] = dist_from_ma(c, 200)

    df["vol_ratio_10"] = volume_ratio(v, 10)
    df["vol_ratio_20"] = volume_ratio(v, 20)
    df["down_volume"]  = df["vol_ratio_20"] * (c.pct_change() < 0).astype(float)

    df["atr_norm"]   = atr(h, l, c, 14) / c
    df["vol_regime"] = vol_regime(c)

    df["gap"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)

    df["consec_down"] = consec_days(c, "down")
    df["consec_up"]   = consec_days(c, "up")

    df["dist_52w_high"] = (c / c.rolling(252).max()) - 1
    df["dist_52w_low"]  = (c / c.rolling(252).min()) - 1

    df["intraday_range"] = (h - l) / c

    return df


def build_features_all(master: pd.DataFrame) -> pd.DataFrame:
    log.info("Building features for all tickers...")

    master = master.copy()
    master["date"] = _to_naive(master["date"])

    start = (master["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end   = (master["date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    ctx         = get_market_context(start, end)
    sector_rets = get_sector_returns(start, end)

    has_sector_col = "sector_etf" in master.columns

    master = master.set_index("date")
    frames = []
    tickers = master["ticker"].unique()

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            log.info(f"  {i}/{len(tickers)} tickers...")
        tdf = master[master["ticker"] == ticker].copy()
        tdf = build_features(tdf)
        frames.append(tdf)

    result = pd.concat(frames).reset_index()
    result["date"] = _to_naive(result["date"])

    # Merge VIX and SPY context
    ctx_reset = ctx.copy()
    ctx_reset.index.name = "date"
    ctx_reset = ctx_reset.reset_index()
    # Pandas 2.x: read_csv parse_dates returns datetime64[us], yfinance returns datetime64[ns].
    # Mismatched units cause a silent left-join failure (all NaN on right side).
    result["date"]      = result["date"].astype("datetime64[ns]")
    ctx_reset["date"]   = ctx_reset["date"].astype("datetime64[ns]")
    result = result.merge(ctx_reset, on="date", how="left")

    if result["spy_ret_21d"].isna().all():
        raise RuntimeError(
            "spy_ret_21d is entirely NaN — SPY download failed during feature engineering. "
            "Check your internet connection and re-run step 01."
        )

    # Raw VIX dominates the regime flag bins in SHAP — keep only the continuous value.

    # Sector-relative z-score
    log.info("Computing sector-relative features...")
    result["sector_rel_zscore"] = np.nan

    for ticker in tickers:
        tmask = result["ticker"] == ticker
        tdf   = result[tmask].set_index("date")
        stock_ret = tdf["ret_5d"].dropna()

        if len(stock_ret) < 60:
            continue

        # Use GICS-assigned sector ETF if available; fall back to best correlation
        sector_etf = None
        if has_sector_col and not tdf["sector_etf"].isna().all():
            candidate = tdf["sector_etf"].iloc[0]
            if candidate in sector_rets.columns:
                sector_etf = candidate

        if sector_etf is None:
            best_corr = -1
            for etf in sector_rets.columns:
                etf_aligned = sector_rets[etf].reindex(stock_ret.index)
                if etf_aligned.dropna().__len__() < 60:
                    continue
                corr = stock_ret.corr(etf_aligned)
                if not np.isnan(corr) and corr > best_corr:
                    best_corr, sector_etf = corr, etf

        if sector_etf is None:
            continue

        sector_5d  = sector_rets[sector_etf].rolling(5).sum()
        rel_ret    = tdf["ret_5d"] - sector_5d.reindex(tdf.index)
        rel_zscore = zscore(rel_ret, 20)
        result.loc[tmask, "sector_rel_zscore"] = rel_zscore.values

    log.info("Sector features done.")

    # Cross-sectional rank features: absolute RSI tells you little;
    # "most oversold stock in XLK today" is the actual signal.
    # Sector one-hot: lets the model learn sector-specific biases (e.g. XLRE
    # longs fail in rising-rate environments even when technically oversold).
    log.info("Computing sector one-hot features...")
    if "sector_etf" in result.columns:
        for etf in ALL_SECTOR_ETFS:
            result[f"sector_{etf}"] = (result["sector_etf"] == etf).astype(float)
    else:
        for etf in ALL_SECTOR_ETFS:
            result[f"sector_{etf}"] = 0.0

    # Sector vs market: how is each sector ETF performing relative to SPY?
    # This gives the model the middle layer of the relative-value hierarchy:
    # market → sector vs market → stock vs sector (sector_rel_zscore)
    # Without this, the model can't distinguish "XLK stock beaten down in a healthy
    # sector" from "XLK stock beaten down because the whole sector is under macro pressure."
    log.info("Computing sector vs SPY relative return features...")
    if "spy_ret" in ctx.columns and "sector_etf" in result.columns:
        spy_ret_s = ctx["spy_ret"].copy()
        spy_ret_s.index = spy_ret_s.index.astype("datetime64[ns]")
        svs_frames = []
        for etf in ALL_SECTOR_ETFS:
            if etf not in sector_rets.columns:
                continue
            sec = sector_rets[etf].copy()
            sec.index = sec.index.astype("datetime64[ns]")
            spy_aligned = spy_ret_s.reindex(sec.index)
            s20 = sec.rolling(20).sum() - spy_aligned.rolling(20).sum()
            s60 = sec.rolling(60).sum() - spy_aligned.rolling(60).sum()
            frame = pd.DataFrame({"sector_vs_spy_20d": s20, "sector_vs_spy_60d": s60})
            frame.index.name = "date"
            frame = frame.reset_index()
            frame["sector_etf"] = etf
            svs_frames.append(frame)
        if svs_frames:
            svs_df = pd.concat(svs_frames, ignore_index=True)
            svs_df["date"] = svs_df["date"].astype("datetime64[ns]")
            result = result.merge(svs_df, on=["date", "sector_etf"], how="left")
        else:
            result["sector_vs_spy_20d"] = 0.0
            result["sector_vs_spy_60d"] = 0.0
    else:
        result["sector_vs_spy_20d"] = 0.0
        result["sector_vs_spy_60d"] = 0.0
    log.info("Sector vs SPY features done.")

    log.info("Computing cross-sectional rank features...")
    xs_cols = ["rsi_14", "zscore_20", "zscore_60", "dist_ma20", "ret_5d", "sector_rel_zscore",
               "dist_52w_high", "vol_ratio_20"]
    for col in xs_cols:
        if col in result.columns:
            # global rank across all S&P 500 on each date
            result[f"{col}_xs_rank"] = (
                result.groupby("date")[col].rank(pct=True, na_option="keep")
            )
    # sector-relative rank (within sector ETF group on each date)
    if "sector_etf" in result.columns:
        for col in ["rsi_14", "zscore_20", "ret_5d", "sector_rel_zscore"]:
            if col in result.columns:
                result[f"{col}_sec_rank"] = (
                    result.groupby(["date", "sector_etf"])[col]
                    .rank(pct=True, na_option="keep")
                )
    log.info("Cross-sectional features done.")

    # Regime dispersion: std of daily returns across all stocks on each date.
    # When low, everything moves together (macro shock / risk-off) and relative
    # value signals lose power. The model learns to self-suppress in those regimes.
    log.info("Computing regime dispersion features...")
    daily_xs_std = result.groupby("date")["ret_1d"].std()
    xs_disp_5d   = daily_xs_std.rolling(5,  min_periods=3).mean()
    xs_disp_20d  = daily_xs_std.rolling(20, min_periods=10).mean()
    result["xs_disp_5d"]  = result["date"].map(xs_disp_5d)
    result["xs_disp_20d"] = result["date"].map(xs_disp_20d)
    log.info("Regime dispersion done.")

    # Drop rows missing core features or target; fill optional enrichment with neutral values
    sector_onehot_cols = {f"sector_{etf}" for etf in ALL_SECTOR_ETFS}
    soft_cols = {"vix", "sector_rel_zscore", "xs_disp_5d", "xs_disp_20d",
                 "sector_vs_spy_20d", "sector_vs_spy_60d"} | sector_onehot_cols
    core_cols = [c for c in get_feature_columns()
                 if c not in soft_cols and "_xs_rank" not in c and "_sec_rank" not in c]
    n_before = len(result)
    result = result.dropna(subset=core_cols)
    result = result.dropna(subset=["fwd_ret_21d"])
    result["vix"]               = result["vix"].fillna(20.0)
    result["sector_rel_zscore"] = result["sector_rel_zscore"].fillna(0.0)
    xs_rank_cols = [c for c in result.columns if "_xs_rank" in c or "_sec_rank" in c]
    result[xs_rank_cols] = result[xs_rank_cols].fillna(0.5)
    result["xs_disp_5d"]        = result["xs_disp_5d"].fillna(result["xs_disp_5d"].mean())
    result["xs_disp_20d"]       = result["xs_disp_20d"].fillna(result["xs_disp_20d"].mean())
    result["sector_vs_spy_20d"] = result["sector_vs_spy_20d"].fillna(0.0)
    result["sector_vs_spy_60d"] = result["sector_vs_spy_60d"].fillna(0.0)
    log.info(f"Dropped {n_before - len(result):,} NaN rows. {len(result):,} remaining.")

    return result


def get_feature_columns() -> list:
    return [
        # price-based technicals
        "rsi_14", "rsi_21",
        "zscore_60",
        "dist_ma20", "dist_ma50", "dist_ma200",
        "dist_52w_high", "dist_52w_low",
        "atr_norm", "vol_regime", "intraday_range",
        # volume
        "vol_ratio_20",
        # returns (5d+ only — shorter windows are too noisy)
        "ret_5d", "ret_10d", "ret_20d",
        # market context
        "vix",
        # sector-relative signal
        "sector_rel_zscore",
        # cross-sectional ranks: absolute values matter less than rank within universe
        "rsi_14_xs_rank", "zscore_20_xs_rank", "zscore_60_xs_rank",
        "dist_ma20_xs_rank", "ret_5d_xs_rank", "sector_rel_zscore_xs_rank",
        "dist_52w_high_xs_rank", "vol_ratio_20_xs_rank",
        # sector-relative ranks
        "rsi_14_sec_rank", "zscore_20_sec_rank",
        "ret_5d_sec_rank", "sector_rel_zscore_sec_rank",
        # regime gate
        "xs_disp_5d", "xs_disp_20d",
        # sector identity (one-hot) — model learns sector-specific biases vs SPY
        # XLU/XLE/XLY/XLC dropped (bottom 25% SHAP importance)
        "sector_XLK", "sector_XLF", "sector_XLV",
        "sector_XLI", "sector_XLP", "sector_XLB", "sector_XLRE",
    ]


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    master   = run_pipeline(use_cache=True)
    featured = build_features_all(master)
    out = Path("data/clean/featured.parquet")
    featured.to_parquet(out, index=False)
    log.info(f"Saved to {out}")
