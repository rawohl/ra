import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pickle

from feature_engineering import build_features, get_feature_columns, zscore, ALL_SECTOR_ETFS
from data_pipeline import get_sp500_universe
from config import TOP_N, MIN_SPREAD

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOOKBACK_DAYS   = 504   # 2 trading years: covers 52-week rolling windows + sector history
UNIVERSE_CACHE  = Path("data/clean/universe.json")
UNIVERSE_TTL    = 7     # days before re-scraping Wikipedia
PRICE_CACHE     = Path("data/clean/price_cache.parquet")
SECTOR_CACHE    = Path("data/clean/sector_cache.parquet")
CACHE_OVERLAP   = 7     # days of overlap to re-fetch on incremental update (handles late adjustments)


def _get_universe() -> dict[str, str]:
    """Return cached universe dict; only re-scrapes Wikipedia after UNIVERSE_TTL days."""
    import json
    if UNIVERSE_CACHE.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(
            UNIVERSE_CACHE.stat().st_mtime)).total_seconds() / 86400
        if age_days < UNIVERSE_TTL:
            with open(UNIVERSE_CACHE) as f:
                return json.load(f)
    log.info("Universe cache stale or missing — fetching from Wikipedia...")
    universe = get_sp500_universe()
    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(UNIVERSE_CACHE, "w") as f:
        json.dump(universe, f)
    return universe


def load_model():
    path = Path("models/model.pkl")
    if not path.exists():
        raise FileNotFoundError("No model found. Run training first.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


def fetch_current_vix() -> float:
    """Fetch the latest VIX close from FRED."""
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", timeout=10)
        vix = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"],
                          index_col="observation_date")
        vix["VIXCLS"] = pd.to_numeric(vix["VIXCLS"], errors="coerce")
        latest = float(vix["VIXCLS"].dropna().iloc[-1])
        log.info(f"Current VIX: {latest:.1f}")
        return latest
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}. Defaulting to 20.0.")
        return 20.0


def _download_tickers(tickers: list[str], start: str, end: str) -> dict:
    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    data = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(how="all")
            if len(df) < 10 or (df["Close"] <= 0).any():
                continue
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            data[ticker] = df
        except Exception:
            continue
    return data


def _cache_save(data: dict, path: Path) -> None:
    frames = []
    for ticker, df in data.items():
        tmp = df.copy().reset_index().rename(columns={"index": "date", "Date": "date"})
        tmp["ticker"] = ticker
        frames.append(tmp)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def _cache_load(path: Path) -> dict:
    flat = pd.read_parquet(path)
    flat["date"] = pd.to_datetime(flat["date"])
    result = {}
    for ticker, grp in flat.groupby("ticker"):
        result[ticker] = grp.drop(columns="ticker").set_index("date").sort_index()
    return result


def fetch_fresh_data(universe: dict[str, str], start: str, end: str) -> dict:
    tickers  = list(universe.keys())
    start_dt = pd.Timestamp(start)

    if PRICE_CACHE.exists():
        cached = _cache_load(PRICE_CACHE)
        last_dt = max(df.index.max() for df in cached.values())
        inc_start = (last_dt - timedelta(days=CACHE_OVERLAP)).strftime("%Y-%m-%d")
        log.info(f"Price cache through {last_dt.date()} — fetching from {inc_start}...")
        new = _download_tickers(tickers, inc_start, end)
        if new:
            for ticker, df in new.items():
                if ticker in cached:
                    old = cached[ticker]
                    old = old[old.index < df.index.min()]
                    cached[ticker] = pd.concat([old, df]).sort_index()
                else:
                    cached[ticker] = df
        # trim to lookback window and save
        data = {t: df[df.index >= start_dt] for t, df in cached.items()
                if len(df[df.index >= start_dt]) >= 60}
        _cache_save(cached, PRICE_CACHE)
    else:
        log.info(f"Fetching {len(tickers)} tickers ({LOOKBACK_DAYS}d history)...")
        data = _download_tickers(tickers, start, end)
        data = {t: df for t, df in data.items() if len(df) >= 60}
        if data:
            _cache_save(data, PRICE_CACHE)

    log.info(f"Using {len(data)} tickers.")
    return data


def fetch_sector_etf_returns(etfs: list[str], start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start)

    if SECTOR_CACHE.exists():
        cached = pd.read_parquet(SECTOR_CACHE)
        # Normalise: handle date as column ("date"/"Date"/"index") or as named/unnamed index
        if "date" not in cached.columns:
            if "Date" in cached.columns:
                cached = cached.rename(columns={"Date": "date"})
            elif "index" in cached.columns:
                cached = cached.rename(columns={"index": "date"})
            else:
                cached = cached.reset_index()
                cached = cached.rename(columns={"Date": "date", "index": "date"})
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date").sort_index()
        last_dt = cached.index.max()
        inc_start = (last_dt - timedelta(days=CACHE_OVERLAP)).strftime("%Y-%m-%d")
        log.info(f"Sector cache through {last_dt.date()} — fetching from {inc_start}...")
        new = _fetch_etf_raw(etfs, inc_start, end)
        if not new.empty:
            cached = cached[cached.index < new.index.min()]
            cached = pd.concat([cached, new]).sort_index()
        cached.index.name = "date"  # pd.concat loses name when the two sides differ
        result = cached[cached.index >= start_dt]
        cached.reset_index().to_parquet(SECTOR_CACHE, index=False)
        return result
    else:
        data = _fetch_etf_raw(etfs, start, end)
        if not data.empty:
            data.index.name = "date"
            data.reset_index().to_parquet(SECTOR_CACHE, index=False)
        return data


def _fetch_etf_raw(etfs: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(etfs, start=start, end=end,
                          auto_adjust=True, progress=False, group_by="ticker")
        out = {}
        for etf in etfs:
            try:
                prices = raw[etf]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                prices.index = pd.to_datetime(prices.index)
                if prices.index.tz is not None:
                    prices.index = prices.index.tz_convert(None)
                out[etf] = prices.pct_change()
            except Exception:
                continue
        result = pd.DataFrame(out)
        result.index.name = "date"
        return result
    except Exception as e:
        log.warning(f"Sector ETF fetch failed: {e}")
        return pd.DataFrame()


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["ret_1d"]  = close.pct_change(1)
    df["ret_2d"]  = close.pct_change(2)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    return df


def compute_sector_zscore(df: pd.DataFrame, sector_etf: str,
                           sector_rets: pd.DataFrame) -> float:
    """Latest sector-relative 20-period z-score for a ticker."""
    if sector_etf not in sector_rets.columns or len(df) < 25:
        return 0.0
    try:
        stock_ret  = df["ret_5d"]
        sector_5d  = sector_rets[sector_etf].rolling(5).sum()
        rel_ret    = stock_ret - sector_5d.reindex(stock_ret.index)
        rel_z      = zscore(rel_ret, 20)
        val = rel_z.dropna().iloc[-1] if not rel_z.dropna().empty else 0.0
        return float(val) if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def generate_signals(top_n: int = TOP_N) -> pd.DataFrame:
    model, feature_cols = load_model()
    universe = _get_universe()

    end_dt    = datetime.today()
    start_dt  = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    data = fetch_fresh_data(universe, start_str, end_str)
    if not data:
        log.error("No data fetched.")
        return pd.DataFrame()

    # Fetch shared context once for all tickers
    current_vix = fetch_current_vix()

    all_etfs    = list(set(universe.values()))
    sector_rets = fetch_sector_etf_returns(all_etfs + ["SPY"], start_str, end_str)

    # Cross-sectional return dispersion: std of daily returns across all tickers.
    # Low dispersion = correlated market (macro shock) = signals less reliable.
    log.info("Computing regime dispersion...")
    _ret_panel   = pd.DataFrame({t: df["Close"].pct_change() for t, df in data.items()})
    _daily_xs    = _ret_panel.std(axis=1).dropna()
    xs_disp_5d   = float(_daily_xs.rolling(5,  min_periods=3).mean().iloc[-1])
    xs_disp_20d  = float(_daily_xs.rolling(20, min_periods=10).mean().iloc[-1])
    # thresholds calibrated on 5yr S&P 500 history: ~0.008 = typical calm day
    if xs_disp_5d < 0.007:
        log.warning(f"low dispersion {xs_disp_5d:.4f} — stocks highly correlated, "
                    f"regime unfavorable for relative-value signals")
    else:
        log.info(f"regime dispersion  5d={xs_disp_5d:.4f}  20d={xs_disp_20d:.4f}")

    # Pre-compute sector vs SPY relative return for latest date (one value per sector ETF).
    # Used to give the model the middle layer: market → sector vs market → stock vs sector.
    sector_vs_spy_vals: dict = {}
    if "SPY" in sector_rets.columns and len(sector_rets["SPY"].dropna()) >= 60:
        spy_20 = sector_rets["SPY"].rolling(20).sum()
        spy_60 = sector_rets["SPY"].rolling(60).sum()
        for etf in all_etfs:
            if etf in sector_rets.columns:
                v20 = float(sector_rets[etf].rolling(20).sum().iloc[-1]) - float(spy_20.iloc[-1])
                v60 = float(sector_rets[etf].rolling(60).sum().iloc[-1]) - float(spy_60.iloc[-1])
                sector_vs_spy_vals[etf] = (
                    v20 if np.isfinite(v20) else 0.0,
                    v60 if np.isfinite(v60) else 0.0,
                )
    log.info(f"sector vs SPY computed for {len(sector_vs_spy_vals)} ETFs")

    # these are computed globally (not per-ticker) and injected after the loop
    rank_cols   = {c for c in feature_cols if "_xs_rank" in c or "_sec_rank" in c}
    sector_cols = {c for c in feature_cols if c.startswith("sector_")}
    global_ctx  = {"vix", "sector_rel_zscore", "xs_disp_5d", "xs_disp_20d"}
    base_cols   = [c for c in feature_cols
                   if c not in rank_cols and c not in global_ctx and c not in sector_cols]

    rows = []
    for ticker, df in data.items():
        try:
            df = add_returns(df)
            df = build_features(df)
            latest = df.iloc[-1]

            if latest[base_cols].isna().any():
                continue

            row = {col: latest[col] for col in base_cols}

            # global context — same value for all tickers
            row["vix"]               = current_vix

            # Sector one-hot: same encoding used during training
            ticker_sector = universe.get(ticker, "SPY")
            for etf in ALL_SECTOR_ETFS:
                row[f"sector_{etf}"] = 1.0 if etf == ticker_sector else 0.0
            row["sector_rel_zscore"] = compute_sector_zscore(
                df, universe.get(ticker, "SPY"), sector_rets)
            row["xs_disp_5d"]        = xs_disp_5d
            row["xs_disp_20d"]       = xs_disp_20d
            svs = sector_vs_spy_vals.get(ticker_sector, (0.0, 0.0))
            row["sector_vs_spy_20d"] = svs[0]
            row["sector_vs_spy_60d"] = svs[1]

            row["ticker"]        = ticker
            row["current_price"] = float(latest["Close"])
            row["sector_etf"]    = universe.get(ticker, "SPY")
            row["date"]          = df.index[-1]
            rows.append(row)

        except Exception as e:
            log.debug(f"{ticker}: {e}")
            continue

    if not rows:
        log.error("No valid feature rows built.")
        return pd.DataFrame()

    features_df = pd.DataFrame(rows)

    # Compute cross-sectional rank features now that all tickers are assembled.
    # Global rank (vs all S&P 500 in today's universe)
    xs_map = {
        "rsi_14_xs_rank":              "rsi_14",
        "zscore_20_xs_rank":           "zscore_20",
        "zscore_60_xs_rank":           "zscore_60",
        "dist_ma20_xs_rank":           "dist_ma20",
        "ret_5d_xs_rank":              "ret_5d",
        "sector_rel_zscore_xs_rank":   "sector_rel_zscore",
        "dist_52w_high_xs_rank":       "dist_52w_high",
        "vol_ratio_20_xs_rank":        "vol_ratio_20",
    }
    for rank_col, src in xs_map.items():
        features_df[rank_col] = (
            features_df[src].rank(pct=True, na_option="keep").fillna(0.5)
            if src in features_df.columns else 0.5
        )
    # Sector-relative rank (within each sector ETF group)
    sec_map = {
        "rsi_14_sec_rank":             "rsi_14",
        "zscore_20_sec_rank":          "zscore_20",
        "ret_5d_sec_rank":             "ret_5d",
        "sector_rel_zscore_sec_rank":  "sector_rel_zscore",
    }
    for rank_col, src in sec_map.items():
        if src in features_df.columns and "sector_etf" in features_df.columns:
            features_df[rank_col] = (
                features_df.groupby("sector_etf")[src]
                .rank(pct=True, na_option="keep")
                .fillna(0.5)
            )
        else:
            features_df[rank_col] = 0.5

    X = features_df[feature_cols]
    features_df["prob_up"] = model.predict_proba(X)[:, 1]

    spread = float(features_df["prob_up"].max() - features_df["prob_up"].min())
    if spread < MIN_SPREAD:
        log.info(f"Probability spread {spread:.4f} below MIN_SPREAD {MIN_SPREAD} "
                 f"— model too uncertain today, no signals.")
        return pd.DataFrame()

    # Top-N selection: rank all stocks by confidence, take best N each direction.
    longs  = features_df.nlargest(top_n,  "prob_up").copy()
    shorts = features_df.nsmallest(top_n, "prob_up").copy()
    longs["side"]  = "long"
    shorts["side"] = "short"

    signals = pd.concat([longs, shorts], ignore_index=True)
    signals["_sort"] = signals.apply(
        lambda r: r["prob_up"] if r["side"] == "long" else 1.0 - r["prob_up"], axis=1
    )
    signals = signals.sort_values("_sort", ascending=False).drop(columns="_sort")

    regime = "CALM" if current_vix < 15 else "NORMAL" if current_vix < 20 else "ELEVATED" if current_vix < 30 else "FEAR"
    log.info(f"top-{top_n} long  ·  top-{top_n} short  "
             f"from {len(features_df)} stocks  |  VIX={current_vix:.1f} ({regime})")

    out_cols = [
        "ticker", "side", "prob_up", "current_price", "sector_etf",
        "rsi_14", "zscore_60", "dist_52w_high", "dist_ma20",
        "sector_rel_zscore", "vix", "xs_disp_5d", "xs_disp_20d", "vol_regime", "date",
    ]
    available = [c for c in out_cols if c in signals.columns]
    return signals[available].reset_index(drop=True)


if __name__ == "__main__":
    signals = generate_signals()
    if signals.empty:
        print("No signals today.")
    else:
        print(signals.to_string(index=False))
        fname = f"signals_{datetime.today().strftime('%Y%m%d')}.csv"
        signals.to_csv(fname, index=False)
        print(f"\nSaved to {fname}")
