"""
S&P 500 Mean Reversion Signal System v3
Phase 1: Data Pipeline

Downloads 5 years of S&P 500 OHLCV data with GICS sector mapping.
All dates stored as timezone-naive UTC to avoid downstream issues.
Note: the last 5 trading days of each ticker are dropped (fwd_ret_5d is NaN).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
YEARS_BACK = 5
MIN_TRADING_DAYS = 400

# GICS Sector → sector ETF proxy
SECTOR_ETF_MAP = {
    "Information Technology":  "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Utilities":               "XLU",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Communication Services":  "XLC",
}


def get_sp500_universe() -> dict[str, str]:
    """
    Scrape S&P 500 constituents from Wikipedia.
    Returns {ticker: sector_etf} using GICS sector mapping.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    for table in soup.find_all("table"):
        header_row = table.find("tr")
        if header_row and "Symbol" in header_row.get_text():
            universe = {}
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    ticker = cells[0].get_text().strip().replace(".", "-")
                    sector = cells[2].get_text().strip()
                    universe[ticker] = SECTOR_ETF_MAP.get(sector, "SPY")
            log.info(f"Found {len(universe)} tickers.")
            return universe

    raise ValueError("Could not find S&P 500 table on Wikipedia.")


def get_sp500_tickers() -> list[str]:
    """Backward-compatible wrapper — returns just the ticker list."""
    return list(get_sp500_universe().keys())


def download_ohlcv(universe: dict[str, str]) -> dict[str, pd.DataFrame]:
    tickers = list(universe.keys())
    end = datetime.today()
    start = end - timedelta(days=365 * YEARS_BACK)

    log.info(f"Downloading {len(tickers)} tickers from {start.date()} to {end.date()}...")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True
    )

    data = {}
    failed = []

    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(how="all")

            if len(df) < MIN_TRADING_DAYS:
                failed.append(ticker)
                continue

            df = df[df["Close"] > 0]

            # OHLC sanity: drop rows where High < Low or prices are nonsensical
            bad = (df["High"] < df["Low"]) | (df["High"] < df["Close"]) | (df["Low"] > df["Close"])
            if bad.any():
                log.debug(f"{ticker}: removing {bad.sum()} rows with invalid OHLC")
                df = df[~bad]

            if len(df) < MIN_TRADING_DAYS:
                failed.append(ticker)
                continue

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df.index.name = "date"
            df["sector_etf"] = universe[ticker]
            data[ticker] = df

        except Exception as e:
            log.debug(f"{ticker}: {e}")
            failed.append(ticker)

    log.info(f"Loaded {len(data)} tickers. Failed/filtered: {len(failed)}.")
    return data


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["ret_1d"]  = close.pct_change(1)
    df["ret_2d"]  = close.pct_change(2)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["fwd_ret_5d"]  = close.pct_change(5).shift(-5)
    df["fwd_ret_21d"] = close.pct_change(21).shift(-21)  # 1-month forward: reversal is stronger here
    return df


def build_master(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, df in data.items():
        df = df.copy()
        df["ticker"] = ticker
        df = df.reset_index()
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    if master["date"].dt.tz is not None:
        master["date"] = master["date"].dt.tz_convert(None)
    master = master.sort_values(["ticker", "date"]).reset_index(drop=True)
    log.info(f"Master: {len(master):,} rows, {master['ticker'].nunique()} tickers.")
    return master


def run_pipeline(use_cache: bool = True) -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = CLEAN_DIR / "master.parquet"

    if use_cache and cache.exists():
        log.info("Loading cached master...")
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
        return df

    universe = get_sp500_universe()
    # cache for signal generator — avoids re-scraping Wikipedia every signal run
    import json
    with open(CLEAN_DIR / "universe.json", "w") as _f:
        json.dump(universe, _f)
    data = download_ohlcv(universe)
    data = {t: add_returns(df) for t, df in data.items()}
    master = build_master(data)
    master = master.dropna(subset=["fwd_ret_21d"])  # 21d is the training target; stricter than 5d
    master.to_parquet(cache, index=False)
    log.info(f"Saved master to {cache}")
    return master


if __name__ == "__main__":
    df = run_pipeline(use_cache=False)
    print(df.head())
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Tickers: {df['ticker'].nunique()}, Rows: {len(df):,}")
    if "sector_etf" in df.columns:
        print("\nSector ETF distribution:")
        print(df[["ticker", "sector_etf"]].drop_duplicates()["sector_etf"].value_counts())
