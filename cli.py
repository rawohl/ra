"""
ra — cli mode

one-liner:
  python main.py --no-gui signals
  python main.py --no-gui signals --min-prob 0.55 --output picks.csv
  python main.py --no-gui backtest --debug

interactive (diskpart-style):
  python main.py --no-gui
  ra> signals
  ra> train --debug
  ra> exit

--debug  strips decorative formatting to compact key=value lines.
         Same operations and side-effects; just machine-readable output.
"""

import shlex
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import TOP_N

DATA  = Path("data/clean/featured.parquet")
MODEL = Path("models/model.pkl")
PREDS = Path("data/clean/predictions.parquet")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)

W = 52


def _rule():  print("  " + "─" * W)
def _gap():   print()
def _ok(m):   print(f"  ✓  {m}")
def _warn(m): print(f"  ⚠  {m}")
def _err(m):  print(f"  ✗  {m}")

def _d(*parts):
    """Debug print — space-separated key=value parts on one line."""
    print(" ".join(str(p) for p in parts))


def _status():
    d = DATA.exists()
    m = MODEL.exists()
    p = PREDS.exists()
    stale = m and DATA.exists() and DATA.stat().st_mtime > MODEL.stat().st_mtime
    print(f"  data       {'ready' if d else 'missing'}")
    print(f"  model      {'outdated — retrain' if stale else 'ready' if m else 'missing'}")
    print(f"  predictions {'ready' if p else 'missing'}")


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_download(debug: bool = False):
    if not debug:
        print("  01  download data + build features")
        _rule()
    from data_pipeline import run_pipeline
    from feature_engineering import build_features_all

    master   = run_pipeline(use_cache=False)
    featured = build_features_all(master)
    featured.to_parquet(DATA, index=False)

    stale = [p for p in (MODEL, PREDS) if p.exists()]
    for p in stale:
        p.unlink()

    if debug:
        _d("download", f"rows={len(featured):,}", f"tickers={featured['ticker'].nunique()}",
           f"stale_deleted={[p.name for p in stale] or 'none'}")
    else:
        if stale:
            _warn(f"deleted stale artifacts: {[p.name for p in stale]}")
            _warn("retrain required  (train)")
        _gap()
        _ok(f"data ready  —  {len(featured):,} rows  ·  {featured['ticker'].nunique()} tickers")


def cmd_train(debug: bool = False, n_trials: int = 0):
    if not DATA.exists():
        _err("no data found.  run: download"); return

    if not debug:
        print("  02  train model" + (f"  (optuna {n_trials} trials)" if n_trials > 0 else ""))
        _rule()
    from model_training import run_walk_forward

    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    preds = run_walk_forward(df, n_trials=n_trials)
    if preds is None or preds.empty:
        _err("training: FAILED no predictions generated"); return

    preds.to_parquet(PREDS, index=False)
    if debug:
        _d("train", f"predictions={len(preds):,}",
           f"date_range={preds['date'].min().date()}..{preds['date'].max().date()}",
           f"optuna_trials={n_trials}")
    else:
        _gap()
        _ok(f"model ready  —  {len(preds):,} predictions")


def cmd_backtest(debug: bool = False):
    if not PREDS.exists():
        _err("no predictions found.  run: train"); return

    if not debug:
        print("  03  backtest")
        _rule()
    from backtesting import run_backtest, plot_results

    preds = pd.read_parquet(PREDS)
    res   = run_backtest(preds)
    if not res:
        _err("backtest: FAILED no results"); return

    m   = res["metrics"]
    pf  = f"{m['profit_factor']:.3f}" if m["profit_factor"] != float("inf") else "inf"
    cal = f"{m['calmar_ratio']:.3f}"  if m["calmar_ratio"]  != float("inf") else "inf"
    lwr = m.get("long_win_rate",  float("nan"))
    swr = m.get("short_win_rate", float("nan"))

    if debug:
        _d("backtest",
           f"ret={m['total_return']:.2%}", f"ann={m['annualized_return']:.2%}",
           f"sharpe={m['sharpe_ratio']:.3f}", f"calmar={cal}",
           f"max_dd={m['max_drawdown']:.2%}", f"wr={m['win_rate']:.2%}",
           f"pf={pf}", f"trades={m['total_trades']:,}",
           f"long={m.get('long_trades',0)}", f"short={m.get('short_trades',0)}",
           f"lwr={'n/a' if lwr!=lwr else f'{lwr:.2%}'}",
           f"swr={'n/a' if swr!=swr else f'{swr:.2%}'}",
           f"sig/day={m['signals_per_day']:.1f}", f"equity=EUR{m['final_equity']:,.2f}")
    else:
        if m["sharpe_ratio"] > 1.0 and m["calmar_ratio"] > 0.5 and m["max_drawdown"] > -0.20:
            verdict = "viable  —  paper trade before going live"
        elif m["sharpe_ratio"] > 0.7:
            verdict = "marginal edge  —  needs more refinement"
        else:
            verdict = "no meaningful edge yet"

        _gap(); _rule()
        print(f"  {'total return':<24}  {m['total_return']:>9.2%}")
        print(f"  {'annualized':<24}  {m['annualized_return']:>9.2%}")
        print(f"  {'sharpe':<24}  {m['sharpe_ratio']:>9.3f}")
        print(f"  {'calmar':<24}  {cal:>9}")
        print(f"  {'max drawdown':<24}  {m['max_drawdown']:>9.2%}")
        print(f"  {'win rate':<24}  {m['win_rate']:>9.2%}")
        print(f"  {'profit factor':<24}  {pf:>9}")
        print(f"  {'total trades':<24}  {m['total_trades']:>9,}")
        print(f"  {'  long / short':<24}  {m.get('long_trades',0):>5,} / {m.get('short_trades',0)}")
        print(f"  {'long win rate':<24}  {'n/a':>9}" if lwr != lwr else f"  {'long win rate':<24}  {lwr:>9.2%}")
        print(f"  {'short win rate':<24}  {'n/a':>9}" if swr != swr else f"  {'short win rate':<24}  {swr:>9.2%}")
        print(f"  {'signals / day':<24}  {m['signals_per_day']:>9.1f}")
        print(f"  {'final equity':<24}  €{m['final_equity']:>8,.2f}")
        _rule()
        print(f"  {verdict}")
        _rule()

    plot_results(res)
    if not debug:
        _gap()
        _ok("chart saved  →  backtest_results.png")


def cmd_signals(top_n: int = TOP_N, output: str | None = None,
                debug: bool = False):
    if not MODEL.exists():
        _err("no model found.  run: train"); return

    if not debug:
        print(f"  04  signals  (top {top_n} long + {top_n} short)")
        _rule()
    from signal_generator import generate_signals

    sigs = generate_signals(top_n=top_n)
    if sigs is None or sigs.empty:
        _d("signals: none") if debug else (print("  no signals above threshold"), _gap())
        return

    n_long  = (sigs.get("side", pd.Series(["long"] * len(sigs))) == "long").sum()
    n_short = len(sigs) - n_long
    vix     = float(sigs["vix"].iloc[0])         if "vix"        in sigs.columns else float("nan")
    disp    = float(sigs["xs_disp_5d"].iloc[0])  if "xs_disp_5d" in sigs.columns else float("nan")

    if debug:
        _d("signals",
           f"long={n_long}", f"short={n_short}",
           f"vix={vix:.1f}", f"disp={disp:.4f}",
           f"ts={datetime.now().strftime('%H:%M %Y-%m-%d')}")
        for r in sigs.itertuples(index=False):
            side = getattr(r, "side", "long")
            conf = r.prob_up if side == "long" else 1.0 - r.prob_up
            _d(f"  {getattr(r,'ticker','')}", side,
               f"conf={conf:.3f}", f"price={getattr(r,'current_price',0):.2f}",
               f"rsi={getattr(r,'rsi_14',0):.1f}", f"z={getattr(r,'zscore_20',0):.2f}",
               f"sz={getattr(r,'sector_rel_zscore',0):.2f}",
               f"bb={getattr(r,'bb_pos_20',0):.3f}",
               f"sector={getattr(r,'sector_etf','')}")
    else:
        if "vix" in sigs.columns:
            if   vix < 15: vr = "calm      ·  weaker edge"
            elif vix < 20: vr = "normal"
            elif vix < 30: vr = "elevated  ·  stronger edge"
            else:          vr = "fear  ·  high risk"
            print(f"  vix {vix:.1f}  {vr}")
        if "xs_disp_5d" in sigs.columns:
            if   disp < 0.007: dr = "correlated  ·  signals less reliable"
            elif disp < 0.012: dr = "normal"
            else:              dr = "dispersed   ·  signals more reliable"
            print(f"  dispersion {disp:.4f}  {dr}")
        if "vix" in sigs.columns or "xs_disp_5d" in sigs.columns:
            _rule()

        hdr = f"  {'ticker':<7}  {'side':<5}  {'sector':<7}  {'conf':>6}  {'price':>9}  {'rsi-14':>6}  {'z-score':>7}  {'sect-z':>7}  {'bb pos':>6}"
        print(hdr)
        print("  " + "·" * (len(hdr) - 2))
        for r in sigs.itertuples(index=False):
            side    = getattr(r, "side", "long")
            is_long = side == "long"
            conf    = r.prob_up if is_long else 1.0 - r.prob_up
            marker  = "  *" if conf >= 0.65 else "   "
            print(
                f"{marker} {getattr(r, 'ticker', ''):<7}  "
                f"{side:<5}  "
                f"{getattr(r, 'sector_etf', ''):<7}  "
                f"{r.prob_up:.1%}  "
                f"${getattr(r, 'current_price', 0):>8.2f}  "
                f"{getattr(r, 'rsi_14', 0):>6.1f}  "
                f"{getattr(r, 'zscore_20', 0):>7.2f}  "
                f"{getattr(r, 'sector_rel_zscore', 0):>7.2f}  "
                f"{getattr(r, 'bb_pos_20', 0):>6.3f}"
            )

        _gap()
        ts = datetime.now().strftime("%H:%M  %Y-%m-%d")
        print(f"  {n_long} long  ·  {n_short} short  ·  {ts}")
        _gap()

    if output:
        sigs.to_csv(output, index=False)
        if not debug:
            _ok(f"saved  →  {output}")


# ── taha tools helpers ────────────────────────────────────────────────────────

_VIX_BINS   = [0, 15, 20, 30, 999]
_VIX_LABELS = ["calm (<15)", "normal (15-20)", "elevated (20-30)", "fear (>30)"]


def _mark_correct(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    df = df.copy()
    labeled = df["target"].notna()
    df["correct"] = np.where(
        labeled,
        (((df["signal"] ==  1) & (df["target"] == 1)) |
         ((df["signal"] == -1) & (df["target"] == 0))).astype(float),
        np.nan
    )
    return df


def _load_preds_with(extra_cols: list[str]) -> pd.DataFrame | None:
    """Load predictions and left-join requested columns from featured.parquet."""
    preds = pd.read_parquet(PREDS)
    preds["date"] = pd.to_datetime(preds["date"])
    if preds["date"].dt.tz is not None:
        preds["date"] = preds["date"].dt.tz_convert(None)

    missing = [c for c in extra_cols if c not in preds.columns]
    if missing and DATA.exists():
        feat = pd.read_parquet(DATA, columns=["date", "ticker"] + missing)
        feat["date"] = pd.to_datetime(feat["date"])
        if feat["date"].dt.tz is not None:
            feat["date"] = feat["date"].dt.tz_convert(None)
        preds = preds.merge(feat, on=["date", "ticker"], how="left")
    return preds


# ── taha tools commands ───────────────────────────────────────────────────────

def cmd_baseline(debug: bool = False):
    if not PREDS.exists():
        _err("no predictions found.  run: train"); return
    if not DATA.exists():
        _err("featured.parquet missing.  run: download"); return

    if not debug:
        print("  taha  naive baseline vs ml model")
        _rule()

    from backtesting import run_backtest
    import numpy as np

    preds = _load_preds_with(["ret_20d"])

    # Naive reversal: buy the bottom 30% losers of last month, short the top 30%
    naive = preds.copy()
    naive["naive_rank"] = naive.groupby("date")["ret_20d"].rank(pct=True)
    naive["signal"]  = 0
    naive.loc[naive["naive_rank"] <= 0.30, "signal"] =  1
    naive.loc[naive["naive_rank"] >= 0.70, "signal"] = -1
    # Flat confidence so confidence-weighting is neutral (weight = 0.5 for all)
    naive["prob_up"] = np.where(naive["signal"] == 1, 0.52, 0.48)

    res_n = run_backtest(naive)
    res_m = run_backtest(preds)

    if not res_n or not res_m:
        _err("baseline: backtest failed"); return

    n, m = res_n["metrics"], res_m["metrics"]
    edge   = m["sharpe_ratio"] - n["sharpe_ratio"]
    winner = "ml" if edge > 0 else "naive"

    if debug:
        _d("baseline naive",
           f"ret={n['total_return']:.2%}", f"ann={n['annualized_return']:.2%}",
           f"sharpe={n['sharpe_ratio']:.3f}", f"max_dd={n['max_drawdown']:.2%}",
           f"wr={n['win_rate']:.2%}", f"trades={n['total_trades']:,}")
        _d("baseline ml",
           f"ret={m['total_return']:.2%}", f"ann={m['annualized_return']:.2%}",
           f"sharpe={m['sharpe_ratio']:.3f}", f"max_dd={m['max_drawdown']:.2%}",
           f"wr={m['win_rate']:.2%}", f"trades={m['total_trades']:,}")
        _d(f"edge winner={winner} sharpe_diff={abs(edge):.3f}")
    else:
        W = 54
        _gap(); _rule()
        print(f"  {'BASELINE vs ML MODEL':^{W-4}}")
        _rule()
        print(f"  {'metric':<22}  {'naive reversal':>14}  {'ml model':>10}")
        _rule()
        for label, key, fmt in [
            ("total return",  "total_return",      "{:.2%}"),
            ("annualized",    "annualized_return",  "{:.2%}"),
            ("sharpe ratio",  "sharpe_ratio",       "{:.3f}"),
            ("max drawdown",  "max_drawdown",       "{:.2%}"),
            ("win rate",      "win_rate",           "{:.2%}"),
            ("total trades",  "total_trades",       "{:,}"),
        ]:
            print(f"  {label:<22}  {fmt.format(n.get(key,0)):>14}  {fmt.format(m.get(key,0)):>10}")
        _rule()
        print(f"  {winner} outperforms by {abs(edge):.3f} sharpe")
        _rule(); _gap()


def cmd_regime(debug: bool = False):
    if not PREDS.exists():
        _err("no predictions found.  run: train"); return

    if not debug:
        print("  taha  regime breakdown")
        _rule()

    import numpy as np

    preds  = _load_preds_with(["xs_disp_20d"])
    active = _mark_correct(preds[preds["signal"] != 0])
    active["vix_regime"] = pd.cut(active["vix"], bins=_VIX_BINS, labels=_VIX_LABELS)
    total_per_date = preds.groupby("date")["ticker"].count()

    def _regime_row(bucket):
        longs  = bucket[bucket["signal"] ==  1]
        shorts = bucket[bucket["signal"] == -1]
        lp = float(longs["correct"].mean())  if longs["correct"].notna().sum()  > 10 else float("nan")
        sp = float(shorts["correct"].mean()) if shorts["correct"].notna().sum() > 10 else float("nan")
        avail = total_per_date.reindex(bucket["date"].unique()).sum()
        rate  = len(bucket) / avail if avail > 0 else float("nan")
        return lp, sp, rate

    if debug:
        for regime in _VIX_LABELS:
            bucket = active[active["vix_regime"] == regime]
            if bucket.empty:
                _d(f"regime {regime!r} signals=0"); continue
            lp, sp, rate = _regime_row(bucket)
            _d(f"regime {regime!r}",
               f"n={len(bucket):,}",
               f"long_prec={'n/a' if np.isnan(lp) else f'{lp:.1%}'}",
               f"short_prec={'n/a' if np.isnan(sp) else f'{sp:.1%}'}",
               f"sig_rate={'n/a' if np.isnan(rate) else f'{rate:.1%}'}")

        if "xs_disp_20d" in active.columns and active["xs_disp_20d"].notna().any():
            active["disp_q"] = pd.qcut(active["xs_disp_20d"], q=4,
                                        labels=["Q1(low)", "Q2", "Q3", "Q4(high)"],
                                        duplicates="drop")
            for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
                bucket = active[active["disp_q"] == q]
                if bucket.empty: continue
                lp, sp, rate = _regime_row(bucket)
                _d(f"disp {q}",
                   f"n={len(bucket):,}",
                   f"long_prec={'n/a' if np.isnan(lp) else f'{lp:.1%}'}",
                   f"short_prec={'n/a' if np.isnan(sp) else f'{sp:.1%}'}",
                   f"sig_rate={'n/a' if np.isnan(rate) else f'{rate:.1%}'}")
    else:
        W = 62
        _gap(); _rule()
        print(f"  {'VIX REGIME BREAKDOWN':^{W-4}}")
        _rule()
        print(f"  {'regime':<18} {'signals':>8} {'long prec':>10} {'short prec':>11} {'sig rate':>9}")
        _rule()
        for regime in _VIX_LABELS:
            bucket = active[active["vix_regime"] == regime]
            if bucket.empty:
                print(f"  {regime:<18}  {'—':>8}"); continue
            lp, sp, rate = _regime_row(bucket)
            lp_s  = f"{lp:.1%}"   if not np.isnan(lp)   else "n/a"
            sp_s  = f"{sp:.1%}"   if not np.isnan(sp)   else "n/a"
            rt_s  = f"{rate:.1%}" if not np.isnan(rate) else "n/a"
            print(f"  {regime:<18} {len(bucket):>8,} {lp_s:>10} {sp_s:>11} {rt_s:>9}")
        _rule()

        if "xs_disp_20d" in active.columns and active["xs_disp_20d"].notna().any():
            active["disp_q"] = pd.qcut(active["xs_disp_20d"], q=4,
                                        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                        duplicates="drop")
            _gap()
            print(f"  {'DISPERSION QUARTILE BREAKDOWN':^{W-4}}")
            _rule()
            print(f"  {'quartile':<18} {'signals':>8} {'long prec':>10} {'short prec':>11} {'sig rate':>9}")
            _rule()
            for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
                bucket = active[active["disp_q"] == q]
                if bucket.empty: continue
                lp, sp, rate = _regime_row(bucket)
                lp_s  = f"{lp:.1%}"   if not np.isnan(lp)   else "n/a"
                sp_s  = f"{sp:.1%}"   if not np.isnan(sp)   else "n/a"
                rt_s  = f"{rate:.1%}" if not np.isnan(rate) else "n/a"
                print(f"  {q:<18} {len(bucket):>8,} {lp_s:>10} {sp_s:>11} {rt_s:>9}")
            _rule()
        _gap()


def cmd_sector(debug: bool = False):
    if not PREDS.exists():
        _err("no predictions found.  run: train"); return
    if not DATA.exists():
        _err("featured.parquet missing.  run: download"); return

    if not debug:
        print("  taha  precision by sector")
        _rule()

    import numpy as np

    preds  = _load_preds_with(["sector_etf"])
    active = _mark_correct(preds[preds["signal"] != 0])
    sectors = sorted(active["sector_etf"].dropna().unique())

    for side_label, side_val in [("LONG", 1), ("SHORT", -1)]:
        side_df = active[active["signal"] == side_val]
        overall = float(side_df["correct"].mean()) if len(side_df) > 0 else float("nan")

        rows = []
        for sec in sectors:
            sb = side_df[side_df["sector_etf"] == sec]
            if len(sb) < 10: continue
            rows.append((sec, len(sb), len(sb) / max(len(side_df), 1), float(sb["correct"].mean())))
        rows.sort(key=lambda x: x[3], reverse=True)

        if debug:
            _d(f"sector {side_label.lower()} overall={overall:.1%}")
            for sec, cnt, share, prec in rows:
                _d(f"  {sec}", f"n={cnt:,}", f"share={share:.1%}", f"prec={prec:.1%}",
                   f"edge={prec-0.50:+.1%}")
        else:
            W = 64
            _gap(); _rule()
            print(f"  {side_label + ' PRECISION BY SECTOR':^{W-4}}")
            _rule()
            print(f"  {'sector':<8} {'signals':>8} {'% of total':>11} {'precision':>10} {'edge':>8}")
            _rule()
            for sec, cnt, share, prec in rows:
                edge = prec - 0.50
                bar  = "█" * max(1, int(edge * 200)) if edge > 0 else "░"
                print(f"  {sec:<8} {cnt:>8,} {share:>10.1%} {prec:>10.1%}  {bar}")
            _rule()
            print(f"  {'overall':<8} {len(side_df):>8,} {'100.0%':>11} {overall:>10.1%}")
            _rule()
    if not debug:
        _gap()


# ── argument parser (shared between one-liner and REPL) ───────────────────────

_HELP = """\
  commands:
    download                  step 01 — fetch data + build features
    train                     step 02 — walk-forward model training
    backtest                  step 03 — out-of-sample backtest
    signals  [options]        step 04 — generate live signals  [default]
    all      [options]        run all four steps in sequence
    baseline                  taha — naive reversal vs ml model
    regime                    taha — VIX regime + dispersion breakdown
    sector                    taha — precision by sector
    status                    show pipeline readiness
    help                      show this message
    exit / quit               leave

  options:
    --top-n     N             top N long + N short signals per day  (default 5)
    --output    FILE          save signals to csv file
    --trials    N             optuna HPO trials before training  (default 0 = off)
    --debug                   compact key=value output (same operations)\
"""


def _parse(tokens: list[str]) -> tuple[str, int, str | None, int, bool] | None:
    """Parse a command + optional flags. Returns (cmd, top_n, output, n_trials, debug) or None."""
    import argparse
    p = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    p.add_argument("command", nargs="?", default="signals",
                   choices=["download","train","backtest","signals","all",
                             "baseline","regime","sector",
                             "status","help","exit","quit","q"])
    p.add_argument("--top-n", type=int, default=TOP_N, dest="top_n")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--trials", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-gui", action="store_true")   # silently accepted
    try:
        args = p.parse_args(tokens)
        return args.command, args.top_n, args.output, args.trials, args.debug
    except (argparse.ArgumentError, SystemExit):
        print(f"  unknown command or option — type help")
        return None


def _dispatch(cmd, top_n=TOP_N, output=None, n_trials=0, debug=False):
    if debug:
        logging.getLogger().setLevel(logging.WARNING)

    if cmd in ("exit", "quit", "q"):
        print("  bye"); sys.exit(0)
    elif cmd == "help":
        print(_HELP)
    elif cmd == "status":
        _status()
    elif cmd == "download":
        cmd_download(debug)
    elif cmd == "train":
        cmd_train(debug, n_trials)
    elif cmd == "backtest":
        cmd_backtest(debug)
    elif cmd == "signals":
        cmd_signals(top_n, output, debug)
    elif cmd == "baseline":
        cmd_baseline(debug)
    elif cmd == "regime":
        cmd_regime(debug)
    elif cmd == "sector":
        cmd_sector(debug)
    elif cmd == "all":
        cmd_download(debug); _gap() if not debug else None
        cmd_train(debug, n_trials); _gap() if not debug else None
        cmd_backtest(debug); _gap() if not debug else None
        cmd_signals(top_n, output, debug)


# ── entry points ───────────────────────────────────────────────────────────────

def run_once(argv: list[str]):
    """One-liner mode: parse argv and run once, then exit."""
    tokens = [a for a in argv if a != "--no-gui"]
    parsed = _parse(tokens)
    if parsed:
        _dispatch(*parsed)


def run_repl():
    """Interactive REPL mode — stays open until exit/quit."""
    print("  ra  —  s&p 500 mean reversion signals")
    _rule()
    _status()
    _gap()
    print("  type  help  for available commands")
    _gap()

    while True:
        try:
            line = input("  ra> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); print("  bye"); break

        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError as e:
            print(f"  parse error: {e}"); continue

        parsed = _parse(tokens)
        if parsed:
            cmd, top_n, output, n_trials, debug = parsed
            if cmd in ("exit", "quit", "q"):
                print("  bye"); break
            _dispatch(cmd, top_n, output, n_trials, debug)
        _gap()
