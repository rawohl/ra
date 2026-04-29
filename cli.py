"""
ra — cli mode

one-liner:
  python main.py --no-gui signals
  python main.py --no-gui signals --min-prob 0.55 --output picks.csv

interactive (diskpart-style):
  python main.py --no-gui
  ra> signals
  ra> train
  ra> exit
"""

import shlex
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

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


def _status():
    d = DATA.exists()
    m = MODEL.exists()
    p = PREDS.exists()
    stale = m and DATA.exists() and DATA.stat().st_mtime > MODEL.stat().st_mtime
    print(f"  data       {'ready' if d else 'missing'}")
    print(f"  model      {'outdated — retrain' if stale else 'ready' if m else 'missing'}")
    print(f"  predictions {'ready' if p else 'missing'}")


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_download():
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
    if stale:
        _warn(f"deleted stale artifacts: {[p.name for p in stale]}")
        _warn("retrain required  (train)")

    _gap()
    _ok(f"data ready  —  {len(featured):,} rows  ·  {featured['ticker'].nunique()} tickers")


def cmd_train():
    if not DATA.exists():
        _err("no data found.  run: download"); return

    print("  02  train model")
    _rule()
    from model_training import run_walk_forward

    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    preds = run_walk_forward(df)
    if preds is None or preds.empty:
        _err("training failed — no predictions generated"); return

    preds.to_parquet(PREDS, index=False)
    _gap()
    _ok(f"model ready  —  {len(preds):,} predictions")


def cmd_backtest():
    if not PREDS.exists():
        _err("no predictions found.  run: train"); return

    print("  03  backtest")
    _rule()
    from backtesting import run_backtest, plot_results

    preds = pd.read_parquet(PREDS)
    res   = run_backtest(preds)
    if not res:
        _err("backtest returned no results"); return

    m   = res["metrics"]
    pf  = f"{m['profit_factor']:.3f}" if m["profit_factor"] != float("inf") else "∞"
    cal = f"{m['calmar_ratio']:.3f}"  if m["calmar_ratio"]  != float("inf") else "∞"

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
    lwr = m.get('long_win_rate');  print(f"  {'long win rate':<24}  {lwr:>9.2%}" if lwr == lwr else f"  {'long win rate':<24}  {'n/a':>9}")
    swr = m.get('short_win_rate'); print(f"  {'short win rate':<24}  {swr:>9.2%}" if swr == swr else f"  {'short win rate':<24}  {'n/a':>9}")
    print(f"  {'signals / day':<24}  {m['signals_per_day']:>9.1f}")
    print(f"  {'final equity':<24}  €{m['final_equity']:>8,.2f}")
    _rule()
    print(f"  {verdict}")
    _rule()

    plot_results(res)
    _gap()
    _ok("chart saved  →  backtest_results.png")


def cmd_signals(min_prob: float = 0.52, output: str | None = None):
    if not MODEL.exists():
        _err("no model found.  run: train"); return

    print(f"  04  signals  (min confidence {min_prob:.0%})")
    _rule()
    from signal_generator import generate_signals

    sigs = generate_signals(min_prob=min_prob)
    if sigs is None or sigs.empty:
        print("  no signals above threshold"); _gap(); return

    if "vix" in sigs.columns:
        v = float(sigs["vix"].iloc[0])
        if   v < 15: vr = "calm      ·  weaker edge"
        elif v < 20: vr = "normal"
        elif v < 30: vr = "elevated  ·  stronger edge"
        else:        vr = "extreme   ·  high risk"
        print(f"  vix {v:.1f}  {vr}")
    if "xs_disp_5d" in sigs.columns:
        d = float(sigs["xs_disp_5d"].iloc[0])
        if   d < 0.007: dr = "correlated  ·  signals less reliable"
        elif d < 0.012: dr = "normal"
        else:           dr = "dispersed   ·  signals more reliable"
        print(f"  dispersion {d:.4f}  {dr}")
    if "vix" in sigs.columns or "xs_disp_5d" in sigs.columns:
        _rule()

    hdr = f"  {'ticker':<7}  {'side':<5}  {'sector':<7}  {'conf':>6}  {'price':>9}  {'rsi-14':>6}  {'z-score':>7}  {'sect-z':>7}  {'bb pos':>6}"
    print(hdr)
    print("  " + "·" * (len(hdr) - 2))
    for _, r in sigs.iterrows():
        side    = r.get("side", "long")
        is_long = side == "long"
        conf    = r["prob_up"] if is_long else 1.0 - r["prob_up"]
        marker  = "  *" if conf >= 0.65 else "   "
        print(
            f"{marker} {r.get('ticker',''):<7}  "
            f"{side:<5}  "
            f"{r.get('sector_etf',''):<7}  "
            f"{r['prob_up']:.1%}  "
            f"${r.get('current_price', 0):>8.2f}  "
            f"{r.get('rsi_14', 0):>6.1f}  "
            f"{r.get('zscore_20', 0):>7.2f}  "
            f"{r.get('sector_rel_zscore', 0):>7.2f}  "
            f"{r.get('bb_pos_20', 0):>6.3f}"
        )

    _gap()
    n_long  = (sigs.get("side", pd.Series(["long"]*len(sigs))) == "long").sum()
    n_short = len(sigs) - n_long
    ts = datetime.now().strftime("%H:%M  %Y-%m-%d")
    print(f"  {n_long} long  ·  {n_short} short  ·  {ts}")

    if output:
        sigs.to_csv(output, index=False)
        _ok(f"saved  →  {output}")
    _gap()


# ── argument parser (shared between one-liner and REPL) ───────────────────────

_HELP = """\
  commands:
    download                  step 01 — fetch data + build features
    train                     step 02 — walk-forward model training
    backtest                  step 03 — out-of-sample backtest
    signals  [options]        step 04 — generate live signals  [default]
    all      [options]        run all four steps in sequence
    status                    show pipeline readiness
    help                      show this message
    exit / quit               leave

  signal options:
    --min-prob  P             confidence threshold 0–1  (default 0.52)
    --output    FILE          save signals to csv file\
"""


def _parse(tokens: list[str]) -> tuple[str, float, str | None] | None:
    """Parse a command + optional flags. Returns (cmd, min_prob, output) or None."""
    import argparse
    p = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    p.add_argument("command", nargs="?", default="signals",
                   choices=["download","train","backtest","signals","all",
                             "status","help","exit","quit","q"])
    p.add_argument("--min-prob", type=float, default=0.52, dest="min_prob")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--no-gui", action="store_true")   # silently accepted
    try:
        args = p.parse_args(tokens)
        return args.command, args.min_prob, args.output
    except (argparse.ArgumentError, SystemExit):
        print(f"  unknown command or option — type help")
        return None


def _dispatch(cmd, min_prob, output):
    if cmd in ("exit", "quit", "q"):
        print("  bye"); sys.exit(0)
    elif cmd == "help":
        print(_HELP)
    elif cmd == "status":
        _status()
    elif cmd == "download":
        cmd_download()
    elif cmd == "train":
        cmd_train()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "signals":
        cmd_signals(min_prob, output)
    elif cmd == "all":
        cmd_download(); _gap()
        cmd_train();    _gap()
        cmd_backtest(); _gap()
        cmd_signals(min_prob, output)


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
            cmd, min_prob, output = parsed
            if cmd in ("exit", "quit", "q"):
                print("  bye"); break
            _dispatch(cmd, min_prob, output)
        _gap()
