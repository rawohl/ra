"""
ra — s&p 500 mean reversion signals
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

try:
    from PIL import Image, ImageTk
    _pil = True
except ImportError:
    _pil = False

# palette — black base, gold accent
BG    = "#0a0a0a"
CARD  = "#111111"
LINE  = "#1d1d1d"
FG    = "#eeeeee"
MUTED = "#4a4a4a"
GOLD  = "#c9a227"
RED   = "#b03030"
GREEN = "#2e8b57"
AMBER = "#c07020"

F     = ("Segoe UI", 10)
F_SM  = ("Segoe UI",  9)
F_H   = ("Segoe UI", 13, "bold")
F_LOG = ("Consolas",  9)

DATA  = Path("data/clean/featured.parquet")
MODEL = Path("models/model.pkl")
PREDS = Path("data/clean/predictions.parquet")

_VIX_BINS   = [0, 15, 20, 30, 999]
_VIX_LABELS = ["calm (<15)", "normal (15-20)", "elevated (20-30)", "fear (>30)"]


def _mtime(p):
    return p.stat().st_mtime if p.exists() else 0.0


def _mark_correct(df):
    df = df.copy()
    df["correct"] = (
        ((df["signal"] ==  1) & (df["target"] == 1)) |
        ((df["signal"] == -1) & (df["target"] == 0))
    ).astype(int)
    return df


class _Tip:
    """Delayed tooltip window — shows after hover_ms ms, hides on leave."""
    def __init__(self, root, hover_ms=350):
        self._root = root
        self._ms   = hover_ms
        self._win  = None
        self._job  = None

    def schedule(self, text, rx, ry):
        self.cancel()
        self._job = self._root.after(self._ms, lambda: self._show(text, rx, ry))

    def cancel(self):
        if self._job:
            self._root.after_cancel(self._job)
            self._job = None
        self._hide()

    def _show(self, text, rx, ry):
        self._hide()
        self._win = tk.Toplevel(self._root)
        self._win.wm_overrideredirect(True)
        self._win.wm_geometry(f"+{rx}+{ry}")
        tk.Label(
            self._win, text=text, bg="#161616", fg=FG, font=("Segoe UI", 9),
            relief="flat", padx=10, pady=7, justify="left", wraplength=300,
            highlightbackground=GOLD, highlightthickness=1,
        ).pack()

    def _hide(self):
        if self._win:
            try: self._win.destroy()
            except Exception: pass
            self._win = None


# column header tooltip text for the signals treeview
# columns: #1 ticker  #2 side  #3 sector  #4 conf  #5 price  #6 rsi  #7 zscore  #8 sect_z  #9 bb
_COL_TIPS = {
    "#1": "ticker\nstock symbol as listed on the exchange",
    "#2": (
        "side  —  trade direction\n"
        "long   model predicts this stock will rank in the\n"
        "       top 30% of s&p 500 over the next 21 days\n"
        "short  model predicts bottom 30% relative performer\n"
        "       (bet against it)"
    ),
    "#3": (
        "sector  —  gics sector etf proxy\n"
        "XLK  technology        XLF  financials\n"
        "XLV  health care       XLE  energy\n"
        "XLI  industrials       XLY  consumer discret.\n"
        "XLP  consumer staples  XLU  utilities\n"
        "XLB  materials         XLRE real estate\n"
        "XLC  communication services"
    ),
    "#4": (
        "conf  —  directional model confidence\n"
        "long:  prob_up  = probability of top 30% performance\n"
        "short: 1−prob_up = probability of bottom 30% performance\n\n"
        "values cluster near 50–55% by design — the model has\n"
        "real but thin edge. 53% right vs 47% wrong over\n"
        "thousands of trades compounds into meaningful returns.\n"
        "≥65% highlighted in gold (long) or red (short)"
    ),
    "#5": "price\nlatest closing price in usd",
    "#6": (
        "rsi-14  —  relative strength index (14-day)\n"
        ">70  overbought / momentum territory\n"
        "<30  oversold / potential reversal\n"
        " 50  neutral"
    ),
    "#7": (
        "z-score  —  20-day price z-score\n"
        "standard deviations from the 20-day rolling mean.\n"
        "+2 = well above average  /  −2 = well below"
    ),
    "#8": (
        "sect-z  —  sector-relative z-score\n"
        "z-score of this stock's 5-day return minus its sector etf.\n"
        "positive = outperforming sector peers recently\n"
        "negative = lagging sector peers recently"
    ),
    "#9": (
        "bb pos  —  bollinger band position\n"
        "where price sits within the 20-day bollinger bands.\n"
        " 0 = at lower band    1 = at upper band\n"
        ">1 = above upper band (extended / overbought)"
    ),
}


class LogSink(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.w = widget

    def emit(self, record):
        msg = self.format(record)
        def _put():
            if not self.w.winfo_exists():
                return
            self.w.configure(state="normal")
            self.w.insert(tk.END, msg + "\n")
            self.w.see(tk.END)
            self.w.configure(state="disabled")
        self.w.after(0, _put)


class Ra(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ra")
        self.geometry("980x680")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._sigs  = pd.DataFrame()
        self._page  = ""
        self._nav   = {}
        self._pages = {}
        self._style()
        self._ui()
        self._refresh()

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, font=F)
        s.configure("TProgressbar", troughcolor=CARD, background=GOLD, borderwidth=0)
        s.configure("T.Treeview",
                    background=CARD, foreground=FG,
                    fieldbackground=CARD, rowheight=26, font=F)
        s.configure("T.Treeview.Heading",
                    background=BG, foreground=GOLD,
                    font=("Segoe UI", 9, "bold"), relief="flat")
        s.map("T.Treeview",
              background=[("selected", LINE)],
              foreground=[("selected", GOLD)])
        s.configure("TScrollbar",
                    background=CARD, troughcolor=BG,
                    arrowcolor=MUTED, borderwidth=0)

    def _ui(self):
        self._header()
        self._navbar()
        self._wrap = tk.Frame(self, bg=BG)
        self._wrap.pack(fill="both", expand=True)
        self._pages["setup"]      = self._pg_setup(self._wrap)
        self._pages["signals"]    = self._pg_signals(self._wrap)
        self._pages["backtest"]   = self._pg_backtest(self._wrap)
        self._pages["log"]        = self._pg_log(self._wrap)
        self._pages["tahatools"]  = self._pg_tahatools(self._wrap)
        self._pages["about"]      = self._pg_about(self._wrap)
        # stack all pages in the same space — lift() switches without blank flash
        for pg in self._pages.values():
            pg.place(in_=self._wrap, x=0, y=0, relwidth=1, relheight=1)
        self._go("setup")

    # ── header ───────────────────────────────────────────────────────────────

    def _header(self):
        h = tk.Frame(self, bg=BG)
        h.pack(fill="x")
        tk.Frame(h, bg=LINE, height=1).pack(side="bottom", fill="x")

        row = tk.Frame(h, bg=BG)
        row.pack(fill="x", padx=20, pady=10)

        logo = tk.Frame(row, bg=BG)
        logo.pack(side="left")
        self._draw_eye(logo)
        tk.Label(logo, text="ra", bg=BG, fg=FG, font=F_H).pack(side="left", padx=(10, 0))

        self._vix_lbl = tk.Label(row, text="", bg=BG, fg=MUTED, font=F_SM)
        self._vix_lbl.pack(side="right")
        tk.Label(row, text="// technical preview @rawohl",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(side="right", padx=(0, 18))

    def _draw_eye(self, parent):
        p = Path("assets/ra_eye.png")
        if _pil and p.exists():
            try:
                img = Image.open(p).convert("RGBA").resize((40, 32), Image.LANCZOS)
                px  = img.load()
                for y in range(img.height):
                    for x in range(img.width):
                        r, g, b, a = px[x, y]
                        if r > 180 and g > 180 and b > 180:
                            px[x, y] = (0, 0, 0, 0)        # white → transparent
                        else:
                            px[x, y] = (0xc9, 0xa2, 0x27, 255)  # dark → gold
                self._eye = ImageTk.PhotoImage(img)
                tk.Label(parent, image=self._eye, bg=BG).pack(side="left")
                return
            except Exception:
                pass
        tk.Label(parent, text="𓂀", bg=BG, fg=GOLD, font=("Segoe UI", 20)).pack(side="left")

    # ── nav bar ───────────────────────────────────────────────────────────────

    def _navbar(self):
        bar = tk.Frame(self, bg=CARD, height=38)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=LINE, height=1).pack(side="bottom", fill="x")

        row = tk.Frame(bar, bg=CARD)
        row.pack(side="left", fill="y", padx=4)

        for name in ("setup", "signals", "backtest", "log"):
            b = tk.Button(
                row, text=name, bg=CARD, fg=MUTED, font=F,
                bd=0, padx=18, relief="flat", cursor="hand2",
                activebackground=CARD, activeforeground=FG,
                command=lambda n=name: self._go(n),
            )
            b.pack(side="left", fill="y")
            self._nav[name] = b

        # about pinned right; taha tools just left of it
        about_btn = tk.Button(
            bar, text="about", bg=CARD, fg=MUTED, font=F,
            bd=0, padx=18, relief="flat", cursor="hand2",
            activebackground=CARD, activeforeground=FG,
            command=lambda: self._go("about"),
        )
        about_btn.pack(side="right", fill="y")
        self._nav["about"] = about_btn

        tt_btn = tk.Button(
            bar, text="taha tools", bg=CARD, fg=MUTED, font=F,
            bd=0, padx=18, relief="flat", cursor="hand2",
            activebackground=CARD, activeforeground=FG,
            command=lambda: self._go("tahatools"),
        )
        tt_btn.pack(side="right", fill="y")
        self._nav["tahatools"] = tt_btn

    # ── routing ───────────────────────────────────────────────────────────────

    def _go(self, name):
        if name == "signals" and not MODEL.exists():
            messagebox.showinfo("ra", "train the model first  (step 02)")
            return
        if name == "backtest" and not PREDS.exists():
            messagebox.showinfo("ra", "complete training before viewing backtest")
            return

        self._pages[name].lift()
        self._page = name

        for n, b in self._nav.items():
            b.configure(fg=(GOLD if n == name else MUTED),
                        bg=(BG   if n == name else CARD))

    # ── setup page ────────────────────────────────────────────────────────────

    def _pg_setup(self, parent):
        f = tk.Frame(parent, bg=BG)

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=24, pady=(20, 0))
        tk.Label(top, text="setup", bg=BG, fg=FG, font=F_H).pack(side="left")
        self._status_lbl = tk.Label(top, text="", bg=BG, fg=MUTED, font=F_SM)
        self._status_lbl.pack(side="right")

        steps = tk.Frame(f, bg=BG)
        steps.pack(fill="x", padx=24, pady=16)

        # step 01
        card1 = self._card(steps, "01", "download data",
                           "5 years of s&p 500 ohlcv + gics sector features\n"
                           "~3 min  ·  only needed once")
        card1.pack(fill="x", pady=(0, 8))
        row1 = tk.Frame(card1, bg=CARD)
        row1.pack(fill="x", padx=16, pady=(0, 14))
        self._btn_download = self._btn(row1, "run", self._do_download)
        self._btn_download.pack(side="left")
        self._spin_download = tk.Label(row1, text="", bg=CARD, fg=MUTED, font=F_SM)
        self._spin_download.pack(side="left", padx=(10, 0))

        # step 02
        card2 = self._card(steps, "02", "train model",
                           "walk-forward lightgbm  ·  target: beat spy over 21 days\n"
                           "~5 min  ·  retrain monthly")
        card2.pack(fill="x", pady=(0, 8))
        row2 = tk.Frame(card2, bg=CARD)
        row2.pack(fill="x", padx=16, pady=(0, 14))
        self._btn_train = self._btn(row2, "run", self._do_train)
        self._btn_train.pack(side="left")
        self._spin_train = tk.Label(row2, text="", bg=CARD, fg=MUTED, font=F_SM)
        self._spin_train.pack(side="left", padx=(10, 0))

        self._spin_job = None   # pending after() id for spinner animation

        return f

    def _card(self, parent, num, title, sub):
        c = tk.Frame(parent, bg=CARD)
        tk.Frame(c, bg=LINE, height=1).pack(fill="x")   # top rule

        row = tk.Frame(c, bg=CARD)
        row.pack(fill="x", padx=16, pady=(12, 6))

        tk.Label(row, text=num, bg=CARD, fg=MUTED, font=F_SM).pack(side="left", padx=(0, 14))

        body = tk.Frame(row, bg=CARD)
        body.pack(side="left", fill="x", expand=True)
        tk.Label(body, text=title, bg=CARD, fg=FG,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        tk.Label(body, text=sub, bg=CARD, fg=MUTED,
                 font=F_SM, anchor="w", justify="left").pack(fill="x")
        return c

    def _btn(self, parent, label, cmd):
        return tk.Button(
            parent, text=label, command=cmd,
            bg=CARD, fg=GOLD, font=("Segoe UI", 9, "bold"),
            bd=1, relief="solid", padx=16, pady=5,
            cursor="hand2", activebackground=BG, activeforeground=GOLD,
            highlightthickness=1, highlightbackground=GOLD, highlightcolor=GOLD,
        )

    # ── signals page ──────────────────────────────────────────────────────────

    def _pg_signals(self, parent):
        f = tk.Frame(parent, bg=BG)

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=20, pady=14)
        tk.Label(top, text="signals", bg=BG, fg=FG, font=F_H).pack(side="left")
        self._btn(top, "export csv", self._export).pack(side="right", padx=(6, 0))
        self._btn_signals = self._btn(top, "generate", self._do_signals)
        self._btn_signals.pack(side="right")
        self._spin_signals = tk.Label(top, text="", bg=BG, fg=MUTED, font=F_SM, width=6)
        self._spin_signals.pack(side="right", padx=(0, 4))

        sf = tk.Frame(f, bg=BG)
        sf.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(sf, text="min confidence", bg=BG, fg=MUTED, font=F_SM).pack(side="left")
        self._min_prob = tk.DoubleVar(value=0.52)
        self._plbl = tk.Label(sf, text="52%", bg=BG, fg=GOLD,
                              font=("Segoe UI", 9, "bold"), width=4)
        self._plbl.pack(side="left", padx=(6, 0))
        ttk.Scale(sf, from_=0.50, to=0.80, variable=self._min_prob,
                  orient="horizontal", length=180,
                  command=lambda v: self._plbl.configure(text=f"{float(v):.0%}")).pack(
                      side="left", padx=8)

        cols = ("ticker", "side", "sector", "conf", "price", "rsi", "zscore", "sect_z", "bb")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                  height=17, style="T.Treeview")
        self._col_defs = [
            ("ticker", "ticker",  78), ("side",   "side",    50),
            ("sector", "sector",  58), ("conf",   "conf",    62),
            ("price",  "price",   84), ("rsi",    "rsi-14",  62),
            ("zscore", "z-score", 70), ("sect_z", "sect-z",  64),
            ("bb",     "bb pos",  64),
        ]
        self._sort_state = {}   # col → True means currently sorted descending
        for col, lbl, w in self._col_defs:
            self._tree.heading(col, text=lbl,
                               command=lambda c=col: self._sort_tree(c))
            self._tree.column(col, width=w, anchor="center")
            self._sort_state[col] = False

        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=4)
        sb.pack(side="left", fill="y", pady=4)

        self._sig_lbl = tk.Label(f, text="", bg=BG, fg=MUTED, font=F_SM)
        self._sig_lbl.pack(padx=20, pady=(0, 8))

        self._col_tip = _Tip(self)
        self._tree.bind("<Motion>", self._on_tree_motion)
        self._tree.bind("<Leave>",  lambda e: self._col_tip.cancel())

        return f

    def _on_tree_motion(self, event):
        region = self._tree.identify_region(event.x, event.y)
        col    = self._tree.identify_column(event.x)
        if region == "heading" and col in _COL_TIPS:
            rx = self._tree.winfo_rootx() + event.x + 14
            ry = self._tree.winfo_rooty() + event.y + 22
            self._col_tip.schedule(_COL_TIPS[col], rx, ry)
        else:
            self._col_tip.cancel()

    def _sort_tree(self, col):
        desc = not self._sort_state.get(col, False)
        self._sort_state[col] = desc

        def _key(iid):
            val = self._tree.set(iid, col)
            cleaned = val.replace("%", "").replace("$", "").strip()
            try:
                return (0, float(cleaned))   # numeric sort
            except ValueError:
                return (1, cleaned.lower())  # string sort fallback

        items = sorted(self._tree.get_children(""), key=_key, reverse=desc)
        for idx, iid in enumerate(items):
            self._tree.move(iid, "", idx)

        # update headings: show arrow on sorted col, reset all others
        arrow = " ↓" if desc else " ↑"
        for c, lbl, _ in self._col_defs:
            self._tree.heading(c, text=lbl + (arrow if c == col else ""))

    # ── backtest page ─────────────────────────────────────────────────────────

    def _pg_backtest(self, parent):
        f = tk.Frame(parent, bg=BG)

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=20, pady=14)
        tk.Label(top, text="backtest", bg=BG, fg=FG, font=F_H).pack(side="left")
        self._spin_backtest = tk.Label(top, text="", bg=BG, fg=MUTED, font=F_SM)
        self._spin_backtest.pack(side="right", padx=(0, 8))
        self._btn_backtest = self._btn(top, "run backtest", self._do_backtest)
        self._btn_backtest.pack(side="right")

        self._bt_txt = scrolledtext.ScrolledText(
            f, bg=CARD, fg=FG, font=F_LOG,
            state="disabled", relief="flat", borderwidth=0,
            insertbackground=GOLD, selectbackground=LINE)
        self._bt_txt.pack(fill="both", expand=True, padx=20, pady=(0, 4))

        tk.Label(f, text="chart → backtest_results.png", bg=BG, fg=MUTED,
                 font=F_SM).pack(pady=(0, 8))
        return f

    # ── log page ──────────────────────────────────────────────────────────────

    def _pg_log(self, parent):
        f = tk.Frame(parent, bg=BG)
        self._log_txt = scrolledtext.ScrolledText(
            f, bg=CARD, fg=FG, font=F_LOG,
            state="disabled", relief="flat", borderwidth=0)
        self._log_txt.pack(fill="both", expand=True, padx=16, pady=16)

        sink = LogSink(self._log_txt)
        sink.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
        logging.getLogger().addHandler(sink)
        logging.getLogger().setLevel(logging.INFO)
        return f

    # ── status ────────────────────────────────────────────────────────────────

    def _refresh(self):
        data_ok  = DATA.exists()
        model_ok = MODEL.exists()
        stale    = model_ok and _mtime(MODEL) < _mtime(DATA)

        self._btn_train.configure(
            fg=(GOLD if data_ok else MUTED),
            state=("normal" if data_ok else "disabled"),
        )

        if stale:
            self._status_lbl.configure(text="model outdated — retrain", fg=AMBER)
        elif data_ok and model_ok:
            self._status_lbl.configure(text="ready", fg=GREEN)
        elif data_ok:
            self._status_lbl.configure(text="train model to continue", fg=AMBER)
        else:
            self._status_lbl.configure(text="start with step 01", fg=MUTED)

    # ── threading ─────────────────────────────────────────────────────────────

    def _run(self, fn, btn, spin_lbl, *args):
        _FRAMES = ["·", "· ·", "· · ·", "· ·"]
        self._frame_idx = 0

        orig_label = btn.cget("text")
        btn.configure(text="working...", state="disabled", fg=MUTED,
                      highlightbackground=MUTED, cursor="")

        def _tick():
            spin_lbl.configure(text=_FRAMES[self._frame_idx % len(_FRAMES)])
            self._frame_idx += 1
            self._spin_job = self.after(350, _tick)

        def _stop():
            if self._spin_job:
                self.after_cancel(self._spin_job)
                self._spin_job = None
            spin_lbl.configure(text="")
            btn.configure(text=orig_label, state="normal", fg=GOLD,
                          highlightbackground=GOLD, cursor="hand2")

        _tick()

        def go():
            try:
                fn(*args)
            except Exception as e:
                logging.error(str(e))
                self.after(0, lambda m=str(e): messagebox.showerror("ra", m))
            finally:
                self.after(0, _stop)
                self.after(0, self._refresh)
        threading.Thread(target=go, daemon=True).start()

    # ── actions ───────────────────────────────────────────────────────────────

    def _do_download(self):
        self._run(self._download_task, self._btn_download, self._spin_download)

    def _download_task(self):
        logging.info("downloading data...")
        from data_pipeline import run_pipeline
        from feature_engineering import build_features_all

        master   = run_pipeline(use_cache=False)
        featured = build_features_all(master)
        featured.to_parquet(DATA, index=False)

        # model trained on old data → delete it so user must retrain
        stale = [p for p in (MODEL, PREDS) if p.exists()]
        for p in stale:
            p.unlink()
        if stale:
            logging.warning(f"deleted stale artifacts: {[p.name for p in stale]}")
            logging.warning("retrain the model (step 02)")

        logging.info("data ready")

    def _do_train(self):
        if not DATA.exists():
            messagebox.showwarning("ra", "download data first  (step 01)")
            return
        self._run(self._train_task, self._btn_train, self._spin_train)

    def _train_task(self):
        logging.info("training model...")
        from model_training import run_walk_forward
        df = pd.read_parquet(DATA)
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
        preds = run_walk_forward(df)
        if preds is None or preds.empty:
            logging.error("training failed — no predictions generated")
            return
        preds.to_parquet(PREDS, index=False)
        logging.info("model ready")

    def _do_signals(self):
        if not MODEL.exists():
            messagebox.showwarning("ra", "train the model first")
            return
        self._run(self._signals_task, self._btn_signals, self._spin_signals)

    def _signals_task(self):
        logging.info("generating signals...")
        from signal_generator import generate_signals
        sigs = generate_signals(min_prob=self._min_prob.get())
        self._sigs = sigs if sigs is not None else pd.DataFrame()

        if not self._sigs.empty and "vix" in self._sigs.columns:
            vix  = float(self._sigs["vix"].iloc[0])
            disp = float(self._sigs["xs_disp_5d"].iloc[0]) if "xs_disp_5d" in self._sigs.columns else None
            self.after(0, lambda v=vix, d=disp: self._set_vix(v, d))

        self.after(0, lambda s=self._sigs: self._show_sigs(s))

    def _set_vix(self, v, disp=None):
        if   v < 15: vc, vlbl = AMBER, f"vix {v:.1f}  calm"
        elif v < 20: vc, vlbl = FG,    f"vix {v:.1f}  normal"
        elif v < 30: vc, vlbl = GREEN, f"vix {v:.1f}  elevated"
        else:        vc, vlbl = RED,   f"vix {v:.1f}  extreme"

        if disp is not None:
            if   disp < 0.007: dc, dlbl = RED,   f"disp {disp:.3f}  correlated"
            elif disp < 0.012: dc, dlbl = AMBER, f"disp {disp:.3f}  normal"
            else:              dc, dlbl = GREEN, f"disp {disp:.3f}  dispersed"
            _priority  = {RED: 0, AMBER: 1, GREEN: 2, FG: 3}
            combined_c = min((vc, dc), key=lambda c: _priority[c])
            self._vix_lbl.configure(text=f"{vlbl}  ·  {dlbl}", fg=combined_c)
        else:
            self._vix_lbl.configure(text=vlbl, fg=vc)

    def _show_sigs(self, sigs):
        for row in self._tree.get_children():
            self._tree.delete(row)

        if sigs is None or sigs.empty:
            self._sig_lbl.configure(text="no signals above threshold", fg=MUTED)
            return

        for _, r in sigs.iterrows():
            side    = r.get("side", "long")
            is_long = side == "long"
            conf    = r["prob_up"] if is_long else 1.0 - r["prob_up"]
            hi      = conf >= 0.65
            tag     = ("hi_long" if hi else "long") if is_long else ("hi_short" if hi else "short")
            self._tree.insert("", "end", values=(
                r.get("ticker", ""),
                side,
                r.get("sector_etf", ""),
                f"{conf:.1%}",
                f"${r.get('current_price', 0):.2f}",
                f"{r.get('rsi_14', 0):.1f}",
                f"{r.get('zscore_20', 0):.2f}",
                f"{r.get('sector_rel_zscore', 0):.2f}",
                f"{r.get('bb_pos_20', 0):.3f}",
            ), tags=(tag,))

        self._tree.tag_configure("long",     foreground=FG)
        self._tree.tag_configure("hi_long",  foreground=GOLD)
        self._tree.tag_configure("short",    foreground=RED)
        self._tree.tag_configure("hi_short", foreground="#e05050")   # brighter red for hi-conf shorts

        n_long  = (sigs.get("side", pd.Series(["long"] * len(sigs))) == "long").sum()
        n_short = len(sigs) - n_long
        self._sig_lbl.configure(
            text=f"{n_long} long  ·  {n_short} short  ·  {datetime.now().strftime('%H:%M')}  ·  gold/red = ≥65%",
            fg=MUTED)

    def _export(self):
        if self._sigs.empty:
            messagebox.showinfo("ra", "generate signals first")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("csv files", "*.csv")],
            initialfile=f"signals_{datetime.today().strftime('%Y%m%d')}.csv",
        )
        if p:
            self._sigs.to_csv(p, index=False)

    def _do_backtest(self):
        if not PREDS.exists():
            messagebox.showwarning("ra", "train the model first")
            return
        self._run(self._backtest_task, self._btn_backtest, self._spin_backtest)

    def _backtest_task(self):
        logging.info("running backtest...")
        from backtesting import run_backtest, plot_results
        preds = pd.read_parquet(PREDS)
        res   = run_backtest(preds)

        if not res:
            logging.error("backtest returned nothing")
            return

        m = res["metrics"]
        if m["sharpe_ratio"] > 1.0 and m["calmar_ratio"] > 0.5 and m["max_drawdown"] > -0.20:
            verdict = "viable  —  paper trade before going live"
        elif m["sharpe_ratio"] > 0.7:
            verdict = "marginal edge  —  needs more refinement"
        else:
            verdict = "no meaningful edge yet"

        pf  = f"{m['profit_factor']:.3f}" if m["profit_factor"] != float("inf") else "∞"
        cal = f"{m['calmar_ratio']:.3f}"  if m["calmar_ratio"]  != float("inf") else "∞"

        lines = [
            "",
            f"  {'─'*44}",
            f"  backtest results",
            f"  {'─'*44}",
            f"  {'total return':<22}  {m['total_return']:>9.2%}",
            f"  {'annualized':<22}  {m['annualized_return']:>9.2%}",
            f"  {'sharpe':<22}  {m['sharpe_ratio']:>9.3f}",
            f"  {'calmar':<22}  {cal:>9}",
            f"  {'max drawdown':<22}  {m['max_drawdown']:>9.2%}",
            f"  {'win rate':<22}  {m['win_rate']:>9.2%}",
            f"  {'profit factor':<22}  {pf:>9}",
            f"  {'total trades':<22}  {m['total_trades']:>9,}",
            f"  {'  long / short':<22}  {m.get('long_trades',0):>5,} / {m.get('short_trades',0):<5,}",
            f"  {'long win rate':<22}  {m.get('long_win_rate', float('nan')):>9.2%}" if not pd.isna(m.get('long_win_rate', float('nan'))) else f"  {'long win rate':<22}  {'n/a':>9}",
            f"  {'short win rate':<22}  {m.get('short_win_rate', float('nan')):>9.2%}" if not pd.isna(m.get('short_win_rate', float('nan'))) else f"  {'short win rate':<22}  {'n/a':>9}",
            f"  {'signals / day':<22}  {m['signals_per_day']:>9.1f}",
            f"  {'final equity':<22}  €{m['final_equity']:>8,.2f}",
            f"  {'─'*44}",
            f"  {verdict}",
            f"  {'─'*44}",
            "",
        ]
        report = "\n".join(lines)
        plot_results(res)

        def _update():
            self._bt_txt.configure(state="normal")
            self._bt_txt.delete("1.0", tk.END)
            self._bt_txt.insert(tk.END, report)
            self._bt_txt.configure(state="disabled")
        self.after(0, _update)
        logging.info("backtest done")

    def _pg_tahatools(self, parent):
        f = tk.Frame(parent, bg=BG)

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=20, pady=14)
        tk.Label(top, text="taha tools", bg=BG, fg=FG, font=F_H).pack(side="left")
        tk.Label(top, text="research & diagnostics",
                 bg=BG, fg=MUTED, font=F_SM).pack(side="left", padx=(12, 0))
        tk.Frame(f, bg=LINE, height=1).pack(fill="x", padx=20, pady=(0, 0))

        canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        sb     = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        def _on_frame(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<Configure>", _on_resize)
        inner.bind("<Configure>", _on_frame)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        def _section(title, desc):
            sec = tk.Frame(inner, bg=CARD)
            sec.pack(fill="x", padx=20, pady=(12, 0))
            tk.Label(sec, text=title, bg=CARD, fg=GOLD,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(10, 2))
            tk.Label(sec, text=desc, bg=CARD, fg=MUTED,
                     font=F_SM, justify="left").pack(anchor="w", padx=14, pady=(0, 8))
            return sec

        def _tool_btn(sec, label, cmd):
            row = tk.Frame(sec, bg=CARD)
            row.pack(anchor="w", padx=14, pady=(0, 10))
            btn = tk.Button(row, text=label, bg=BG, fg=GOLD, font=F_SM,
                            bd=0, padx=14, pady=5, relief="flat", cursor="hand2",
                            activebackground=LINE, activeforeground=GOLD, command=cmd)
            btn.pack(side="left")
            spin = tk.Label(row, text="", bg=CARD, fg=MUTED, font=F_SM)
            spin.pack(side="left", padx=(10, 0))
            return btn, spin

        def _out(sec):
            txt = tk.Text(
                sec, bg=BG, fg=FG, font=F_LOG, height=1,
                state="disabled", relief="flat", borderwidth=0,
                wrap="word", insertbackground=GOLD, selectbackground=LINE)
            # not packed yet — appears only when first line is logged
            return txt

        # ── naive baseline ────────────────────────────────────────────────────
        sec1 = _section("naive baseline comparison",
                        "buy bottom 30% by last month's return, short top 30%.\n"
                        "no model — pure reversal factor. compare sharpe and win rate\n"
                        "vs the ml model to see whether machine learning adds genuine alpha.")
        self._btn_baseline, self._spin_baseline = _tool_btn(sec1, "run baseline",
                                                             self._run_baseline)
        self._baseline_out = _out(sec1)

        # ── regime breakdown ──────────────────────────────────────────────────
        sec2 = _section("regime breakdown",
                        "split all predictions by vix regime and dispersion quartile.\n"
                        "shows when the model works and when it doesn't.")
        self._btn_regime, self._spin_regime = _tool_btn(sec2, "run breakdown",
                                                         self._run_regime)
        self._regime_out = _out(sec2)

        # ── precision by sector ───────────────────────────────────────────────
        sec3 = _section("precision by sector",
                        "long and short precision per gics sector across all folds.\n"
                        "shows which sectors the model has genuine edge in.")
        self._btn_sector, self._spin_sector = _tool_btn(sec3, "run sector analysis",
                                                         self._run_sector)
        self._sector_out = _out(sec3)

        tk.Frame(inner, bg=BG, height=20).pack()  # bottom padding
        return f

    def _tool_log(self, widget, msg):
        widget.configure(state="normal")
        widget.insert("end", msg + "\n")
        lines = int(widget.index("end-1c").split(".")[0])
        widget.configure(height=lines)
        widget.configure(state="disabled")
        if not widget.winfo_ismapped():
            widget.pack(fill="x", padx=14, pady=(0, 12))

    def _tool_clear(self, widget):
        widget.pack_forget()
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(height=1)
        widget.configure(state="disabled")

    def _run_baseline(self):
        if not PREDS.exists():
            messagebox.showinfo("ra", "run training + backtest first")
            return
        self._tool_clear(self._baseline_out)
        self._run(self._baseline_worker, self._btn_baseline, self._spin_baseline)

    def _baseline_worker(self):
        log = lambda m: self._tool_log(self._baseline_out, m)
        try:
            import numpy as np
            from backtesting import ROUND_TRIP_COST

            log("loading predictions...")
            preds = pd.read_parquet(PREDS)
            preds["date"] = pd.to_datetime(preds["date"])

            if "ret_20d" not in preds.columns:
                if not DATA.exists():
                    log("featured.parquet not found — run step 01 first")
                    return
                feat = pd.read_parquet(DATA, columns=["date", "ticker", "ret_20d"])
                feat["date"] = pd.to_datetime(feat["date"])
                preds = preds.merge(feat, on=["date", "ticker"], how="left")

            log("computing naive signals...")
            preds["naive_rank"] = preds.groupby("date")["ret_20d"].rank(pct=True)
            preds["naive_signal"] = 0
            preds.loc[preds["naive_rank"] <= 0.30, "naive_signal"] =  1
            preds.loc[preds["naive_rank"] >= 0.70, "naive_signal"] = -1

            def _backtest(df, sig_col, label):
                pos = df[df[sig_col] != 0].copy()
                if pos.empty:
                    return {}
                pos["raw"] = pos["fwd_ret_21d"] / 21 * pos[sig_col].astype(float)
                pos["net"] = pos["raw"] - ROUND_TRIP_COST / 21
                daily = pos.groupby("date")["net"].mean()
                cum   = (1 + daily).cumprod()
                total = float(cum.iloc[-1] - 1)
                n_days = len(daily)
                ann   = float((1 + total) ** (252 / max(n_days, 1)) - 1)
                vol   = float(daily.std() * (252 ** 0.5))
                sharpe = ann / vol if vol > 0 else 0.0
                peak   = cum.cummax()
                dd     = float(((cum - peak) / peak).min())
                wins   = int((pos["net"] > 0).sum())
                wr     = wins / max(len(pos), 1)
                return {"label": label, "total": total, "ann": ann,
                        "sharpe": sharpe, "max_dd": dd, "win_rate": wr,
                        "trades": len(pos)}

            naive = _backtest(preds, "naive_signal", "naive reversal")
            ml    = _backtest(preds, "signal",       "ml model     ")

            W = 52
            log("\n" + "─" * W)
            log(f"  {'BASELINE vs ML MODEL':^{W-4}}")
            log("─" * W)
            log(f"  {'metric':<22}  {'naive reversal':>14}  {'ml model':>10}")
            log("─" * W)
            metrics = [
                ("total return",  "total",    "{:.2%}"),
                ("annualized",    "ann",      "{:.2%}"),
                ("sharpe ratio",  "sharpe",   "{:.3f}"),
                ("max drawdown",  "max_dd",   "{:.2%}"),
                ("win rate",      "win_rate", "{:.2%}"),
                ("total trades",  "trades",   "{:,}"),
            ]
            for label, key, fmt in metrics:
                nv = fmt.format(naive.get(key, 0))
                ml_v = fmt.format(ml.get(key, 0))
                log(f"  {label:<22}  {nv:>14}  {ml_v:>10}")
            log("─" * W)

            if ml["sharpe"] > naive["sharpe"]:
                edge = ml["sharpe"] - naive["sharpe"]
                log(f"  ml outperforms naive by {edge:.3f} sharpe")
            else:
                edge = naive["sharpe"] - ml["sharpe"]
                log(f"  naive outperforms ml by {edge:.3f} sharpe")
            log("─" * W + "\n")

        except Exception as e:
            log(f"error: {e}")

    # ── regime breakdown ──────────────────────────────────────────────────────

    def _run_regime(self):
        if not PREDS.exists():
            messagebox.showinfo("ra", "run training + backtest first")
            return
        self._tool_clear(self._regime_out)
        self._run(self._regime_worker, self._btn_regime, self._spin_regime)

    def _regime_worker(self):
        log = lambda m: self._tool_log(self._regime_out, m)
        try:
            import numpy as np
            log("loading data...")
            preds = pd.read_parquet(PREDS)
            preds["date"] = pd.to_datetime(preds["date"])

            # join dispersion from featured
            if DATA.exists():
                feat = pd.read_parquet(DATA, columns=["date", "ticker", "xs_disp_20d"])
                feat["date"] = pd.to_datetime(feat["date"])
                preds = preds.merge(feat, on=["date", "ticker"], how="left")

            active = _mark_correct(preds[preds["signal"] != 0])

            W = 62
            log("\n" + "─" * W)
            log(f"  {'VIX REGIME BREAKDOWN':^{W-4}}")
            log("─" * W)
            log(f"  {'regime':<12} {'signals':>8} {'long prec':>10} {'short prec':>11} {'sig rate':>9}")
            log("─" * W)

            active["vix_regime"] = pd.cut(active["vix"], bins=_VIX_BINS, labels=_VIX_LABELS)
            total_per_date = preds.groupby("date")["ticker"].count()

            for regime in _VIX_LABELS:
                bucket = active[active["vix_regime"] == regime]
                if bucket.empty:
                    log(f"  {regime:<12}  {'—':>8}")
                    continue
                longs  = bucket[bucket["signal"] ==  1]
                shorts = bucket[bucket["signal"] == -1]
                lp = longs["correct"].mean()  if len(longs)  > 10 else float("nan")
                sp = shorts["correct"].mean() if len(shorts) > 10 else float("nan")
                # signal rate: signals / available universe on those dates
                regime_dates = bucket["date"].unique()
                avail = total_per_date.reindex(regime_dates).sum()
                rate  = len(bucket) / avail if avail > 0 else float("nan")
                lp_s  = f"{lp:.1%}" if not np.isnan(lp) else "n/a"
                sp_s  = f"{sp:.1%}" if not np.isnan(sp) else "n/a"
                rt_s  = f"{rate:.1%}" if not np.isnan(rate) else "n/a"
                log(f"  {regime:<16} {len(bucket):>8,} {lp_s:>10} {sp_s:>11} {rt_s:>9}")
            log("─" * W)

            if "xs_disp_20d" in active.columns and active["xs_disp_20d"].notna().any():
                log(f"\n  {'DISPERSION QUARTILE BREAKDOWN':^{W-4}}")
                log("─" * W)
                log(f"  {'quartile':<16} {'signals':>8} {'long prec':>10} {'short prec':>11} {'sig rate':>9}")
                log("─" * W)
                active["disp_q"] = pd.qcut(active["xs_disp_20d"], q=4,
                                            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                            duplicates="drop")
                for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
                    bucket = active[active["disp_q"] == q]
                    if bucket.empty:
                        continue
                    longs  = bucket[bucket["signal"] ==  1]
                    shorts = bucket[bucket["signal"] == -1]
                    lp = longs["correct"].mean()  if len(longs)  > 10 else float("nan")
                    sp = shorts["correct"].mean() if len(shorts) > 10 else float("nan")
                    q_dates = bucket["date"].unique()
                    avail   = total_per_date.reindex(q_dates).sum()
                    rate    = len(bucket) / avail if avail > 0 else float("nan")
                    lp_s = f"{lp:.1%}" if not np.isnan(lp) else "n/a"
                    sp_s = f"{sp:.1%}" if not np.isnan(sp) else "n/a"
                    rt_s = f"{rate:.1%}" if not np.isnan(rate) else "n/a"
                    log(f"  {q:<16} {len(bucket):>8,} {lp_s:>10} {sp_s:>11} {rt_s:>9}")
                log("─" * W)

            log("")
        except Exception as e:
            log(f"error: {e}")

    # ── precision by sector ───────────────────────────────────────────────────

    def _run_sector(self):
        if not PREDS.exists():
            messagebox.showinfo("ra", "run training + backtest first")
            return
        self._tool_clear(self._sector_out)
        self._run(self._sector_worker, self._btn_sector, self._spin_sector)

    def _sector_worker(self):
        log = lambda m: self._tool_log(self._sector_out, m)
        try:
            import numpy as np
            log("loading data...")
            preds = pd.read_parquet(PREDS)
            preds["date"] = pd.to_datetime(preds["date"])

            if not DATA.exists():
                log("featured.parquet not found — run step 01 first")
                return
            feat = pd.read_parquet(DATA, columns=["date", "ticker", "sector_etf"])
            feat["date"] = pd.to_datetime(feat["date"])
            preds = preds.merge(feat, on=["date", "ticker"], how="left")

            active = _mark_correct(preds[preds["signal"] != 0])

            sectors = sorted(active["sector_etf"].dropna().unique())
            total_signals = len(active)

            W = 64
            for side_label, side_val in [("LONG", 1), ("SHORT", -1)]:
                log("\n" + "─" * W)
                log(f"  {side_label + ' PRECISION BY SECTOR':^{W-4}}")
                log("─" * W)
                log(f"  {'sector':<8} {'signals':>8} {'% of total':>11} {'precision':>10} {'edge':>8}")
                log("─" * W)

                side_df = active[active["signal"] == side_val]
                rows = []
                for sec in sectors:
                    sb = side_df[side_df["sector_etf"] == sec]
                    if len(sb) < 10:
                        continue
                    prec  = sb["correct"].mean()
                    share = len(sb) / max(len(side_df), 1)
                    rows.append((sec, len(sb), share, prec))

                rows.sort(key=lambda x: x[3], reverse=True)
                overall = side_df["correct"].mean() if len(side_df) > 0 else float("nan")

                for sec, cnt, share, prec in rows:
                    edge  = prec - 0.50
                    bar   = "█" * max(1, int(edge * 200)) if edge > 0 else "░"
                    log(f"  {sec:<8} {cnt:>8,} {share:>10.1%} {prec:>10.1%}  {bar}")
                log("─" * W)
                log(f"  {'overall':<8} {len(side_df):>8,} {'100.0%':>11} {overall:>10.1%}")
                log("─" * W)

            log("")
        except Exception as e:
            log(f"error: {e}")

    # ── about page ────────────────────────────────────────────────────────────

    def _pg_about(self, parent):
        f = tk.Frame(parent, bg=BG)

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=20, pady=14)
        tk.Label(top, text="about", bg=BG, fg=FG, font=F_H).pack(side="left")

        txt = scrolledtext.ScrolledText(
            f, bg=CARD, fg=FG, font=F_LOG,
            state="normal", relief="flat", borderwidth=0,
            insertbackground=GOLD, selectbackground=LINE,
            wrap="word",
        )
        txt.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        about_text = """\
  ra  —  s&p 500 mean reversion signal system
  ──────────────────────────────────────────────────────

  what it does

  ra identifies s&p 500 stocks likely to outperform the market
  index over the next 21 trading days using machine learning.

  the model (lightgbm) is trained on 5 years of price/volume
  data and learns to rank stocks cross-sectionally — predicting
  which will land in the top 30% of relative performers vs spy.

  ──────────────────────────────────────────────────────
  pipeline
  ──────────────────────────────────────────────────────

    01  data download
        5 years of s&p 500 ohlcv history via yfinance.
        gics sector etf assignments scraped from wikipedia
        (cached locally — only re-fetches after 7 days).
        features: rsi, bollinger bands, z-scores, moving
        averages, volume ratios, atr, gap, vix regime.

    02  model training
        walk-forward cross-validation (8 folds × 3 months).
        target: top 30% vs bottom 30% of 21-day excess return
        vs spy, computed cross-sectionally per trading day.
        the noisy middle 40% is dropped during training.
        key insight: cross-sectional rank features (xs_rank,
        sec_rank) are the primary alpha source. "most oversold
        stock in xlk today" carries far more signal than
        an absolute rsi reading of 30.

    03  signals
        fetches fresh data for all ~503 s&p 500 tickers.
        computes all features on the latest completed bar.
        ranks cross-sectionally across today's universe.
        outputs tickers where model confidence ≥ min threshold.

    04  backtest
        out-of-sample simulation on walk-forward test folds.
        21-day holding period, 16 bps round-trip cost.
        confidence-weighted position sizing.
        sharpe, calmar, max drawdown, win rate reported.

  ──────────────────────────────────────────────────────
  signal columns
  ──────────────────────────────────────────────────────

    sector   gics sector etf proxy for each stock
             xlk  tech       xlf  financials   xlv  health care
             xle  energy     xli  industrials  xly  cons. discret.
             xlp  cons. stap xlu  utilities    xlb  materials
             xlre real estate               xlc  comm. services

    conf     model confidence — probability of top 30% 21d
             relative performance vs spy. ≥65% = gold (high conviction)

    rsi-14   relative strength index (14-day)
             >70 overbought / momentum  |  <30 oversold / reversal

    z-score  20-day price z-score — standard deviations from mean

    sect-z   sector-relative z-score — stock vs its sector etf
             positive = outperforming peers  |  negative = lagging

    bb pos   bollinger band position: 0 = lower band, 1 = upper band

  ──────────────────────────────────────────────────────
  vix & strategy edge
  ──────────────────────────────────────────────────────

    < 15    calm (amber)    trending market, mean reversion weaker
    15–20   normal (white)  moderate, balanced conditions
    20–30   elevated (green) best conditions — reversion strengthens
    > 30    extreme (red)   high risk, chaotic correlations

  ──────────────────────────────────────────────────────
  credits
  ──────────────────────────────────────────────────────

    development            @rawohl

    data sources           yfinance  ·  fred (vix)  ·  wikipedia
    ml stack               lightgbm  ·  scikit-learn  ·  pandas
    ui                     tkinter  ·  python 3.11+

  ──────────────────────────────────────────────────────
"""
        txt.insert("1.0", about_text)
        txt.configure(state="disabled")
        return f


if __name__ == "__main__":
    import sys
    if "--no-gui" in sys.argv:
        from cli import run_once, run_repl
        # one-liner if any real command/flag given alongside --no-gui,
        # otherwise drop into interactive REPL (diskpart-style)
        extra = [a for a in sys.argv[1:] if a != "--no-gui"]
        if extra:
            run_once(sys.argv[1:])
        else:
            run_repl()
    else:
        Ra().mainloop()
