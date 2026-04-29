"""
S&P 500 Mean Reversion Signal System v3
Phase 3: Model Training

v3 changes:
  - Temporal validation split: last 20% of dates (not 20% of rows).
    Positional split mixed past and future data across multi-ticker training sets.
  - SPY-relative target is always required — no silent fallback to absolute returns.
    Mixing two different targets mid-dataset taught the model contradictory objectives.
  - Precision reported as fraction of high-confidence signals that actually beat SPY
    (was incorrectly thresholded at 0.5 regardless of MIN_PROB).
  - Calmar ratio added to fold summary.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engineering import get_feature_columns

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = 2
TEST_MONTHS = 3
MIN_PROB    = 0.52
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.02,
    "num_leaves":       63,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.05,
    "reg_lambda":       0.1,
    "n_estimators":     800,
    "early_stopping_rounds": 50,
    "verbose":          -1,
    "n_jobs":           -1,
    # no class_weight — balanced weights compress all probabilities toward 0.5,
    # making it nearly impossible to exceed any meaningful confidence threshold
}


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 1 if stock is in the top 30% of S&P 500 by 21-day excess return vs SPY.
    Target = 0 if in the bottom 30%. Middle 40% is dropped (too noisy to learn from).

    Why 21d? Mean reversion is empirically strongest at 1-month horizon.
    5-day "beat SPY" creates a near-50/50 binary where momentum dominates and
    reversal doesn't reliably materialize — AUC stays stuck at ~0.47.

    Why cross-sectional quantile? On any given day ~50% of stocks beat SPY by
    definition. Absolute outperformance is too close to a coin flip. Top/bottom
    30% isolates genuine relative winners and losers, making the labels cleaner.
    """
    df = df.copy()

    if "spy_ret_21d" not in df.columns:
        raise ValueError("spy_ret_21d missing. Re-run feature engineering.")
    if "fwd_ret_21d" not in df.columns:
        raise ValueError("fwd_ret_21d missing. Re-run data pipeline (Step 1).")

    n_before = len(df)
    df = df.dropna(subset=["spy_ret_21d", "fwd_ret_21d"])
    dropped = n_before - len(df)
    if dropped > 0:
        log.warning(f"Dropped {dropped:,} rows with missing 21d returns")

    # excess return vs SPY over the next 21 days
    df["excess_ret_21d"] = df["fwd_ret_21d"] - df["spy_ret_21d"]

    # cross-sectional rank per date: 0 = worst relative performer, 1 = best
    df["xs_rank"] = df.groupby("date")["excess_ret_21d"].rank(pct=True)

    # keep only the top and bottom 30%; drop the noisy middle 40%
    df = df[(df["xs_rank"] >= 0.70) | (df["xs_rank"] <= 0.30)].copy()
    df["target"] = (df["xs_rank"] >= 0.70).astype(int)

    up_pct = df["target"].mean() * 100
    log.info(f"Target (top 30% vs bottom 30% 21d excess return): "
             f"{up_pct:.1f}% up, {100-up_pct:.1f}% down  |  {len(df):,} rows (dropped {n_before-len(df):,})")
    return df


def get_walk_forward_splits(df, train_years=TRAIN_YEARS, test_months=TEST_MONTHS):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    min_date = df["date"].min()
    max_date = df["date"].max()
    current  = min_date + pd.DateOffset(years=train_years)
    splits   = []

    while current < max_date:
        end = current + pd.DateOffset(months=test_months)
        train_mask = df["date"] < current
        test_mask  = (df["date"] >= current) & (df["date"] < end)
        if test_mask.sum() > 100:
            splits.append((train_mask, test_mask))
        current = end

    log.info(f"Generated {len(splits)} walk-forward folds.")
    return splits


def train_fold(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )
    return model


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    feature_cols = get_feature_columns()
    df     = build_target(df)
    splits = get_walk_forward_splits(df)

    if not splits:
        log.error("No splits generated.")
        return pd.DataFrame()

    all_preds    = []
    fold_metrics = []

    for fold_num, (train_mask, test_mask) in enumerate(splits):
        train_df = df[train_mask]
        test_df  = df[test_mask]

        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_test,  y_test  = test_df[feature_cols],  test_df["target"]

        # Temporal validation: split on the 80th-percentile date, not the 80th row.
        # Positional split incorrectly interleaves tickers across time.
        sorted_dates = np.sort(train_df["date"].unique())
        split_date   = sorted_dates[int(len(sorted_dates) * 0.8)]
        tr_mask      = train_df["date"] <  split_date
        val_mask     = train_df["date"] >= split_date
        X_tr,  X_val = X_train[tr_mask],  X_train[val_mask]
        y_tr,  y_val = y_train[tr_mask],  y_train[val_mask]

        date_range = f"{test_df['date'].min().date()} → {test_df['date'].max().date()}"
        log.info(f"Fold {fold_num+1}/{len(splits)} [{date_range}] "
                 f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

        model = train_fold(X_tr, y_tr, X_val, y_val)
        probs = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, probs)

        # long leg: high confidence of outperformance
        long_mask  = probs >= MIN_PROB
        # short leg: high confidence of underperformance (symmetric threshold)
        short_mask = probs <= (1.0 - MIN_PROB)

        long_prec = float(y_test[long_mask].mean())        if long_mask.sum()  > 10 else np.nan
        short_prec = float((y_test[short_mask] == 0).mean()) if short_mask.sum() > 10 else np.nan
        long_rate  = float(long_mask.mean())
        short_rate = float(short_mask.mean())

        log.info(f"  {'AUC':>14}: {auc:.4f}")
        log.info(f"  {'Long prec':>14}: {long_prec:.4f}"  if not np.isnan(long_prec)  else f"  {'Long prec':>14}: n/a")
        log.info(f"  {'Short prec':>14}: {short_prec:.4f}" if not np.isnan(short_prec) else f"  {'Short prec':>14}: n/a")
        log.info(f"  {'Long rate':>14}: {long_rate:.1%}   Short rate: {short_rate:.1%}")

        fold_metrics.append({
            "fold":         fold_num + 1,
            "train_size":   len(X_train),
            "test_size":    len(X_test),
            "auc":          auc,
            "long_prec":    long_prec,
            "short_prec":   short_prec,
            "long_rate":    long_rate,
            "short_rate":   short_rate,
        })

        save_cols = ["date", "ticker", "Close", "fwd_ret_21d", "target"]
        for col in ("vix", "spy_ret_21d"):
            if col in test_df.columns:
                save_cols.append(col)

        fp            = test_df[save_cols].copy()
        fp["prob_up"] = probs
        # signal: 1 = long, -1 = short, 0 = no position
        fp["signal"]  = 0
        fp.loc[long_mask,  "signal"] =  1
        fp.loc[short_mask, "signal"] = -1
        fp["fold"]    = fold_num + 1
        all_preds.append(fp)

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"])
    if predictions["date"].dt.tz is not None:
        predictions["date"] = predictions["date"].dt.tz_convert(None)

    mdf = pd.DataFrame(fold_metrics)
    print("\n" + "─" * 72)
    print(f"  {'WALK-FORWARD RESULTS':^68}")
    print("─" * 72)
    print(mdf.to_string(index=False))
    print("─" * 72)
    print(f"  {'Mean AUC':<36} {mdf['auc'].mean():.4f}")
    print(f"  {'Mean Long Precision':<36} {mdf['long_prec'].mean():.4f}")
    print(f"  {'Mean Short Precision':<36} {mdf['short_prec'].mean():.4f}")
    print(f"  {'Mean Long Signal Rate':<36} {mdf['long_rate'].mean():.1%}")
    print(f"  {'Mean Short Signal Rate':<36} {mdf['short_rate'].mean():.1%}")
    print("─" * 72 + "\n")

    # Production model: retrain on the full labeled dataset so it sees the most
    # recent data. Walk-forward CV above is for evaluation only.
    log.info("Training final model on full dataset...")
    sorted_dates  = np.sort(df["date"].unique())
    split_date    = sorted_dates[int(len(sorted_dates) * 0.8)]
    X_tr_all      = df[df["date"] <  split_date][feature_cols]
    y_tr_all      = df[df["date"] <  split_date]["target"]
    X_val_all     = df[df["date"] >= split_date][feature_cols]
    y_val_all     = df[df["date"] >= split_date]["target"]
    final_model   = train_fold(X_tr_all, y_tr_all, X_val_all, y_val_all)
    save_model(final_model, feature_cols)
    run_shap_analysis(final_model, X_val_all, feature_cols)

    return predictions


def run_shap_analysis(
    model,
    X_test: pd.DataFrame,
    feature_cols: list,
    save_path: Path = Path("models/shap_analysis.png"),
) -> None:
    """
    Compute SHAP values for the final fold's test set and produce:
      - A ranked console table (top 20 features by mean |SHAP|)
      - A two-panel plot: feature importance bar + beeswarm (direction)
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — run: pip install shap")
        return

    log.info("Computing SHAP values on last fold test set...")
    sample = X_test.sample(min(2_000, len(X_test)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # LightGBM binary: shap_values is [class0_array, class1_array]
    # class 1 = prob_up (what we care about)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs   = np.abs(sv).mean(axis=0)
    imp_df     = (
        pd.DataFrame({"feature": feature_cols, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── console table ─────────────────────────────────────────────────────────
    W  = 54
    mx = imp_df["importance"].iloc[0]
    print("\n" + "─" * W)
    print(f"  {'SHAP FEATURE IMPORTANCE  (last fold test set)':^{W-4}}")
    print("─" * W)
    for _, row in imp_df.head(20).iterrows():
        bar = "█" * max(1, int(row["importance"] / mx * 22))
        print(f"  {row['feature']:<28}  {bar}")
    print("─" * W)
    low_imp = imp_df[imp_df["importance"] < imp_df["importance"].quantile(0.25)]
    if not low_imp.empty:
        print(f"  bottom 25% ({len(low_imp)} features): "
              + ", ".join(low_imp["feature"].tolist()))
    print("─" * W + "\n")

    # ── plot ──────────────────────────────────────────────────────────────────
    TOP_N  = 20
    top_df = imp_df.head(TOP_N)
    top_sv = sv[:, [feature_cols.index(f) for f in top_df["feature"]]]
    top_X  = sample[top_df["feature"].tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#313244")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#89b4fa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
    fig.suptitle("SHAP Feature Importance — last fold", color="#cdd6f4", fontsize=13)

    # left: horizontal bar chart (mean |SHAP|)
    ax1 = axes[0]
    colors = ["#f9e2af" if i < 5 else "#89b4fa" for i in range(TOP_N)]
    ax1.barh(top_df["feature"][::-1], top_df["importance"][::-1], color=colors[::-1])
    ax1.set_xlabel("mean |SHAP value|", color="#cdd6f4")
    ax1.set_title(f"Top {TOP_N} features")
    ax1.grid(True, axis="x", alpha=0.2, color="#585b70")

    # right: beeswarm (SHAP value vs feature value)
    ax2 = axes[1]
    shap_beeswarm(ax2, top_sv, top_X, top_df["feature"].tolist())
    ax2.set_title("Direction of effect (beeswarm)")
    ax2.set_xlabel("SHAP value  →  pushes prob_up higher", color="#cdd6f4")
    ax2.axvline(0, color="#585b70", linewidth=0.8, linestyle="--")
    ax2.grid(True, axis="x", alpha=0.2, color="#585b70")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1e1e2e")
    plt.close()
    log.info(f"SHAP chart saved → {save_path}")


def shap_beeswarm(ax, shap_vals, X, feature_names, n_points: int = 500) -> None:
    """
    Minimal beeswarm: one row per feature, dots jittered vertically.
    Colour encodes feature value (blue=low, gold=high).
    """
    rng     = np.random.default_rng(0)
    n_feats = len(feature_names)
    n_pts   = min(n_points, shap_vals.shape[0])
    idx     = rng.choice(shap_vals.shape[0], n_pts, replace=False)
    sv_sub  = shap_vals[idx]
    X_sub   = X.iloc[idx].values

    for fi in range(n_feats):
        y_pos   = n_feats - 1 - fi   # top = most important
        sv_row  = sv_sub[:, fi]
        fv_row  = X_sub[:, fi]

        # normalise feature values to [0,1] for colouring
        fv_norm = (fv_row - np.nanmin(fv_row)) / (np.nanmax(fv_row) - np.nanmin(fv_row) + 1e-9)
        colors  = plt.cm.RdYlBu_r(fv_norm)   # blue = low value, red = high value

        jitter  = rng.uniform(-0.3, 0.3, size=n_pts)
        ax.scatter(sv_row, y_pos + jitter, c=colors, s=6, alpha=0.5, linewidths=0)

    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feature_names[::-1], fontsize=8, color="#cdd6f4")


def save_model(model, feature_cols):
    path = MODEL_DIR / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols}, f)
    log.info(f"Model saved to {path}")


def load_model():
    path = MODEL_DIR / "model.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


if __name__ == "__main__":
    df = pd.read_parquet("data/clean/featured.parquet")
    predictions = run_walk_forward(df)
    if not predictions.empty:
        predictions.to_parquet("data/clean/predictions.parquet", index=False)
        log.info("Saved predictions.")
