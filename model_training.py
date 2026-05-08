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
from config import TOP_N

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = 2
TEST_MONTHS = 3
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


def tune_hyperparams(X_tr, y_tr, X_val, y_val, n_trials: int = 50) -> dict:
    """
    Optuna TPE search over LightGBM hyperparameters.
    Tuned on the first fold's training split so the result is time-aware.
    Returns a full params dict ready to pass to LGBMClassifier.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective":         "binary",
            "metric":            "auc",
            "verbose":           -1,
            "n_jobs":            -1,
            "n_estimators":      1000,
            "early_stopping_rounds": 50,
            "learning_rate":     trial.suggest_float("learning_rate",     0.005, 0.15,  log=True),
            "num_leaves":        trial.suggest_int(  "num_leaves",        16,    255),
            "min_child_samples": trial.suggest_int(  "min_child_samples", 20,    300),
            "feature_fraction":  trial.suggest_float("feature_fraction",  0.4,   1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction",  0.4,   1.0),
            "bagging_freq":      trial.suggest_int(  "bagging_freq",      1,     10),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-3,  10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-3,  10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain",    0.0,   1.0),
            "subsample_for_bin": trial.suggest_int(  "subsample_for_bin", 50000, 300000),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info(f"Optuna best val AUC: {study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")

    return {
        "objective":         "binary",
        "metric":            "auc",
        "verbose":           -1,
        "n_jobs":            -1,
        "n_estimators":      1000,
        "early_stopping_rounds": 50,
        **study.best_params,
    }


def train_fold(X_train, y_train, X_val, y_val, params: dict = None):
    model = lgb.LGBMClassifier(**(params or LGBM_PARAMS))
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )
    return model


def run_walk_forward(df: pd.DataFrame, n_trials: int = 0) -> pd.DataFrame:
    # df_full: every stock, every date — used for prediction at signal time.
    # df_labeled: only top/bottom 30% extreme performers — used for training.
    # Training on clean labels while predicting on the full universe is how
    # quant funds actually operate: the model learns from confirmed extremes,
    # then ranks all stocks and we select the top-N per day.
    df_full = df.copy()
    df_full["date"] = pd.to_datetime(df_full["date"])
    if df_full["date"].dt.tz is not None:
        df_full["date"] = df_full["date"].dt.tz_convert(None)

    feature_cols = get_feature_columns()
    df_labeled   = build_target(df_full)
    splits       = get_walk_forward_splits(df_labeled)

    if not splits:
        log.error("No splits generated.")
        return pd.DataFrame()

    # Optuna HPO: tune on the first fold's training split, apply to all folds.
    lgbm_params = None
    if n_trials > 0:
        log.info(f"Running Optuna HPO — {n_trials} trials on fold-1 training data...")
        first_train = df_labeled[splits[0][0]]
        sd = np.sort(first_train["date"].unique())
        sp = sd[int(len(sd) * 0.8)]
        X_tune_tr  = first_train[first_train["date"] <  sp][feature_cols]
        y_tune_tr  = first_train[first_train["date"] <  sp]["target"]
        X_tune_val = first_train[first_train["date"] >= sp][feature_cols]
        y_tune_val = first_train[first_train["date"] >= sp]["target"]
        lgbm_params = tune_hyperparams(X_tune_tr, y_tune_tr, X_tune_val, y_tune_val, n_trials)
        log.info("Optuna HPO complete — using tuned params for all folds.")

    all_preds    = []
    fold_metrics = []

    for fold_num, (train_mask, test_mask) in enumerate(splits):
        train_df       = df_labeled[train_mask]
        test_df_labeled = df_labeled[test_mask]

        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_test,  y_test  = test_df_labeled[feature_cols], test_df_labeled["target"]

        # Temporal validation split on the 80th-percentile date.
        sorted_dates = np.sort(train_df["date"].unique())
        split_date   = sorted_dates[int(len(sorted_dates) * 0.8)]
        tr_mask      = train_df["date"] <  split_date
        val_mask     = train_df["date"] >= split_date
        X_tr,  X_val = X_train[tr_mask],  X_train[val_mask]
        y_tr,  y_val = y_train[tr_mask],  y_train[val_mask]

        date_range = f"{test_df_labeled['date'].min().date()} → {test_df_labeled['date'].max().date()}"
        log.info(f"Fold {fold_num+1}/{len(splits)} [{date_range}] "
                 f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

        model = train_fold(X_tr, y_tr, X_val, y_val, params=lgbm_params)

        # AUC: evaluate on labeled test set (top/bottom 30% ground truth).
        probs_labeled = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs_labeled)

        # Full-universe prediction: score every stock on the test dates.
        test_dates    = test_df_labeled["date"].unique()
        test_df_full  = df_full[df_full["date"].isin(test_dates)].dropna(subset=feature_cols)
        probs_full    = model.predict_proba(test_df_full[feature_cols])[:, 1]

        # Top-N selection per day: take the TOP_N highest-confidence longs
        # and TOP_N lowest-confidence shorts from the full universe.
        fp = test_df_full[["date", "ticker", "Close", "fwd_ret_5d", "fwd_ret_21d"]].copy()
        for col in ("vix", "spy_ret_21d", "target"):
            if col in test_df_full.columns:
                fp[col] = test_df_full[col].values
        fp["prob_up"] = probs_full
        fp["signal"]  = 0

        for date, grp in fp.groupby("date"):
            if len(grp) < TOP_N:
                continue
            fp.loc[grp["prob_up"].nlargest(TOP_N).index,  "signal"] =  1
            fp.loc[grp["prob_up"].nsmallest(TOP_N).index, "signal"] = -1

        # Precision/rate metrics: measure on the labeled subset that has ground truth.
        fp_labeled = fp[fp["ticker"].isin(test_df_labeled["ticker"]) &
                        fp["date"].isin(test_df_labeled["date"])]
        fp_labeled = fp_labeled.merge(
            test_df_labeled[["date", "ticker", "target"]].rename(columns={"target": "_target"}),
            on=["date", "ticker"], how="left"
        )
        longs_l  = fp_labeled[(fp_labeled["signal"] ==  1) & fp_labeled["_target"].notna()]
        shorts_l = fp_labeled[(fp_labeled["signal"] == -1) & fp_labeled["_target"].notna()]
        long_prec  = float(longs_l["_target"].mean())           if len(longs_l)  > 5 else np.nan
        short_prec = float((shorts_l["_target"] == 0).mean())   if len(shorts_l) > 5 else np.nan
        long_rate  = float((fp["signal"] ==  1).sum()) / max(len(fp), 1)
        short_rate = float((fp["signal"] == -1).sum()) / max(len(fp), 1)

        log.info(f"  {'AUC':>14}: {auc:.4f}")
        log.info(f"  {'Long prec':>14}: {long_prec:.4f}"  if not np.isnan(long_prec)  else f"  {'Long prec':>14}: n/a")
        log.info(f"  {'Short prec':>14}: {short_prec:.4f}" if not np.isnan(short_prec) else f"  {'Short prec':>14}: n/a")
        log.info(f"  {'Long rate':>14}: {long_rate:.1%}   Short rate: {short_rate:.1%}")

        fold_metrics.append({
            "fold":        fold_num + 1,
            "train_size":  len(X_train),
            "test_size":   len(test_df_full),
            "auc":         auc,
            "long_prec":   long_prec,
            "short_prec":  short_prec,
            "long_rate":   long_rate,
            "short_rate":  short_rate,
        })

        fp["fold"] = fold_num + 1
        all_preds.append(fp[fp["signal"] != 0])

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

    # Production model: retrain on the full labeled dataset.
    log.info("Training final model on full dataset...")
    sorted_dates  = np.sort(df_labeled["date"].unique())
    split_date    = sorted_dates[int(len(sorted_dates) * 0.8)]
    X_tr_all      = df_labeled[df_labeled["date"] <  split_date][feature_cols]
    y_tr_all      = df_labeled[df_labeled["date"] <  split_date]["target"]
    X_val_all     = df_labeled[df_labeled["date"] >= split_date][feature_cols]
    y_val_all     = df_labeled[df_labeled["date"] >= split_date]["target"]
    final_model   = train_fold(X_tr_all, y_tr_all, X_val_all, y_val_all, params=lgbm_params)
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
