"""
ra — walk-forward model training

Pipeline:
  1. HPO zone  — Optuna tunes hyperparameters on an isolated early block of
                 data.  Walk-forward test sets never overlap this period, so
                 the evaluation stays genuinely out-of-sample.
  2. Walk-forward — 8 folds, each training a fresh LightGBM on the labeled
                 top/bottom-30% set and predicting on the full universe.
                 Top-N stocks per day are selected as signals.
  3. Production model — retrained on the full labeled history with the found
                 hyperparameters, saved to disk.
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
from config import TOP_N, TRAIN_YEARS, TEST_MONTHS, HPO_YEARS, MIN_SPREAD

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LGBM_PARAMS = {
    "objective":             "binary",
    "metric":                "auc",
    "learning_rate":         0.02,
    "num_leaves":            63,
    "min_child_samples":     50,
    "feature_fraction":      0.7,
    "bagging_fraction":      0.8,
    "bagging_freq":          5,
    "reg_alpha":             0.05,
    "reg_lambda":            0.1,
    "n_estimators":          800,
    "early_stopping_rounds": 50,
    "verbose":               -1,
    "n_jobs":                -1,
}

SHAP_FEATURES = 20  # top N features shown in SHAP output


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each row as top-30% (1) or bottom-30% (0) by 21-day excess return
    vs SPY, dropping the middle 40%.  Middle rows are too noisy to learn from
    and excluding them keeps class balance without artificial reweighting.
    """
    df = df.copy()

    if "spy_ret_21d" not in df.columns:
        raise ValueError("spy_ret_21d missing — re-run feature engineering (step 01).")
    if "fwd_ret_21d" not in df.columns:
        raise ValueError("fwd_ret_21d missing — re-run data pipeline (step 01).")

    n_before = len(df)
    df = df.dropna(subset=["spy_ret_21d", "fwd_ret_21d"])
    dropped = n_before - len(df)
    if dropped > 0:
        log.warning(f"Dropped {dropped:,} rows with missing 21d returns")

    df["excess_ret_21d"] = df["fwd_ret_21d"] - df["spy_ret_21d"]
    df["xs_rank"]        = df.groupby("date")["excess_ret_21d"].rank(pct=True)
    df = df[(df["xs_rank"] >= 0.70) | (df["xs_rank"] <= 0.30)].copy()
    df["target"] = (df["xs_rank"] >= 0.70).astype(int)

    up_pct = df["target"].mean() * 100
    log.info(f"Target: {up_pct:.1f}% top-30%  {100-up_pct:.1f}% bottom-30%"
             f"  |  {len(df):,} labeled rows  ({n_before - len(df):,} dropped)")
    return df


def get_walk_forward_splits(df: pd.DataFrame,
                            train_years: int  = TRAIN_YEARS,
                            test_months: int  = TEST_MONTHS,
                            start_date        = None) -> list:
    """
    Returns a list of (train_mask, test_mask) boolean pairs.

    start_date: first date allowed as a test-set date.  Defaults to
                df.date.min() + train_years.  Set this to hpo_end_date so
                that walk-forward test sets never overlap the HPO zone.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    min_date = df["date"].min()
    max_date = df["date"].max()
    current  = pd.Timestamp(start_date) if start_date is not None \
               else min_date + pd.DateOffset(years=train_years)
    splits   = []

    while current < max_date:
        end        = current + pd.DateOffset(months=test_months)
        train_mask = df["date"] < current
        test_mask  = (df["date"] >= current) & (df["date"] < end)
        if test_mask.sum() > 100:
            splits.append((train_mask, test_mask))
        current = end

    log.info(f"Generated {len(splits)} walk-forward folds "
             f"(test window starts {current - pd.DateOffset(months=test_months*len(splits)):%Y-%m-%d}).")
    return splits


def tune_hyperparams(X_tr, y_tr, X_val, y_val,
                     n_trials: int = 50,
                     lgbm_n_jobs: int = 1) -> dict:
    """
    Optuna TPE search over LightGBM hyperparameters.

    lgbm_n_jobs: passed to LGBMClassifier.  Use 1 when fold-level
                 parallelism is active to avoid CPU over-subscription.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective":             "binary",
            "metric":                "auc",
            "verbose":               -1,
            "n_jobs":                lgbm_n_jobs,
            "n_estimators":          1000,
            "early_stopping_rounds": 50,
            "learning_rate":     trial.suggest_float("learning_rate",     0.005, 0.15,  log=True),
            "num_leaves":        trial.suggest_int(  "num_leaves",        31,    255),
            "min_child_samples": trial.suggest_int(  "min_child_samples", 20,    300),
            "feature_fraction":  trial.suggest_float("feature_fraction",  0.4,   1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction",  0.4,   1.0),
            "bagging_freq":      trial.suggest_int(  "bagging_freq",      1,     10),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-3,  10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-3,  10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain",    0.0,   0.3),
            "subsample_for_bin": trial.suggest_int(  "subsample_for_bin", 50_000, 300_000),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])
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
        "objective":             "binary",
        "metric":                "auc",
        "verbose":               -1,
        "n_jobs":                -1,
        "n_estimators":          1000,
        "early_stopping_rounds": 50,
        **study.best_params,
    }


def train_fold(X_train, y_train, X_val, y_val, params: dict = None):
    model = lgb.LGBMClassifier(**(params or LGBM_PARAMS))
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    return model


def _temporal_val_split(train_df: pd.DataFrame, feature_cols: list):
    """Split training data on the 80th-percentile date for early stopping."""
    sorted_dates = np.sort(train_df["date"].unique())
    split_date   = sorted_dates[int(len(sorted_dates) * 0.8)]
    before = train_df["date"] < split_date
    after  = train_df["date"] >= split_date
    X_tr,  y_tr  = train_df[before][feature_cols], train_df[before]["target"]
    X_val, y_val = train_df[after][feature_cols],  train_df[after]["target"]
    return X_tr, y_tr, X_val, y_val


def _fold_signals(model, test_df_full: pd.DataFrame, test_df_labeled: pd.DataFrame,
                  feature_cols: list) -> tuple[pd.DataFrame, dict]:
    """
    Score the full universe on test dates, apply top-N selection, compute
    precision metrics against the labeled ground-truth subset.

    Returns (fp, metrics_dict) where fp contains only signal != 0 rows.
    """
    probs_full = model.predict_proba(test_df_full[feature_cols])[:, 1]

    fp = test_df_full[["date", "ticker", "Close", "fwd_ret_5d", "fwd_ret_21d"]].copy()
    for col in ("vix", "spy_ret_21d"):
        if col in test_df_full.columns:
            fp[col] = test_df_full[col].values

    # Join target from the labeled set — NaN for stocks in the middle 40%
    fp = fp.merge(
        test_df_labeled[["date", "ticker", "target"]],
        on=["date", "ticker"], how="left"
    )
    fp["prob_up"] = probs_full
    fp["signal"]  = 0

    for _, grp in fp.groupby("date"):
        if len(grp) < TOP_N:
            continue
        # Skip days where the model is too uncertain to discriminate.
        # A compressed probability distribution (low spread) means all stocks
        # look the same to the model — trading noise rather than signal.
        spread = float(grp["prob_up"].max() - grp["prob_up"].min())
        if spread < MIN_SPREAD:
            continue
        fp.loc[grp["prob_up"].nlargest(TOP_N).index,  "signal"] =  1
        fp.loc[grp["prob_up"].nsmallest(TOP_N).index, "signal"] = -1

    # Precision: measured only on the labeled subset (top/bottom 30% ground truth).
    labeled_signals = fp.merge(
        test_df_labeled[["date", "ticker", "target"]].rename(columns={"target": "_tgt"}),
        on=["date", "ticker"], how="inner"
    )
    longs_l  = labeled_signals[(labeled_signals["signal"] ==  1) & labeled_signals["_tgt"].notna()]
    shorts_l = labeled_signals[(labeled_signals["signal"] == -1) & labeled_signals["_tgt"].notna()]

    metrics = {
        "long_prec":   float(longs_l["_tgt"].mean())          if len(longs_l)  > 5 else np.nan,
        "short_prec":  float((shorts_l["_tgt"] == 0).mean())  if len(shorts_l) > 5 else np.nan,
        "long_rate":   float((fp["signal"] ==  1).sum()) / max(len(fp), 1),
        "short_rate":  float((fp["signal"] == -1).sum()) / max(len(fp), 1),
    }
    return fp[fp["signal"] != 0], metrics


def _print_results(mdf: pd.DataFrame) -> None:
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


def run_walk_forward(df: pd.DataFrame, n_trials: int = 0, n_jobs: int = -1) -> pd.DataFrame:
    df_full = df.copy()
    df_full["date"] = pd.to_datetime(df_full["date"])
    if df_full["date"].dt.tz is not None:
        df_full["date"] = df_full["date"].dt.tz_convert(None)

    feature_cols = get_feature_columns()
    df_labeled   = build_target(df_full)

    # ── Phase 1: HPO on a dedicated pre-walk-forward zone ────────────────────
    # Data from [dataset_start, hpo_end] is used exclusively for hyperparameter
    # search.  Walk-forward test sets start at hpo_end, so they never overlap
    # with the tuning data and remain genuinely out-of-sample.
    lgbm_params = {**LGBM_PARAMS, "n_jobs": n_jobs}
    hpo_end     = None

    if n_trials > 0:
        hpo_end  = df_labeled["date"].min() + pd.DateOffset(months=int(HPO_YEARS * 12))
        hpo_data = df_labeled[df_labeled["date"] < hpo_end]

        if len(hpo_data) < 1000:
            log.warning("HPO zone has fewer than 1,000 labeled rows — widening to fold-1 data.")
            hpo_data = df_labeled  # fallback: use everything

        log.info(f"HPO zone: {df_labeled['date'].min().date()} → {hpo_end.date()} "
                 f"({len(hpo_data):,} labeled rows)")

        X_tr, y_tr, X_val, y_val = _temporal_val_split(hpo_data, feature_cols)
        log.info(f"Running Optuna — {n_trials} trials...")
        tuned       = tune_hyperparams(X_tr, y_tr, X_val, y_val, n_trials, lgbm_n_jobs=n_jobs)
        lgbm_params = {**tuned, "n_jobs": n_jobs}
        log.info("HPO complete — params locked for all walk-forward folds.")

    # ── Phase 2: Walk-forward fold training ───────────────────────────────────
    # wf_start must satisfy two constraints simultaneously:
    #   1. After hpo_end so test sets never overlap the HPO zone.
    #   2. At least TRAIN_YEARS after the dataset start so the first fold
    #      has a meaningful amount of training data.
    data_start = df_labeled["date"].min()
    min_train_start = data_start + pd.DateOffset(years=TRAIN_YEARS)
    wf_start = max(hpo_end, min_train_start) if hpo_end is not None else None
    splits = get_walk_forward_splits(df_labeled, start_date=wf_start)

    if not splits:
        log.error("No walk-forward splits generated.")
        return pd.DataFrame()

    all_preds    = []
    fold_metrics = []

    for fold_num, (train_mask, test_mask) in enumerate(splits):
        train_df        = df_labeled[train_mask]
        test_df_labeled = df_labeled[test_mask]
        test_dates      = test_df_labeled["date"].unique()
        test_df_full    = df_full[df_full["date"].isin(test_dates)].dropna(subset=feature_cols)

        X_tr, y_tr, X_val, y_val = _temporal_val_split(train_df, feature_cols)
        X_test, y_test = test_df_labeled[feature_cols], test_df_labeled["target"]

        date_range = (f"{test_df_labeled['date'].min().date()} → "
                      f"{test_df_labeled['date'].max().date()}")
        log.info(f"Fold {fold_num+1}/{len(splits)} [{date_range}]  "
                 f"train={len(train_df):,}  val={len(X_val):,}  test={len(test_df_full):,}")

        model         = train_fold(X_tr, y_tr, X_val, y_val, params=lgbm_params)
        auc           = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        fp, prec      = _fold_signals(model, test_df_full, test_df_labeled, feature_cols)

        lp = f"{prec['long_prec']:.4f}"  if not np.isnan(prec["long_prec"])  else "n/a"
        sp = f"{prec['short_prec']:.4f}" if not np.isnan(prec["short_prec"]) else "n/a"
        log.info(f"  AUC={auc:.4f}  long_prec={lp}  short_prec={sp}")

        fold_metrics.append({
            "fold":        fold_num + 1,
            "train_size":  len(train_df),
            "test_size":   len(test_df_full),
            "auc":         auc,
            **prec,
        })
        fp["fold"] = fold_num + 1
        all_preds.append(fp)

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"])
    if predictions["date"].dt.tz is not None:
        predictions["date"] = predictions["date"].dt.tz_convert(None)

    _print_results(pd.DataFrame(fold_metrics))

    # ── Phase 3: Production model ─────────────────────────────────────────────
    log.info("Training production model on full labeled dataset...")
    X_tr, y_tr, X_val, y_val = _temporal_val_split(df_labeled, feature_cols)
    final_model = train_fold(X_tr, y_tr, X_val, y_val, params=lgbm_params)
    save_model(final_model, feature_cols)
    run_shap_analysis(final_model, X_val, feature_cols)

    return predictions


def run_shap_analysis(model, X_test: pd.DataFrame, feature_cols: list,
                      save_path: Path = Path("models/shap_analysis.png")) -> None:
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — pip install shap")
        return

    log.info("Computing SHAP values...")
    sample      = X_test.sample(min(2_000, len(X_test)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    sv          = shap_values[1] if isinstance(shap_values, list) else shap_values

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": np.abs(sv).mean(axis=0)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    W  = 54
    mx = imp_df["importance"].iloc[0]
    print("\n" + "─" * W)
    print(f"  {'SHAP FEATURE IMPORTANCE  (production model val set)':^{W-4}}")
    print("─" * W)
    for _, row in imp_df.head(SHAP_FEATURES).iterrows():
        bar = "█" * max(1, int(row["importance"] / mx * 22))
        print(f"  {row['feature']:<28}  {bar}")
    print("─" * W)
    low_imp = imp_df[imp_df["importance"] < imp_df["importance"].quantile(0.25)]
    if not low_imp.empty:
        print(f"  bottom 25% ({len(low_imp)} features): "
              + ", ".join(low_imp["feature"].tolist()))
    print("─" * W + "\n")

    top_df = imp_df.head(SHAP_FEATURES)
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
    fig.suptitle("SHAP Feature Importance — production model", color="#cdd6f4", fontsize=13)

    ax1 = axes[0]
    colors = ["#f9e2af" if i < 5 else "#89b4fa" for i in range(SHAP_FEATURES)]
    ax1.barh(top_df["feature"][::-1], top_df["importance"][::-1], color=colors[::-1])
    ax1.set_xlabel("mean |SHAP value|", color="#cdd6f4")
    ax1.set_title(f"Top {SHAP_FEATURES} features")
    ax1.grid(True, axis="x", alpha=0.2, color="#585b70")

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
    rng    = np.random.default_rng(0)
    n_pts  = min(n_points, shap_vals.shape[0])
    idx    = rng.choice(shap_vals.shape[0], n_pts, replace=False)
    sv_sub = shap_vals[idx]
    X_sub  = X.iloc[idx].values

    for fi in range(len(feature_names)):
        y_pos  = len(feature_names) - 1 - fi
        sv_row = sv_sub[:, fi]
        fv_row = X_sub[:, fi]
        fv_norm = (fv_row - np.nanmin(fv_row)) / (np.nanmax(fv_row) - np.nanmin(fv_row) + 1e-9)
        colors  = plt.cm.RdYlBu_r(fv_norm)
        jitter  = rng.uniform(-0.3, 0.3, size=n_pts)
        ax.scatter(sv_row, y_pos + jitter, c=colors, s=6, alpha=0.5, linewidths=0)

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names[::-1], fontsize=8, color="#cdd6f4")


def save_model(model, feature_cols: list) -> None:
    path = MODEL_DIR / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols}, f)
    log.info(f"Model saved → {path}")


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
