"""
analysis.py ─ Industrial Predictive Maintenance ML Pipeline
Stateful module: functions must run in order:
  load → preprocess → eda → split → train → tune → evaluate → features → errors
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    auc, f1_score, recall_score, precision_score
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# Global pipeline state (in-memory)
# ─────────────────────────────────────────────
_state = {}


def _py(v):
    """Convert numpy scalars to native Python for JSON serialisation."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def get_status():
    return {
        "loaded":      "df"               in _state,
        "preprocessed":"df_clean"         in _state,
        "eda":         "eda_done"         in _state,
        "split":       "X_train"          in _state,
        "trained":     "fitted_pipelines" in _state,
        "tuned":       "best_model"       in _state,
        "evaluated":   "test_pred"        in _state,
        "features":    "feat_imp"         in _state,
        "errors":      "error_summary"    in _state,
    }


# ─────────────────────────────────────────────
# STEP 1 · Load dataset
# ─────────────────────────────────────────────
def run_load():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    base = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(base, "ai4i2020.csv")

    if os.path.exists(local):
        df = pd.read_csv(local)
        cached = True
    else:
        try:
            df = pd.read_csv(url)
            df.to_csv(local, index=False)
            cached = False
        except Exception as exc:
            raise RuntimeError(
                f"Download failed: {exc}\n"
                "Place ai4i2020.csv in the IDSE folder and retry."
            )

    _state["df"] = df

    vc     = df["Machine failure"].value_counts()
    vc_pct = df["Machine failure"].value_counts(normalize=True) * 100

    LEAKAGE_COLS = {"TWF", "HDF", "PWF", "OSF", "RNF"}
    ID_COLS      = {"UDI", "Product ID"}
    TARGET_COL   = "Machine failure"

    dtype_rows = []
    for col in df.columns:
        try:
            sample = str(df[col].iloc[0])
        except Exception:
            sample = "N/A"
        role = "Target" if col == TARGET_COL else \
               "Leakage" if col in LEAKAGE_COLS else \
               "ID" if col in ID_COLS else "Feature"
        dtype_rows.append({
            "column":   col,
            "dtype":    str(df[col].dtype),
            "non_null": _py(df[col].notna().sum()),
            "unique":   _py(df[col].nunique()),
            "sample":   sample,
            "role":     role,
        })

    return {
        "shape":    {"rows": _py(df.shape[0]), "cols": _py(df.shape[1])},
        "missing":  _py(df.isna().sum().sum()),
        "duplicates": _py(df.duplicated().sum()),
        "cached":   cached,
        "class_balance": {
            "healthy_count": _py(vc.get(0, 0)),
            "failure_count": _py(vc.get(1, 0)),
            "healthy_pct":   round(float(vc_pct.get(0, 0)), 2),
            "failure_pct":   round(float(vc_pct.get(1, 0)), 2),
        },
        "dtypes": dtype_rows,
    }


# ─────────────────────────────────────────────
# STEP 2 · Preprocess & Feature Engineer
# ─────────────────────────────────────────────
def run_preprocess():
    if "df" not in _state:
        raise RuntimeError("Run data loading first.")
    df = _state["df"]

    leakage_and_ids = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df_c = df.drop(columns=leakage_and_ids).copy()

    df_c["Temp_Diff_K"]      = df_c["Process temperature [K]"] - df_c["Air temperature [K]"]
    df_c["Power_W"]          = df_c["Torque [Nm]"] * (df_c["Rotational speed [rpm]"] * 2 * np.pi / 60.0)
    df_c["Overstrain_Index"] = df_c["Tool wear [min]"] * df_c["Torque [Nm]"]

    _state["df_clean"] = df_c

    num_cols = df_c.select_dtypes(include=[np.number]).columns.drop("Machine failure", errors="ignore")
    stats    = df_c[num_cols].describe().round(3)
    stats_rows = []
    for col in num_cols:
        stats_rows.append({
            "feature":       col,
            "min":           float(stats.loc["min",   col]),
            "mean":          float(stats.loc["mean",  col]),
            "median":        float(stats.loc["50%",   col]),
            "max":           float(stats.loc["max",   col]),
            "std":           float(stats.loc["std",   col]),
            "is_engineered": col in {"Temp_Diff_K", "Power_W", "Overstrain_Index"},
        })

    return {
        "columns_removed": leakage_and_ids,
        "new_features": [
            {
                "name":    "Temp_Diff_K",
                "formula": "Process temp [K] − Air temp [K]",
                "physics": "Heat dissipation margin. Small gap → cooling failure → overheating.",
            },
            {
                "name":    "Power_W",
                "formula": "Torque [Nm] × (RPM × 2π / 60)",
                "physics": "True mechanical power (W). Captures both stalling and overspeed.",
            },
            {
                "name":    "Overstrain_Index",
                "formula": "Tool wear [min] × Torque [Nm]",
                "physics": "Cumulative strain: a dull tool under high force is primed to snap.",
            },
        ],
        "shape":  {"rows": _py(df_c.shape[0]), "cols": _py(df_c.shape[1])},
        "stats":  stats_rows,
    }


# ─────────────────────────────────────────────
# STEP 3 · Exploratory Data Analysis
# ─────────────────────────────────────────────
def run_eda():
    if "df_clean" not in _state:
        raise RuntimeError("Run preprocessing first.")
    df_c  = _state["df_clean"]
    df_or = _state["df"]

    healthy = df_c[df_c["Machine failure"] == 0]
    failure = df_c[df_c["Machine failure"] == 1]
    h_samp  = healthy.sample(n=min(1500, len(healthy)), random_state=RANDOM_STATE)

    BOX_FEATURES = {
        "Power_W":                 {"label": "Mechanical Power",   "unit": "W"},
        "Overstrain_Index":        {"label": "Overstrain Index",   "unit": "min·Nm"},
        "Temp_Diff_K":             {"label": "Temperature Diff",   "unit": "K"},
        "Torque [Nm]":             {"label": "Torque",             "unit": "Nm"},
        "Rotational speed [rpm]":  {"label": "Rotational Speed",   "unit": "rpm"},
        "Tool wear [min]":         {"label": "Tool Wear",          "unit": "min"},
    }
    box_data = {}
    for feat, meta in BOX_FEATURES.items():
        box_data[feat] = {
            "label":   meta["label"],
            "unit":    meta["unit"],
            "healthy": h_samp[feat].round(3).tolist(),
            "failure": failure[feat].round(3).tolist(),
        }

    # Correlation
    RENAME = {
        "Air temperature [K]":    "Air Temp",
        "Process temperature [K]":"Proc Temp",
        "Rotational speed [rpm]": "RPM",
        "Torque [Nm]":            "Torque",
        "Tool wear [min]":        "Tool Wear",
        "Temp_Diff_K":            "Temp Diff",
        "Power_W":                "Power (W)",
        "Overstrain_Index":       "Overstrain",
        "Machine failure":        "Failure",
    }
    num_df = df_c.select_dtypes(include=[np.number]).rename(columns=RENAME)
    corr   = num_df.corr().round(2)

    # Type distribution
    tg = df_c.groupby("Type")["Machine failure"].agg(["sum", "count"])
    tg["healthy"]      = tg["count"] - tg["sum"]
    tg["failure_rate"] = (tg["sum"] / tg["count"] * 100).round(2)

    # Failure sub-modes (from original df)
    fm = df_or[df_or["Machine failure"] == 1][["TWF","HDF","PWF","OSF","RNF"]].sum()

    _state["eda_done"] = True
    return {
        "class_balance": {
            "counts": [_py(len(healthy)), _py(len(failure))],
            "labels": ["Healthy (0)", "Failure (1)"],
        },
        "box_data": box_data,
        "correlation": {
            "z":    corr.values.tolist(),
            "x":    corr.columns.tolist(),
            "y":    corr.index.tolist(),
            "text": [[f"{v:.2f}" for v in row] for row in corr.values.tolist()],
        },
        "type_distribution": {
            "types":          tg.index.tolist(),
            "healthy":        [_py(v) for v in tg["healthy"].tolist()],
            "failure":        [_py(v) for v in tg["sum"].tolist()],
            "failure_rates":  [float(v) for v in tg["failure_rate"].tolist()],
        },
        "failure_modes": {
            "labels": fm.index.tolist(),
            "counts": [_py(v) for v in fm.values.tolist()],
        },
    }


# ─────────────────────────────────────────────
# STEP 4 · Stratified Train / Val / Test Split
# ─────────────────────────────────────────────
def run_split():
    if "df_clean" not in _state:
        raise RuntimeError("Run preprocessing first.")
    df_c = _state["df_clean"]

    X = df_c.drop(columns=["Machine failure"])
    y = df_c["Machine failure"]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.17647, stratify=y_tmp, random_state=RANDOM_STATE
    )

    num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = ["Type"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_feats),
    ])

    _state.update({
        "X": X, "y": y,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "num_feats": num_feats, "cat_feats": cat_feats,
        "preprocessor": preprocessor,
    })

    def _split_info(Xs, ys, pct):
        return {
            "rows":        _py(len(Xs)),
            "failures":    _py(int(ys.sum())),
            "failure_pct": round(float(ys.mean() * 100), 2),
            "pct_of_total": pct,
        }

    return {
        "total":  _py(len(X)),
        "train":  _split_info(X_train, y_train, 70),
        "val":    _split_info(X_val,   y_val,   15),
        "test":   _split_info(X_test,  y_test,  15),
        "num_features": num_feats,
        "cat_features": cat_feats,
    }


# ─────────────────────────────────────────────
# STEP 5 · Train Three Models
# ─────────────────────────────────────────────
def run_train():
    if "X_train" not in _state:
        raise RuntimeError("Run split first.")

    X_train = _state["X_train"]; y_train = _state["y_train"]
    X_val   = _state["X_val"];   y_val   = _state["y_val"]
    prep    = _state["preprocessor"]

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE),
    }
    COLORS = {
        "Logistic Regression": "#D4D4D4",
        "Random Forest":       "#FF4500",
        "Gradient Boosting":   "#D9381E",
    }
    DESCS = {
        "Logistic Regression": "Linear baseline with balanced class weights",
        "Random Forest":       "200-tree ensemble voting on random feature subsets",
        "Gradient Boosting":   "Sequential boosting: each tree corrects prior mistakes",
    }

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted  = {}

    for name, est in models.items():
        pipe    = Pipeline([("prep", prep), ("clf", est)])
        cv_scrs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        vp   = pipe.predict(X_val)
        vprb = pipe.predict_proba(X_val)[:, 1]
        pc, rc, _ = precision_recall_curve(y_val, vprb)

        results.append({
            "model":         name,
            "description":   DESCS[name],
            "color":         COLORS[name],
            "cv_f1":         round(float(cv_scrs.mean()), 4),
            "cv_f1_std":     round(float(cv_scrs.std()),  4),
            "val_precision": round(float(precision_score(y_val, vp, zero_division=0)), 4),
            "val_recall":    round(float(recall_score(y_val,    vp, zero_division=0)), 4),
            "val_f1":        round(float(f1_score(y_val,        vp, zero_division=0)), 4),
            "val_pr_auc":    round(float(auc(rc, pc)), 4),
        })

    _state["fitted_pipelines"] = fitted
    _state["cv"]               = cv
    _state["model_colors"]     = COLORS
    return {"results": results}


# ─────────────────────────────────────────────
# STEP 6 · Hyperparameter Tuning (GridSearchCV)
# ─────────────────────────────────────────────
def run_tune():
    if "fitted_pipelines" not in _state:
        raise RuntimeError("Run training first.")

    X_train = _state["X_train"]; y_train = _state["y_train"]
    prep = _state["preprocessor"]; cv = _state["cv"]

    gb_pipe = Pipeline([
        ("prep", prep),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "clf__n_estimators":  [150, 250],
        "clf__max_depth":     [3, 4],
        "clf__learning_rate": [0.05, 0.1],
    }
    gs = GridSearchCV(gb_pipe, param_grid, cv=cv, scoring="f1",
                      n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train)

    _state["best_model"] = gs.best_estimator_

    cvre = pd.DataFrame(gs.cv_results_)
    p_cols = [c for c in cvre.columns if c.startswith("param_clf__")]
    grid_rows = []
    for _, row in cvre.iterrows():
        entry = {
            "mean_test_score":  round(float(row["mean_test_score"]),  4),
            "std_test_score":   round(float(row["std_test_score"]),   4),
            "mean_train_score": round(float(row["mean_train_score"]), 4),
        }
        for pc in p_cols:
            entry[pc.replace("param_clf__", "")] = str(row[pc])
        grid_rows.append(entry)
    grid_rows.sort(key=lambda x: x["mean_test_score"], reverse=True)

    best_clean = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    return {
        "best_params":    best_clean,
        "best_score":     round(float(gs.best_score_), 4),
        "grid_results":   grid_rows,
        "n_combinations": len(grid_rows),
    }


# ─────────────────────────────────────────────
# STEP 7 · Final Test Evaluation
# ─────────────────────────────────────────────
def run_evaluate():
    if "best_model" not in _state:
        raise RuntimeError("Run tuning first.")

    X_test = _state["X_test"]; y_test = _state["y_test"]
    best   = _state["best_model"]
    fitted = _state["fitted_pipelines"]
    mcolors= _state.get("model_colors", {})

    test_pred = best.predict(X_test)
    test_prob = best.predict_proba(X_test)[:, 1]
    _state["test_pred"] = test_pred
    _state["test_prob"] = test_prob

    pr_curves = []
    fallback   = ["#D4D4D4", "#FF4500", "#D9381E"]
    for i, (name, pipe) in enumerate(fitted.items()):
        prb = pipe.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, prb)
        pr_curves.append({
            "name":      name,
            "precision": [round(float(v), 4) for v in p],
            "recall":    [round(float(v), 4) for v in r],
            "pr_auc":    round(float(auc(r, p)), 4),
            "color":     mcolors.get(name, fallback[i % 3]),
            "is_best":   False,
        })
    p_t, r_t, _ = precision_recall_curve(y_test, test_prob)
    pr_curves.append({
        "name":      "Tuned Gradient Boosting",
        "precision": [round(float(v), 4) for v in p_t],
        "recall":    [round(float(v), 4) for v in r_t],
        "pr_auc":    round(float(auc(r_t, p_t)), 4),
        "color":     "#FF4500",
        "is_best":   True,
    })

    cm     = confusion_matrix(y_test, test_pred).tolist()
    report = classification_report(y_test, test_pred,
                                   output_dict=True, digits=4, zero_division=0)

    def _rpt(d):
        return {k: round(float(v), 4) for k, v in d.items() if k != "support"}

    return {
        "pr_curves":        pr_curves,
        "confusion_matrix": cm,
        "conf_labels":      ["Healthy", "Failure"],
        "classification_report": {
            "healthy":      _rpt(report.get("0", {})),
            "failure":      _rpt(report.get("1", {})),
            "accuracy":     round(float(report.get("accuracy", 0)), 4),
            "macro_avg":    _rpt(report.get("macro avg",    {})),
            "weighted_avg": _rpt(report.get("weighted avg", {})),
        },
        "summary": {
            "total_test": _py(len(y_test)),
            "tp": _py(int(((y_test == 1) & (test_pred == 1)).sum())),
            "tn": _py(int(((y_test == 0) & (test_pred == 0)).sum())),
            "fp": _py(int(((y_test == 0) & (test_pred == 1)).sum())),
            "fn": _py(int(((y_test == 1) & (test_pred == 0)).sum())),
        },
    }


# ─────────────────────────────────────────────
# STEP 8 · Feature Importance
# ─────────────────────────────────────────────
def run_features():
    if "best_model" not in _state:
        raise RuntimeError("Run tuning first.")

    best      = _state["best_model"]
    num_feats = _state["num_feats"]
    cat_feats = _state["cat_feats"]
    ENG       = {"Temp_Diff_K", "Power_W", "Overstrain_Index"}

    ohe_names = best.named_steps["prep"].named_transformers_["cat"]\
        .get_feature_names_out(cat_feats).tolist()
    all_names = num_feats + ohe_names
    imps      = best.named_steps["clf"].feature_importances_

    fs = pd.Series(imps, index=all_names).sort_values(ascending=True)

    _state["feat_imp"] = fs
    return {
        "features":     fs.index.tolist(),
        "importances":  [round(float(v), 4) for v in fs.values],
        "is_engineered":[n in ENG for n in fs.index],
        "colors":       ["#FF4500" if n in ENG else "#525252" for n in fs.index],
    }


# ─────────────────────────────────────────────
# STEP 9 · Error Analysis
# ─────────────────────────────────────────────
def run_errors():
    if "test_pred" not in _state:
        raise RuntimeError("Run evaluation first.")

    df       = _state["df"]
    X_test   = _state["X_test"]
    y_test   = _state["y_test"]
    t_pred   = _state["test_pred"]
    t_prob   = _state["test_prob"]

    audit = df.loc[X_test.index].copy()
    audit["Predicted"]   = t_pred
    audit["Failure_Prob"] = np.round(t_prob, 3)

    fn = audit[(audit["Machine failure"] == 1) & (audit["Predicted"] == 0)]
    fp = audit[(audit["Machine failure"] == 0) & (audit["Predicted"] == 1)]

    FN_COLS = ["Type","Torque [Nm]","Rotational speed [rpm]",
               "Tool wear [min]","Failure_Prob","TWF","HDF","PWF","OSF","RNF"]
    FP_COLS = ["Type","Torque [Nm]","Rotational speed [rpm]",
               "Tool wear [min]","Failure_Prob"]

    def _rows(dframe, cols):
        avail = [c for c in cols if c in dframe.columns]
        out   = []
        for _, row in dframe[avail].head(10).iterrows():
            rec = {}
            for col in avail:
                v = row[col]
                if isinstance(v, (np.integer,)): rec[col] = int(v)
                elif isinstance(v, (np.floating,)): rec[col] = round(float(v), 3)
                else: rec[col] = str(v)
            out.append(rec)
        return out

    fn_modes = {}
    for m in ["TWF","HDF","PWF","OSF","RNF"]:
        if m in fn.columns:
            fn_modes[m] = int(fn[m].sum())

    tp = int(((y_test == 1) & (t_pred == 1)).sum())
    _state["error_summary"] = True

    return {
        "summary": {
            "total_test_failures": int(y_test.sum()),
            "true_positives":      tp,
            "false_negatives":     len(fn),
            "false_positives":     len(fp),
            "catch_rate":          round(tp / max(1, int(y_test.sum())) * 100, 1),
        },
        "false_negatives": _rows(fn, FN_COLS),
        "false_positives": _rows(fp, FP_COLS),
        "fn_failure_modes": fn_modes,
        "fn_probs": fn["Failure_Prob"].tolist() if len(fn) > 0 else [],
        "fp_probs": fp["Failure_Prob"].tolist() if len(fp) > 0 else [],
    }


# ─────────────────────────────────────────────
# FULL PIPELINE (Run All)
# ─────────────────────────────────────────────
def run_all():
    return {
        "load":       run_load(),
        "preprocess": run_preprocess(),
        "eda":        run_eda(),
        "split":      run_split(),
        "train":      run_train(),
        "tune":       run_tune(),
        "evaluate":   run_evaluate(),
        "features":   run_features(),
        "errors":     run_errors(),
    }
