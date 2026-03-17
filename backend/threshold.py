"""
3.
threshold.py
------------
Ensemble anomaly threshold selection using three methods:
  1. 95th percentile of normal training errors
  2. ROC-optimal threshold (maximises TPR - FPR)
  3. F1-optimal threshold (maximises binary F1)

Final threshold = weighted average of all three.
All computations done on log-transformed reconstruction errors
to handle the extreme scale variance in NSL-KDD.

Usage:
    python threshold.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc,
                             f1_score, precision_score,
                             recall_score, confusion_matrix)
import joblib

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
FIG_DIR    = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOG TRANSFORM  (stabilises extreme errors)
# ─────────────────────────────────────────────
def log_transform(errors: np.ndarray) -> np.ndarray:
    return np.log1p(errors)


# ─────────────────────────────────────────────
# METHOD 1 — 95th Percentile
# ─────────────────────────────────────────────
def percentile_threshold(log_errors_normal: np.ndarray,
                         pct: float = 95) -> float:
    t = float(np.percentile(log_errors_normal, pct))
    print(f"  [Method 1] {pct}th percentile threshold : {t:.6f}")
    return t


# ─────────────────────────────────────────────
# METHOD 2 — ROC-Optimal  (max TPR - FPR)
# ─────────────────────────────────────────────
def roc_threshold(log_errors_all: np.ndarray,
                  binary_labels: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(binary_labels, log_errors_all)
    roc_auc = auc(fpr, tpr)
    youden   = tpr - fpr
    best_idx = int(np.argmax(youden))
    t        = float(thresholds[best_idx])
    print(f"  [Method 2] ROC-optimal threshold       : {t:.6f}  "
          f"(AUC = {roc_auc:.4f}, TPR={tpr[best_idx]:.3f}, "
          f"FPR={fpr[best_idx]:.3f})")
    return t, fpr, tpr, roc_auc, thresholds


# ─────────────────────────────────────────────
# METHOD 3 — F1-Optimal  (sweep thresholds)
# ─────────────────────────────────────────────
def f1_threshold(log_errors_all: np.ndarray,
                 binary_labels: np.ndarray,
                 n_steps: int = 300) -> float:
    candidates = np.linspace(log_errors_all.min(),
                             log_errors_all.max(), n_steps)
    best_f1, best_t = 0.0, 0.0
    for t in candidates:
        preds = (log_errors_all >= t).astype(int)
        f1    = f1_score(binary_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"  [Method 3] F1-optimal threshold        : {best_t:.6f}  "
          f"(F1 = {best_f1:.4f})")
    return best_t


# ─────────────────────────────────────────────
# ENSEMBLE  (weighted average)
# ─────────────────────────────────────────────
def ensemble_threshold(t_pct: float,
                       t_roc: float,
                       t_f1:  float,
                       weights=(0.25, 0.40, 0.35)) -> float:
    t = weights[0]*t_pct + weights[1]*t_roc + weights[2]*t_f1
    print(f"\n  [Ensemble] Weighted threshold          : {t:.6f}")
    print(f"             Weights → pct:{weights[0]}  "
          f"roc:{weights[1]}  f1:{weights[2]}")
    return t


# ─────────────────────────────────────────────
# EVALUATE at chosen threshold
# ─────────────────────────────────────────────
def evaluate_threshold(log_errors: np.ndarray,
                       binary_labels: np.ndarray,
                       threshold: float,
                       label: str = "Ensemble"):
    preds = (log_errors >= threshold).astype(int)
    acc   = (preds == binary_labels).mean()
    prec  = precision_score(binary_labels, preds, zero_division=0)
    rec   = recall_score(binary_labels, preds, zero_division=0)
    f1    = f1_score(binary_labels, preds, zero_division=0)
    cm    = confusion_matrix(binary_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ── {label} threshold evaluation ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1-score  : {f1:.4f}")
    print(f"     TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(accuracy=acc, precision=prec,
                recall=rec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_roc(fpr, tpr, roc_auc):
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"VAE ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — VAE Anomaly Detector")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "roc_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"  ROC curve saved → {path}")


def plot_error_distribution_log(log_errors_normal, log_errors_attack,
                                threshold):
    plt.figure(figsize=(10, 5))
    plt.hist(log_errors_normal, bins=120, alpha=0.6,
             color="steelblue", label="Normal",  density=True)
    plt.hist(log_errors_attack, bins=120, alpha=0.6,
             color="crimson",   label="Attack",  density=True)
    plt.axvline(threshold, color="gold", linewidth=2,
                linestyle="--", label=f"Threshold = {threshold:.3f}")
    plt.xlabel("log(1 + Reconstruction Error)")
    plt.ylabel("Density")
    plt.title("Log-Scaled VAE Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "vae_error_log_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Log error distribution saved → {path}")


def plot_f1_sweep(log_errors_all, binary_labels, final_threshold,
                  n_steps=300):
    candidates = np.linspace(log_errors_all.min(),
                             log_errors_all.max(), n_steps)
    f1s = []
    for t in candidates:
        preds = (log_errors_all >= t).astype(int)
        f1s.append(f1_score(binary_labels, preds, zero_division=0))

    plt.figure(figsize=(9, 4))
    plt.plot(candidates, f1s, color="mediumseagreen", lw=2)
    plt.axvline(final_threshold, color="gold", linestyle="--",
                linewidth=2, label=f"Ensemble threshold = {final_threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold Sweep")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "f1_threshold_sweep.png")
    plt.savefig(path)
    plt.close()
    print(f"  F1 sweep plot saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def compute_threshold():
    print("\n━━━ Loading error arrays ━━━")
    errors_normal = np.load(os.path.join(PROC_DIR, "train_normal_errors.npy"))
    errors_attack = np.load(os.path.join(PROC_DIR, "train_attack_errors.npy"))
    errors_all    = np.load(os.path.join(PROC_DIR, "train_all_errors.npy"))
    binary_labels = np.load(os.path.join(PROC_DIR, "train_binary_labels.npy"))

    print(f"  Normal samples : {len(errors_normal)}")
    print(f"  Attack samples : {len(errors_attack)}")

    print("\n━━━ Log-transforming errors ━━━")
    log_normal = log_transform(errors_normal)
    log_attack = log_transform(errors_attack)
    log_all    = log_transform(errors_all)

    print(f"  Normal log-error — mean: {log_normal.mean():.4f}  "
          f"std: {log_normal.std():.4f}")
    print(f"  Attack log-error — mean: {log_attack.mean():.4f}  "
          f"std: {log_attack.std():.4f}")

    print("\n━━━ Computing thresholds ━━━")
    t_pct                          = percentile_threshold(log_normal)
    t_roc, fpr, tpr, roc_auc, _   = roc_threshold(log_all, binary_labels)
    t_f1                           = f1_threshold(log_all, binary_labels)
    t_ensemble                     = ensemble_threshold(t_pct, t_roc, t_f1)

    print("\n━━━ Evaluating threshold ━━━")
    metrics = evaluate_threshold(log_all, binary_labels, t_ensemble)

    print("\n━━━ Saving threshold ━━━")
    threshold_data = {
        "threshold":      t_ensemble,
        "t_percentile":   t_pct,
        "t_roc":          t_roc,
        "t_f1":           t_f1,
        "roc_auc":        roc_auc,
        "train_metrics":  metrics
    }
    path = os.path.join(MODELS_DIR, "threshold.pkl")
    joblib.dump(threshold_data, path)
    print(f"  Threshold data saved → {path}")

    print("\n━━━ Generating plots ━━━")
    plot_roc(fpr, tpr, roc_auc)
    plot_error_distribution_log(log_normal, log_attack, t_ensemble)
    plot_f1_sweep(log_all, binary_labels, t_ensemble)

    print("\n✅  Threshold computation complete!\n")
    return threshold_data


if __name__ == "__main__":
    compute_threshold()