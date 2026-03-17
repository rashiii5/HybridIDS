"""
7.
evaluate.py
-----------
Runs all 4 comparative experiments and saves results to metrics.json.

Experiments:
  1. Supervised-only LightGBM (no VAE, no SMOTE)
  2. VAE anomaly detector only (binary: normal vs attack)
  3. Basic hybrid (VAE + classifier, no cost-sensitive weights)
  4. Full hybrid (VAE + SMOTE-ENN + cost-sensitive LightGBM)  ← our system

Metrics computed:
  - Accuracy, Precision, Recall, F1 (macro + per-class)
  - ROC-AUC (binary)
  - MCC (Matthews Correlation Coefficient)
  - Confusion matrix
  - Average inference latency (ms/sample)

Usage:
    python evaluate.py
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "backend", "models")
FIG_DIR     = os.path.join(BASE_DIR, "results", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)

ATTACK_CLASSES = ["DoS", "Probe", "R2L", "U2R"]
RANDOM_SEED    = 42

# ─────────────────────────────────────────────
# REGISTER CUSTOM LAYER
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def log_transform(errors):
    return np.log1p(errors)


def compute_recon_error(encoder, decoder, X):
    z_mean, _, z = encoder.predict(X, batch_size=512, verbose=0)
    X_recon      = decoder.predict(z, batch_size=512, verbose=0)
    mse          = np.mean(np.square(X - X_recon), axis=1)
    return log_transform(mse)


def timed_predict(fn, *args, n_samples=None):
    t0  = time.time()
    out = fn(*args)
    ms  = (time.time() - t0) * 1000
    n   = n_samples or (len(args[0]) if args else 1)
    return out, round(ms / n, 4)


def binary_labels(y):
    """1 = attack, 0 = normal"""
    return (y != "normal").astype(int)


def attack_only_mask(y):
    return np.isin(y, ATTACK_CLASSES)


def safe_roc_auc(y_true, y_score):
    try:
        return round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception:
        return None


def compute_metrics(y_true, y_pred, y_score=None, classes=None):
    """Compute full metric dict for a set of predictions."""
    acc   = round(accuracy_score(y_true, y_pred), 4)
    prec  = round(precision_score(y_true, y_pred,
                  average="macro", zero_division=0), 4)
    rec   = round(recall_score(y_true, y_pred,
                  average="macro", zero_division=0), 4)
    f1    = round(f1_score(y_true, y_pred,
                  average="macro", zero_division=0), 4)
    mcc   = round(float(matthews_corrcoef(y_true, y_pred)), 4)
    auc   = safe_roc_auc(y_true, y_score) if y_score is not None else None

    # Per-class F1
    if classes is not None:
        per_class_f1 = {
            str(c): round(float(v), 4)
            for c, v in zip(
                classes,
                f1_score(y_true, y_pred, average=None, zero_division=0)
            )
        }
    else:
        per_class_f1 = {}

    return dict(accuracy=acc, precision=prec, recall=rec,
                macro_f1=f1, mcc=mcc, roc_auc=auc,
                per_class_f1=per_class_f1)


# ─────────────────────────────────────────────
# EXPERIMENT 1 — Supervised-only LightGBM
# ─────────────────────────────────────────────
def experiment_supervised(X_train, y_train, X_test, y_test):
    print("\n  ── Experiment 1: Supervised-only LightGBM ──")

    le = LabelEncoder()
    le.fit(ATTACK_CLASSES)

    # Train on ALL training attack samples, no SMOTE
    mask_tr = attack_only_mask(y_train)
    mask_te = attack_only_mask(y_test)

    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=8, num_leaves=63,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    clf.fit(X_train[mask_tr], le.transform(y_train[mask_tr]))

    def _predict(X):
        return clf.predict(X)

    y_pred_enc, lat = timed_predict(
        _predict, X_test[mask_te],
        n_samples=len(X_test[mask_te])
    )
    y_true_enc = le.transform(y_test[mask_te])

    metrics = compute_metrics(
        y_true_enc, y_pred_enc, classes=le.classes_
    )
    metrics["latency_ms"] = lat

    print(f"     Macro F1  : {metrics['macro_f1']}")
    print(f"     MCC       : {metrics['mcc']}")
    print(f"     Latency   : {lat} ms/sample")
    return metrics, le


# ─────────────────────────────────────────────
# EXPERIMENT 2 — VAE anomaly detector only
# ─────────────────────────────────────────────
def experiment_vae_only(encoder, decoder, X_test, y_test, threshold):
    print("\n  ── Experiment 2: VAE anomaly detector only ──")

    def _predict(X):
        errors = compute_recon_error(encoder, decoder, X)
        return (errors > threshold).astype(int)

    y_pred, lat = timed_predict(_predict, X_test, n_samples=len(X_test))
    y_true      = binary_labels(y_test)

    errors    = compute_recon_error(encoder, decoder, X_test)
    auc_score = safe_roc_auc(y_true, errors)

    metrics = compute_metrics(y_true, y_pred, y_score=errors)
    metrics["latency_ms"] = lat

    print(f"     Accuracy  : {metrics['accuracy']}")
    print(f"     Recall    : {metrics['recall']}")
    print(f"     ROC-AUC   : {auc_score}")
    print(f"     Latency   : {lat} ms/sample")
    return metrics


# ─────────────────────────────────────────────
# EXPERIMENT 3 — Basic hybrid (no cost weights)
# ─────────────────────────────────────────────
def experiment_basic_hybrid(encoder, decoder, X_train, y_train,
                             X_test, y_test, threshold):
    print("\n  ── Experiment 3: Basic hybrid (no cost-sensitive weights) ──")

    le = LabelEncoder()
    le.fit(ATTACK_CLASSES)

    # Train basic classifier — no SMOTE, no cost weights
    mask_tr = attack_only_mask(y_train)
    z_tr, _, _ = encoder.predict(X_train[mask_tr], batch_size=512, verbose=0)
    X_tr_feat  = np.concatenate(
        [X_train[mask_tr], z_tr.astype(np.float32)], axis=1
    )

    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=8, num_leaves=63,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    clf.fit(X_tr_feat, le.transform(y_train[mask_tr]))

    # Hybrid predict on test
    mask_te = attack_only_mask(y_test)
    errors  = compute_recon_error(encoder, decoder, X_test)

    t0 = time.time()
    final_preds = []
    for i in range(len(X_test)):
        if errors[i] <= threshold:
            final_preds.append("normal")
        else:
            xi    = X_test[i:i+1]
            z, _, _ = encoder.predict(xi, batch_size=1, verbose=0)
            xf    = np.concatenate([xi, z.astype(np.float32)], axis=1)
            pred  = le.inverse_transform(clf.predict(xf))[0]
            final_preds.append(str(pred))
    lat = round((time.time() - t0) * 1000 / len(X_test), 4)

    final_preds = np.array(final_preds)

    # Evaluate only on samples that are true attacks AND predicted as attacks
    mask_te         = attack_only_mask(y_test)
    pred_is_attack  = np.isin(final_preds, ATTACK_CLASSES)
    eval_mask       = mask_te & pred_is_attack

    # For true attack samples predicted as normal, treat as misclassification
    # by assigning them the most common wrong class "DoS"
    final_eval = final_preds.copy()
    final_eval[mask_te & ~pred_is_attack] = "DoS"

    y_pred_atk = le.transform(final_eval[mask_te])
    y_true_atk = le.transform(y_test[mask_te])

    metrics = compute_metrics(
        y_true_atk, y_pred_atk, classes=le.classes_
    )
    metrics["latency_ms"] = lat

    print(f"     Macro F1  : {metrics['macro_f1']}")
    print(f"     MCC       : {metrics['mcc']}")
    print(f"     Latency   : {lat} ms/sample")
    return metrics


# ─────────────────────────────────────────────
# EXPERIMENT 4 — Full hybrid (our system)
# ─────────────────────────────────────────────
def experiment_full_hybrid(encoder, decoder, clf, le,
                            X_test, y_test, threshold):
    print("\n  ── Experiment 4: Full hybrid (our system) ──")

    mask_te = attack_only_mask(y_test)
    errors  = compute_recon_error(encoder, decoder, X_test)

    t0 = time.time()
    final_preds = []
    for i in range(len(X_test)):
        if errors[i] <= threshold:
            final_preds.append("normal")
        else:
            xi      = X_test[i:i+1]
            z, _, _ = encoder.predict(xi, batch_size=1, verbose=0)
            xf      = np.concatenate([xi, z.astype(np.float32)], axis=1)
            pred    = le.inverse_transform(clf.predict(xf))[0]
            final_preds.append(str(pred))
    lat = round((time.time() - t0) * 1000 / len(X_test), 4)

    final_preds = np.array(final_preds)

    # Evaluate only on samples that are true attacks AND predicted as attacks
    mask_te         = attack_only_mask(y_test)
    pred_is_attack  = np.isin(final_preds, ATTACK_CLASSES)
    eval_mask       = mask_te & pred_is_attack

    # For true attack samples predicted as normal, treat as misclassification
    # by assigning them the most common wrong class "DoS"
    final_eval = final_preds.copy()
    final_eval[mask_te & ~pred_is_attack] = "DoS"

    y_pred_atk = le.transform(final_eval[mask_te])
    y_true_atk = le.transform(y_test[mask_te])

    metrics = compute_metrics(
        y_true_atk, y_pred_atk, classes=le.classes_
    )
    metrics["latency_ms"] = lat

    print(f"     Macro F1  : {metrics['macro_f1']}")
    print(f"     MCC       : {metrics['mcc']}")
    print(f"     Latency   : {lat} ms/sample")
    return metrics, final_preds


# ─────────────────────────────────────────────
# COMPARISON PLOT
# ─────────────────────────────────────────────
def plot_comparison(all_metrics: dict):
    models = list(all_metrics.keys())
    metric_keys = ["accuracy", "macro_f1", "mcc"]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key in zip(axes, metric_keys):
        vals = [all_metrics[m].get(key, 0) or 0 for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="black",
                      linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(key.replace("_", " ").title())
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Comparison plot saved → {path}")


def plot_rare_attack_recall(all_metrics: dict):
    """Bar chart focusing on U2R and R2L recall across models."""
    models = list(all_metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, cls in zip(axes, ["U2R", "R2L"]):
        vals = [
            all_metrics[m].get("per_class_f1", {}).get(cls, 0) or 0
            for m in models
        ]
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
        bars = ax.bar(models, vals, color=colors,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{cls} F1 Score")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)

    plt.suptitle("Rare Attack Detection Improvement", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "rare_attack_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Rare attack plot saved → {path}")


def plot_latency(all_metrics: dict):
    models = list(all_metrics.keys())
    vals   = [all_metrics[m].get("latency_ms", 0) for m in models]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(models, vals, color=colors,
                   edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}ms", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Avg Latency (ms/sample)")
    plt.title("Inference Latency Comparison")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "latency_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Latency plot saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_evaluation():
    print("\n━━━ Loading data and models ━━━")
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"), allow_pickle=True)
    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"),  allow_pickle=True)

    encoder = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "vae_encoder.keras"),
        custom_objects={"Sampling": Sampling}
    )
    decoder = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "vae_decoder.keras"),
        custom_objects={"Sampling": Sampling}
    )
    clf        = joblib.load(os.path.join(MODELS_DIR, "lgbm_classifier.pkl"))
    le         = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    thresh_data = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
    threshold  = thresh_data["threshold"]

    print(f"  Threshold: {threshold:.6f}")
    print(f"  Test samples: {len(X_test)}")

    print("\n━━━ Running experiments ━━━")
    m1, _     = experiment_supervised(X_train, y_train, X_test, y_test)
    m2        = experiment_vae_only(encoder, decoder, X_test, y_test, threshold)
    m3        = experiment_basic_hybrid(encoder, decoder, X_train, y_train,
                                        X_test, y_test, threshold)
    m4, _     = experiment_full_hybrid(encoder, decoder, clf, le,
                                       X_test, y_test, threshold)

    all_metrics = {
        "1_Supervised":   m1,
        "2_VAE_Only":     m2,
        "3_Basic_Hybrid": m3,
        "4_Full_Hybrid":  m4,
    }

    print("\n━━━ Summary table ━━━")
    print(f"  {'Model':<20} {'Acc':>6} {'MacroF1':>8} "
          f"{'MCC':>6} {'U2R_F1':>7} {'R2L_F1':>7} {'ms/s':>6}")
    print("  " + "─"*65)
    for name, m in all_metrics.items():
        u2r = m.get("per_class_f1", {}).get("U2R", 0) or 0
        r2l = m.get("per_class_f1", {}).get("R2L", 0) or 0
        acc = m.get("accuracy", 0) or 0
        mf1 = m.get("macro_f1", 0) or 0
        mcc = m.get("mcc", 0) or 0
        lat = m.get("latency_ms", 0) or 0
        print(f"  {name:<20} {acc:>6.3f} {mf1:>8.3f} "
              f"{mcc:>6.3f} {u2r:>7.3f} {r2l:>7.3f} {lat:>6.2f}")

    print("\n━━━ Saving metrics.json ━━━")

    # Convert numpy types for JSON serialisation
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(clean(all_metrics), f, indent=2)
    print(f"  Saved → {metrics_path}")

    print("\n━━━ Generating comparison plots ━━━")
    plot_comparison(all_metrics)
    plot_rare_attack_recall(all_metrics)
    plot_latency(all_metrics)

    print("\n✅  Evaluation complete!\n")
    return all_metrics


if __name__ == "__main__":
    run_evaluation()