"""
4.
classifier.py
-------------
Imbalance-aware multi-class attack classifier using LightGBM.
- Input features: original scaled features + VAE latent vector (concatenated)
- Imbalance handling: SMOTE-ENN
- Trained only on ATTACK samples (normal class excluded — gated by VAE)

Usage:
    python classifier.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, matthews_corrcoef
)
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
import tensorflow as tf

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
FIG_DIR    = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ATTACK_CLASSES = ["DoS", "Probe", "R2L", "U2R"]


# ─────────────────────────────────────────────
# EXTRACT LATENT FEATURES FROM VAE ENCODER
# ─────────────────────────────────────────────
def get_latent_features(X: np.ndarray, encoder_path: str) -> np.ndarray:
    """Returns z_mean (8-dim) for each sample — used as extra features."""
    @tf.keras.utils.register_keras_serializable()
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim   = tf.shape(z_mean)[1]
            eps   = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * eps
        
    encoder = tf.keras.models.load_model(
        encoder_path,
        custom_objects={"Sampling": Sampling}
    )
    z_mean, _, _ = encoder.predict(X, batch_size=512, verbose=0)
    return z_mean.astype(np.float32)


# ─────────────────────────────────────────────
# BUILD COMBINED FEATURE MATRIX
# ─────────────────────────────────────────────
def build_features(X: np.ndarray, encoder_path: str) -> np.ndarray:
    """Concatenate original features + latent vector."""
    latent = get_latent_features(X, encoder_path)
    combined = np.concatenate([X, latent], axis=1)
    print(f"  Feature dim: {X.shape[1]} original + "
          f"{latent.shape[1]} latent = {combined.shape[1]} total")
    return combined


# ─────────────────────────────────────────────
# APPLY SMOTE-ENN
# ─────────────────────────────────────────────
def apply_smoteenn(X: np.ndarray, y: np.ndarray):
    print("\n  Class distribution BEFORE SMOTE-ENN:")
    for cls in np.unique(y):
        print(f"    {cls:<10} {(y == cls).sum():>6}")

    smoteenn = SMOTEENN(random_state=RANDOM_SEED)
    X_res, y_res = smoteenn.fit_resample(X, y)

    print("\n  Class distribution AFTER SMOTE-ENN:")
    for cls in np.unique(y_res):
        print(f"    {cls:<10} {(y_res == cls).sum():>6}")

    return X_res, y_res


# ─────────────────────────────────────────────
# TRAIN LIGHTGBM
# ─────────────────────────────────────────────
def train_lgbm(X_train: np.ndarray, y_train: np.ndarray,
               label_encoder: LabelEncoder) -> LGBMClassifier:
    y_encoded = label_encoder.transform(y_train)

    # Cost-sensitive weights — penalise rare class misclassification heavily
    classes      = label_encoder.classes_   # ['DoS', 'Probe', 'R2L', 'U2R']
    class_weight = {0: 1.0,   # DoS   — common, no boost needed
                    1: 1.5,   # Probe — slight boost
                    2: 4.0,   # R2L   — rare, boost recall
                    3: 12.0}  # U2R   — critical + very rare, heavy boost

    sample_weights = np.array([class_weight[c] for c in y_encoded])

    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,       # slower learning = better generalisation
        max_depth=10,
        num_leaves=80,
        min_child_samples=5,      # lower = model can learn from tiny classes
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=False,       # we handle balance via sample weights
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )

    clf.fit(X_train, y_encoded, sample_weight=sample_weights)
    print(f"  LightGBM trained on {len(y_train)} samples (cost-sensitive)")
    return clf


# ─────────────────────────────────────────────
# EVALUATE CLASSIFIER
# ─────────────────────────────────────────────
def evaluate_classifier(clf, X_test, y_test, label_encoder):
    y_encoded = label_encoder.transform(y_test)
    y_pred    = clf.predict(X_test)

    print("\n  ── Classification Report ──")
    print(classification_report(
        y_encoded, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))

    macro_f1 = f1_score(y_encoded, y_pred, average="macro", zero_division=0)
    mcc      = matthews_corrcoef(y_encoded, y_pred)
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")

    return y_encoded, y_pred, macro_f1, mcc


# ─────────────────────────────────────────────
# PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


# ─────────────────────────────────────────────
# PLOT CLASS-WISE F1
# ─────────────────────────────────────────────
def plot_classwise_f1(y_true, y_pred, class_names):
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    colors = ["steelblue", "mediumseagreen", "orange", "crimson"]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_names, f1s, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 1.1)
    plt.ylabel("F1 Score")
    plt.title("Class-wise F1 Score (Attack Classifier)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "classwise_f1.png")
    plt.savefig(path)
    plt.close()
    print(f"  Class-wise F1 plot saved → {path}")


# ─────────────────────────────────────────────
# PLOT FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(clf, feature_dim_original, top_n=20):
    importances = clf.feature_importances_
    n_latent    = len(importances) - feature_dim_original
    names = ([f"feat_{i}" for i in range(feature_dim_original)] +
             [f"latent_{i}" for i in range(n_latent)])

    idx     = np.argsort(importances)[-top_n:][::-1]
    top_imp = importances[idx]
    top_nm  = [names[i] for i in idx]

    plt.figure(figsize=(10, 6))
    colors = ["darkorange" if "latent" in n else "steelblue" for n in top_nm]
    plt.barh(top_nm[::-1], top_imp[::-1], color=colors[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances (orange = latent)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"  Feature importance plot saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def train_classifier():
    print("\n━━━ Loading processed data ━━━")
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"), allow_pickle=True)
    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"),  allow_pickle=True)

    encoder_path = os.path.join(MODELS_DIR, "vae_encoder.keras")

    # ── Filter to attack samples only ──
    train_mask = np.isin(y_train, ATTACK_CLASSES)
    test_mask  = np.isin(y_test,  ATTACK_CLASSES)

    X_train_atk = X_train[train_mask]
    y_train_atk = y_train[train_mask]
    X_test_atk  = X_test[test_mask]
    y_test_atk  = y_test[test_mask]

    print(f"  Train attack samples : {len(X_train_atk)}")
    print(f"  Test  attack samples : {len(X_test_atk)}")

    print("\n━━━ Building combined features (original + latent) ━━━")
    X_train_feat = build_features(X_train_atk, encoder_path)
    X_test_feat  = build_features(X_test_atk,  encoder_path)
    original_dim = X_train_atk.shape[1]

    print("\n━━━ Applying SMOTE-ENN ━━━")
    X_train_bal, y_train_bal = apply_smoteenn(X_train_feat, y_train_atk)

    print("\n━━━ Encoding labels ━━━")
    le = LabelEncoder()
    le.fit(ATTACK_CLASSES)
    print(f"  Classes: {list(le.classes_)}")

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"  Label encoder saved → {MODELS_DIR}/label_encoder.pkl")

    print("\n━━━ Training LightGBM ━━━")
    clf = train_lgbm(X_train_bal, y_train_bal, le)

    print("\n━━━ Saving classifier ━━━")
    clf_path = os.path.join(MODELS_DIR, "lgbm_classifier.pkl")
    joblib.dump(clf, clf_path)
    print(f"  Classifier saved → {clf_path}")

    print("\n━━━ Evaluating on test set ━━━")
    y_true_enc, y_pred_enc, macro_f1, mcc = evaluate_classifier(
        clf, X_test_feat, y_test_atk, le
    )

    print("\n━━━ Generating plots ━━━")
    plot_confusion_matrix(y_true_enc, y_pred_enc, le.classes_)
    plot_classwise_f1(y_true_enc, y_pred_enc, le.classes_)
    plot_feature_importance(clf, original_dim)

    print("\n✅  Classifier training complete!\n")
    return clf, le


if __name__ == "__main__":
    train_classifier()