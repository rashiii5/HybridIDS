"""
5.
pipeline.py
-----------
End-to-end hybrid detection pipeline.

Flow:
    Raw input → Preprocess → VAE reconstruction error
        → If normal  : return "normal"
        → If anomaly : LightGBM → return attack category

All models are loaded once at startup and reused across requests.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ─────────────────────────────────────────────
# REGISTER CUSTOM LAYER  (needed for loading)
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
# MODEL REGISTRY  (singleton — loaded once)
# ─────────────────────────────────────────────
class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print("  Loading models into registry...")

        self.encoder     = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "vae_encoder.keras"),
            custom_objects={"Sampling": Sampling}
        )
        self.decoder     = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "vae_decoder.keras"),
            custom_objects={"Sampling": Sampling}
        )
        self.classifier  = joblib.load(
            os.path.join(MODELS_DIR, "lgbm_classifier.pkl")
        )
        self.scaler      = joblib.load(
            os.path.join(MODELS_DIR, "scaler.pkl")
        )
        self.label_enc   = joblib.load(
            os.path.join(MODELS_DIR, "label_encoder.pkl")
        )
        self.feat_cols   = joblib.load(
            os.path.join(MODELS_DIR, "feature_columns.pkl")
        )
        threshold_data   = joblib.load(
            os.path.join(MODELS_DIR, "threshold.pkl")
        )
        self.threshold   = threshold_data["threshold"]
        self.threshold_data = threshold_data

        self._loaded = True
        print(f"  ✓ Models loaded. Threshold = {self.threshold:.6f}")


# Global registry instance
registry = ModelRegistry()


# ─────────────────────────────────────────────
# PREPROCESSING  (for inference)
# ─────────────────────────────────────────────
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

COLUMNS_NO_LABEL = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]


def preprocess_dataframe(df: pd.DataFrame) -> np.ndarray:
    """
    Takes a raw dataframe (with or without label/difficulty columns),
    encodes and scales it using the saved scaler and feature columns.
    Returns scaled numpy array ready for inference.
    """
    # Drop label/difficulty columns if present
    df = df.copy()
    for col in ["label", "difficulty", "category"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure correct column order if raw feature names present
    if "duration" in df.columns:
        available = [c for c in COLUMNS_NO_LABEL if c in df.columns]
        df = df[available]

    # One-hot encode
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Align to training feature columns
    df_enc = df_enc.reindex(columns=registry.feat_cols, fill_value=0)

    # Scale
    X = registry.scaler.transform(df_enc).astype(np.float32)
    return X


# ─────────────────────────────────────────────
# CORE DETECTION FUNCTIONS
# ─────────────────────────────────────────────
def compute_reconstruction_error(X: np.ndarray) -> np.ndarray:
    """Per-sample log(1 + MSE) reconstruction error."""
    X_recon = registry.decoder.predict(
        registry.encoder.predict(X, batch_size=512, verbose=0)[2],
        batch_size=512, verbose=0
    )
    mse    = np.mean(np.square(X - X_recon), axis=1)
    return np.log1p(mse)


def get_latent(X: np.ndarray) -> np.ndarray:
    """Returns z_mean latent vector."""
    z_mean, _, _ = registry.encoder.predict(X, batch_size=512, verbose=0)
    return z_mean.astype(np.float32)


def classify_attacks(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns predicted class labels and probabilities."""
    latent    = get_latent(X)
    X_combined = np.concatenate([X, latent], axis=1)
    probs     = registry.classifier.predict_proba(X_combined)
    preds_idx = np.argmax(probs, axis=1)
    labels    = registry.label_enc.inverse_transform(preds_idx)
    return labels, probs


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def predict(X: np.ndarray) -> list[dict]:
    """
    Full hybrid pipeline for a batch of preprocessed samples.

    Returns a list of dicts, one per sample:
        {
            "sample_id":        int,
            "reconstruction_error": float,
            "is_anomaly":       bool,
            "label":            str,   # "normal" or attack category
            "confidence":       float, # 0-1
            "attack_probs":     dict,  # per-class probabilities (attacks only)
            "latency_ms":       float
        }
    """
    registry.load()
    results = []
    t_start = time.time()

    # Step 1: reconstruction errors for all samples
    errors = compute_reconstruction_error(X)
    is_anomaly = errors > registry.threshold

    # Step 2: classify anomalous samples
    anomaly_idx = np.where(is_anomaly)[0]
    attack_labels = {}
    attack_probs  = {}

    if len(anomaly_idx) > 0:
        X_anomaly = X[anomaly_idx]
        labels, probs = classify_attacks(X_anomaly)
        for i, idx in enumerate(anomaly_idx):
            attack_labels[idx] = labels[i]
            attack_probs[idx]  = {
                cls: float(probs[i][j])
                for j, cls in enumerate(registry.label_enc.classes_)
            }

    t_end = time.time()
    total_ms = (t_end - t_start) * 1000

    # Step 3: assemble results
    for i in range(len(X)):
        if is_anomaly[i]:
            label      = str(attack_labels[i])
            probs_dict = attack_probs[i]
            confidence = float(max(probs_dict.values()))
        else:
            label      = "normal"
            probs_dict = {}
            confidence = float(1.0 - (errors[i] / registry.threshold))
            confidence = max(0.0, min(1.0, confidence))

        results.append({
            "sample_id":            i,
            "reconstruction_error": float(errors[i]),
            "is_anomaly":           bool(is_anomaly[i]),
            "label":                label,
            "confidence":           round(confidence, 4),
            "attack_probs":         probs_dict,
            "latency_ms":           round(total_ms / len(X), 4)
        })

    return results


# ─────────────────────────────────────────────
# PREDICT FROM DATAFRAME  (used by API)
# ─────────────────────────────────────────────
def predict_dataframe(df: pd.DataFrame) -> list[dict]:
    """Preprocess a raw dataframe then run predict()."""
    registry.load()
    X = preprocess_dataframe(df)
    return predict(X)


def predict_single(sample: dict) -> dict:
    """Predict a single sample from a dict of raw feature values."""
    registry.load()
    df = pd.DataFrame([sample])
    results = predict_dataframe(df)
    return results[0]


# ─────────────────────────────────────────────
# SUMMARY STATS  (used by dashboard)
# ─────────────────────────────────────────────
def summarise(results: list[dict]) -> dict:
    """Aggregate prediction results into summary statistics."""
    total   = len(results)
    labels  = [r["label"] for r in results]
    counts  = {l: labels.count(l) for l in set(labels)}
    n_anom  = sum(1 for r in results if r["is_anomaly"])
    avg_lat = np.mean([r["latency_ms"] for r in results])

    return {
        "total_samples":    total,
        "normal_count":     counts.get("normal", 0),
        "anomaly_count":    n_anom,
        "attack_breakdown": {
            k: v for k, v in counts.items() if k != "normal"
        },
        "detection_rate":   round(n_anom / total, 4) if total > 0 else 0,
        "avg_latency_ms":   round(float(avg_lat), 4)
    }


# ─────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n━━━ Pipeline smoke test ━━━")
    registry.load()

    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"), allow_pickle=True)

    # Test on 200 samples
    sample  = X_test[:200]
    results = predict(sample)
    summary = summarise(results)

    print(f"\n  Results for first 200 test samples:")
    print(f"  Total       : {summary['total_samples']}")
    print(f"  Normal      : {summary['normal_count']}")
    print(f"  Anomalies   : {summary['anomaly_count']}")
    print(f"  Breakdown   : {summary['attack_breakdown']}")
    print(f"  Avg latency : {summary['avg_latency_ms']} ms/sample")

    print("\n  Sample predictions (first 5):")
    for r in results[:5]:
        true = y_test[r["sample_id"]]
        print(f"  [{r['sample_id']}] true={true:<10} "
              f"pred={r['label']:<10} "
              f"error={r['reconstruction_error']:.4f} "
              f"conf={r['confidence']:.3f}")

    print("\n✅  Pipeline smoke test complete!\n")