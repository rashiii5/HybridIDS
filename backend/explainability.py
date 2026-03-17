"""
6.
explainability.py
-----------------
SHAP-based explainability for the LightGBM attack classifier.
For each flagged sample, returns:
  - SHAP values per feature (which features drove the prediction)
  - Top contributing features (positive and negative)
  - Plain-English summary sentence

Usage:
    python explainability.py   (runs a demo on test samples)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR    = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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
# FEATURE NAMES
# ─────────────────────────────────────────────
def get_feature_names() -> list:
    """Returns full feature name list: original encoded + latent."""
    feat_cols    = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    latent_names = [f"latent_{i}" for i in range(8)]
    return list(feat_cols) + latent_names


# ─────────────────────────────────────────────
# HUMAN-READABLE FEATURE DESCRIPTIONS
# ─────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "duration":                   "connection duration (seconds)",
    "src_bytes":                  "bytes sent from source",
    "dst_bytes":                  "bytes sent to destination",
    "wrong_fragment":             "number of wrong fragments",
    "urgent":                     "number of urgent packets",
    "hot":                        "number of hot indicators",
    "num_failed_logins":          "failed login attempts",
    "logged_in":                  "login status",
    "num_compromised":            "compromised conditions",
    "root_shell":                 "root shell obtained",
    "su_attempted":               "su root attempted",
    "num_root":                   "root accesses",
    "num_file_creations":         "file creation operations",
    "num_shells":                 "shell prompts",
    "num_access_files":           "operations on access control files",
    "count":                      "connections to same host (last 2s)",
    "srv_count":                  "connections to same service (last 2s)",
    "serror_rate":                "SYN error rate",
    "rerror_rate":                "REJ error rate",
    "same_srv_rate":              "same service connection rate",
    "diff_srv_rate":              "different service connection rate",
    "dst_host_count":             "destination host connection count",
    "dst_host_srv_count":         "destination host service count",
    "dst_host_serror_rate":       "destination host SYN error rate",
    "dst_host_rerror_rate":       "destination host REJ error rate",
}

ATTACK_EXPLANATIONS = {
    "DoS":   "denial-of-service attack — overwhelming the target with traffic",
    "Probe": "network probe/scan — attacker surveying the network",
    "R2L":   "remote-to-local attack — unauthorized access from remote machine",
    "U2R":   "user-to-root attack — privilege escalation to root access",
}


# ─────────────────────────────────────────────
# PLAIN-ENGLISH SUMMARY
# ─────────────────────────────────────────────
def _generate_summary(label: str, top3: list, confidence: float) -> str:
    attack_desc = ATTACK_EXPLANATIONS.get(str(label), str(label))
    feat_parts  = []
    for f in top3:
        desc      = f["description"]
        direction = "elevated" if f["shap_value"] > 0 else "low"
        feat_parts.append(f"{direction} {desc}")

    feats_str = ", ".join(feat_parts) if feat_parts else "anomalous network behaviour"
    conf_pct  = int(confidence * 100)

    return (f"This sample was classified as a {label} ({attack_desc}) "
            f"with {conf_pct}% confidence. "
            f"Key indicators: {feats_str}.")


# ─────────────────────────────────────────────
# EXPLAIN A SINGLE SAMPLE
# ─────────────────────────────────────────────
def explain_sample(X_combined: np.ndarray,
                   classifier,
                   label_encoder,
                   feature_names: list,
                   top_n: int = 10) -> dict:
    # Ensure plain Python list of strings
    feature_names = [str(n) for n in list(feature_names)]

    explainer  = shap.TreeExplainer(classifier)
    shap_vals  = explainer.shap_values(X_combined)

    pred_idx   = int(classifier.predict(X_combined)[0])
    pred_label = str(label_encoder.inverse_transform([pred_idx])[0])
    pred_probs = classifier.predict_proba(X_combined)[0]
    confidence = float(pred_probs[pred_idx])

    # Robustly extract SHAP values as a flat 1D numpy array
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[pred_idx]).flatten()
    else:
        sv = np.array(shap_vals).flatten()

    # If sv has more elements than features, it may be 2D collapsed — trim
    sv = sv[:len(feature_names)]

    abs_sv  = np.abs(sv)
    top_idx = np.argsort(abs_sv)[-top_n:][::-1].tolist()

    top_features = []
    for idx in top_idx:
        idx  = int(idx)
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        desc = FEATURE_DESCRIPTIONS.get(name, name)
        top_features.append({
            "feature":     name,
            "description": desc,
            "shap_value":  round(float(sv[idx]), 6),
            "direction":   "increases" if float(sv[idx]) > 0 else "decreases",
            "importance":  round(float(abs_sv[idx]), 6)
        })

    summary = _generate_summary(pred_label, top_features[:3], confidence)

    return {
        "predicted_label": pred_label,
        "confidence":      round(confidence, 4),
        "attack_type":     ATTACK_EXPLANATIONS.get(pred_label, ""),
        "top_features":    top_features,
        "shap_values":     [round(float(v), 6) for v in sv],
        "feature_names":   feature_names,
        "summary":         summary
    }


# ─────────────────────────────────────────────
# EXPLAIN BATCH  (for API)
# ─────────────────────────────────────────────
def explain_batch(X_combined: np.ndarray,
                  classifier,
                  label_encoder,
                  feature_names: list,
                  sample_ids: list = None) -> list:
    """Explain multiple samples. Returns list of explanation dicts."""
    feature_names = [str(n) for n in list(feature_names)]
    results = []
    for i in range(len(X_combined)):
        exp = explain_sample(
            X_combined[i:i+1], classifier, label_encoder, feature_names
        )
        exp["sample_id"] = sample_ids[i] if sample_ids else i
        results.append(exp)
    return results


# ─────────────────────────────────────────────
# PLOT SHAP BAR CHART FOR ONE SAMPLE
# ─────────────────────────────────────────────
def plot_shap_bar(explanation: dict, save_path: str = None) -> str:
    top    = explanation["top_features"][:12]
    names  = [f["feature"] for f in top]
    values = [f["shap_value"] for f in top]
    colors = ["crimson" if v > 0 else "steelblue" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names[::-1], values[::-1], color=colors[::-1],
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(
        f"Feature Contributions → {explanation['predicted_label']} "
        f"({int(explanation['confidence']*100)}% confidence)"
    )
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIG_DIR, "shap_example.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# RECONSTRUCTION ERROR PER FEATURE GROUP
# ─────────────────────────────────────────────
FEATURE_GROUPS = {
    "Basic Connection": list(range(0, 9)),
    "Content":          list(range(9, 22)),
    "Traffic (Time)":   list(range(22, 31)),
    "Traffic (Host)":   list(range(31, 41)),
}

def reconstruction_error_by_group(X_original: np.ndarray,
                                   X_reconstructed: np.ndarray) -> dict:
    """Returns per-group mean squared error."""
    errors = np.square(X_original - X_reconstructed)
    group_errors = {}
    for group_name, indices in FEATURE_GROUPS.items():
        valid_idx = [i for i in indices if i < errors.shape[1]]
        if valid_idx:
            group_errors[group_name] = round(
                float(np.mean(errors[:, valid_idx])), 6
            )
    return group_errors


# ─────────────────────────────────────────────
# DEMO / SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n━━━ Explainability demo ━━━")

    classifier  = joblib.load(os.path.join(MODELS_DIR, "lgbm_classifier.pkl"))
    label_enc   = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    encoder_mdl = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "vae_encoder.keras"),
        custom_objects={"Sampling": Sampling}
    )
    feat_names = get_feature_names()
    print(f"  Total features: {len(feat_names)} "
          f"({len(feat_names)-8} original + 8 latent)")

    X_test = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"), allow_pickle=True)

    attack_mask = y_test != "normal"
    X_atk = X_test[attack_mask][:5]
    y_atk = y_test[attack_mask][:5]

    z_mean, _, _ = encoder_mdl.predict(X_atk, verbose=0)
    X_combined   = np.concatenate([X_atk, z_mean.astype(np.float32)], axis=1)

    print(f"\n  Explaining {len(X_atk)} attack samples...\n")
    for i in range(len(X_atk)):
        exp = explain_sample(
            X_combined[i:i+1], classifier, label_enc, feat_names
        )
        print(f"  Sample {i} | true={str(y_atk[i]):<10} "
              f"pred={exp['predicted_label']:<10} "
              f"conf={exp['confidence']:.3f}")
        print(f"  → {exp['summary']}")
        print(f"  Top feature: {exp['top_features'][0]['feature']} "
              f"(SHAP={exp['top_features'][0]['shap_value']:.4f})\n")

    exp_example = explain_sample(
        X_combined[0:1], classifier, label_enc, feat_names
    )
    path = plot_shap_bar(exp_example)
    print(f"  SHAP bar chart saved → {path}")

    print("\n✅  Explainability demo complete!\n")