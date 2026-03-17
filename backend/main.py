"""
main.py
-------
FastAPI backend for the Hybrid IDS system.

Endpoints:
  POST /api/detect/file       — upload CSV, returns predictions
  POST /api/detect/single     — single sample JSON, returns prediction
  GET  /api/metrics           — precomputed model comparison metrics
  GET  /api/roc               — ROC curve data
  GET  /api/confusion         — confusion matrix data
  GET  /api/shap/{sample_id}  — SHAP explanation for a sample
  GET  /api/summary           — summary of last detection run
  GET  /api/threshold         — threshold metadata
  GET  /health                — health check

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import io
import json
import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "backend", "models")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

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
# APP INIT
# ─────────────────────────────────────────────
app = FastAPI(
    title="Hybrid IDS API",
    description="Hybrid VAE + LightGBM Network Intrusion Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─────────────────────────────────────────────
# MODEL REGISTRY  (loaded once on startup)
# ─────────────────────────────────────────────
class Models:
    encoder      = None
    decoder      = None
    classifier   = None
    scaler       = None
    label_enc    = None
    feat_cols    = None
    threshold    = None
    threshold_data = None


@app.on_event("startup")
async def load_models():
    print("  Loading models...")
    Models.encoder  = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "vae_encoder.keras"),
        custom_objects={"Sampling": Sampling}
    )
    Models.decoder  = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "vae_decoder.keras"),
        custom_objects={"Sampling": Sampling}
    )
    Models.classifier  = joblib.load(os.path.join(MODELS_DIR, "lgbm_classifier.pkl"))
    Models.scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    Models.label_enc   = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    Models.feat_cols   = list(joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl")))
    thresh             = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
    Models.threshold   = thresh["threshold"]
    Models.threshold_data = thresh
    print(f"  ✓ Models loaded. Threshold = {Models.threshold:.6f}")


# ─────────────────────────────────────────────
# SESSION CACHE  (stores last detection run)
# ─────────────────────────────────────────────
session_cache = {
    "last_results":    None,
    "last_X":         None,
    "last_X_combined": None,
    "last_summary":   None,
}

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    for col in ["label", "difficulty", "category"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df_enc = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
    df_enc = df_enc.reindex(columns=Models.feat_cols, fill_value=0)
    return Models.scaler.transform(df_enc).astype(np.float32)


def run_pipeline(X: np.ndarray) -> tuple:
    """Returns (results, X_combined_for_anomalies)"""
    # Reconstruction errors
    z_mean, _, z = Models.encoder.predict(X, batch_size=512, verbose=0)
    X_recon      = Models.decoder.predict(z, batch_size=512, verbose=0)
    mse          = np.mean(np.square(X - X_recon), axis=1)
    errors       = np.log1p(mse)
    is_anomaly   = errors > Models.threshold

    # Classify anomalies
    anomaly_idx  = np.where(is_anomaly)[0]
    attack_labels = {}
    attack_probs  = {}
    X_combined_cache = {}

    if len(anomaly_idx) > 0:
        X_anom   = X[anomaly_idx]
        z_anom, _, _ = Models.encoder.predict(X_anom, batch_size=512, verbose=0)
        X_comb   = np.concatenate([X_anom, z_anom.astype(np.float32)], axis=1)

        probs    = Models.classifier.predict_proba(X_comb)
        preds    = np.argmax(probs, axis=1)
        labels   = Models.label_enc.inverse_transform(preds)

        for i, idx in enumerate(anomaly_idx):
            attack_labels[int(idx)] = str(labels[i])
            attack_probs[int(idx)]  = {
                str(cls): round(float(probs[i][j]), 4)
                for j, cls in enumerate(Models.label_enc.classes_)
            }
            X_combined_cache[int(idx)] = X_comb[i]

    results = []
    for i in range(len(X)):
        if is_anomaly[i]:
            label      = attack_labels[i]
            probs_dict = attack_probs[i]
            confidence = round(float(max(probs_dict.values())), 4)
        else:
            label      = "normal"
            probs_dict = {}
            confidence = round(
                float(max(0.0, min(1.0,
                    1.0 - errors[i] / Models.threshold))), 4
            )

        results.append({
            "sample_id":            i,
            "reconstruction_error": round(float(errors[i]), 6),
            "is_anomaly":           bool(is_anomaly[i]),
            "label":                label,
            "confidence":           confidence,
            "attack_probs":         probs_dict,
        })

    return results, X_combined_cache


def make_summary(results: list) -> dict:
    total  = len(results)
    labels = [r["label"] for r in results]
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    n_anom = sum(1 for r in results if r["is_anomaly"])
    return {
        "total_samples":    total,
        "normal_count":     counts.get("normal", 0),
        "anomaly_count":    n_anom,
        "attack_breakdown": {k: v for k, v in counts.items() if k != "normal"},
        "detection_rate":   round(n_anom / total, 4) if total > 0 else 0,
        "label_counts":     counts,
    }


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────
class SingleSample(BaseModel):
    duration:                 float = 0
    protocol_type:            str   = "tcp"
    service:                  str   = "http"
    flag:                     str   = "SF"
    src_bytes:                float = 0
    dst_bytes:                float = 0
    land:                     float = 0
    wrong_fragment:           float = 0
    urgent:                   float = 0
    hot:                      float = 0
    num_failed_logins:        float = 0
    logged_in:                float = 0
    num_compromised:          float = 0
    root_shell:               float = 0
    su_attempted:             float = 0
    num_root:                 float = 0
    num_file_creations:       float = 0
    num_shells:               float = 0
    num_access_files:         float = 0
    num_outbound_cmds:        float = 0
    is_host_login:            float = 0
    is_guest_login:           float = 0
    count:                    float = 0
    srv_count:                float = 0
    serror_rate:              float = 0
    srv_serror_rate:          float = 0
    rerror_rate:              float = 0
    srv_rerror_rate:          float = 0
    same_srv_rate:            float = 0
    diff_srv_rate:            float = 0
    srv_diff_host_rate:       float = 0
    dst_host_count:           float = 0
    dst_host_srv_count:       float = 0
    dst_host_same_srv_rate:   float = 0
    dst_host_diff_srv_rate:   float = 0
    dst_host_same_src_port_rate: float = 0
    dst_host_srv_diff_host_rate: float = 0
    dst_host_serror_rate:     float = 0
    dst_host_srv_serror_rate: float = 0
    dst_host_rerror_rate:     float = 0
    dst_host_srv_rerror_rate: float = 0


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": Models.encoder is not None}


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Hybrid IDS API running. Frontend not yet built."}


@app.post("/api/detect/file")
async def detect_file(file: UploadFile = File(...)):
    """Upload a CSV file of network traffic samples for batch detection."""
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(400, "Only CSV and TXT files are supported.")

    t0 = time.time()
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), header=None)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Assign column names if no header
    from backend.preprocess import COLUMNS
    if df.shape[1] == len(COLUMNS):
        df.columns = COLUMNS
    elif df.shape[1] == len(COLUMNS) - 1:
        df.columns = COLUMNS[:-1]   # no difficulty column

    try:
        X = preprocess(df)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing failed: {e}")

    results, X_combined_cache = run_pipeline(X)
    summary = make_summary(results)

    # Cache for SHAP endpoint
    session_cache["last_results"]     = results
    session_cache["last_X"]           = X
    session_cache["last_X_combined"]  = X_combined_cache
    session_cache["last_summary"]     = summary

    elapsed = round((time.time() - t0) * 1000, 2)

    return JSONResponse({
        "status":       "ok",
        "elapsed_ms":   elapsed,
        "summary":      summary,
        "predictions":  results,
    })


@app.post("/api/detect/single")
async def detect_single(sample: SingleSample):
    """Detect a single network traffic sample."""
    t0  = time.time()
    df  = pd.DataFrame([sample.dict()])

    try:
        X = preprocess(df)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing failed: {e}")

    results, X_combined_cache = run_pipeline(X)
    elapsed = round((time.time() - t0) * 1000, 2)

    # Cache
    session_cache["last_results"]    = results
    session_cache["last_X"]          = X
    session_cache["last_X_combined"] = X_combined_cache

    return JSONResponse({
        "status":     "ok",
        "elapsed_ms": elapsed,
        "prediction": results[0],
    })


@app.get("/api/metrics")
def get_metrics():
    """Return precomputed model comparison metrics from metrics.json."""
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(404, "metrics.json not found. Run evaluate.py first.")
    with open(path) as f:
        return JSONResponse(json.load(f))


@app.get("/api/summary")
def get_summary():
    """Return summary of the last detection run."""
    if session_cache["last_summary"] is None:
        return JSONResponse({"message": "No detection run yet."})
    return JSONResponse(session_cache["last_summary"])


@app.get("/api/threshold")
def get_threshold():
    """Return threshold metadata."""
    td = Models.threshold_data
    return JSONResponse({
        "threshold":    round(float(td["threshold"]), 6),
        "t_percentile": round(float(td["t_percentile"]), 6),
        "t_roc":        round(float(td["t_roc"]), 6),
        "t_f1":         round(float(td["t_f1"]), 6),
        "roc_auc":      round(float(td["roc_auc"]), 4),
    })


@app.get("/api/roc")
def get_roc():
    """Return ROC curve data for the VAE anomaly detector."""
    try:
        X_test = np.load(os.path.join(PROC_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"), allow_pickle=True)

        z_mean, _, z = Models.encoder.predict(X_test, batch_size=512, verbose=0)
        X_recon      = Models.decoder.predict(z, batch_size=512, verbose=0)
        mse          = np.mean(np.square(X_test - X_recon), axis=1)
        errors       = np.log1p(mse)
        y_binary     = (y_test != "normal").astype(int)

        fpr, tpr, thresholds = roc_curve(y_binary, errors)
        roc_auc = float(auc(fpr, tpr))

        # Downsample to 200 points for response size
        step = max(1, len(fpr) // 200)
        return JSONResponse({
            "fpr":     [round(float(v), 4) for v in fpr[::step]],
            "tpr":     [round(float(v), 4) for v in tpr[::step]],
            "auc":     round(roc_auc, 4),
            "threshold": round(float(Models.threshold), 6),
        })
    except Exception as e:
        raise HTTPException(500, f"ROC computation failed: {e}")


@app.get("/api/confusion")
def get_confusion():
    """Return confusion matrix for the full hybrid classifier on test set."""
    try:
        X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
        y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"), allow_pickle=True)

        ATTACK_CLASSES = ["DoS", "Probe", "R2L", "U2R"]
        mask = np.isin(y_test, ATTACK_CLASSES)
        X_atk = X_test[mask]
        y_atk = y_test[mask]

        z_mean, _, _ = Models.encoder.predict(X_atk, batch_size=512, verbose=0)
        X_comb = np.concatenate([X_atk, z_mean.astype(np.float32)], axis=1)

        preds      = Models.classifier.predict(X_comb)
        pred_labels = Models.label_enc.inverse_transform(preds)
        true_enc   = Models.label_enc.transform(y_atk)

        cm = confusion_matrix(true_enc, preds).tolist()
        return JSONResponse({
            "matrix":  cm,
            "classes": [str(c) for c in Models.label_enc.classes_],
        })
    except Exception as e:
        raise HTTPException(500, f"Confusion matrix failed: {e}")


@app.get("/api/shap/{sample_id}")
def get_shap(sample_id: int):
    """Return SHAP explanation for a sample from the last detection run."""
    from backend.explainability import explain_sample, get_feature_names

    cache = session_cache["last_X_combined"]
    if cache is None or sample_id not in cache:
        raise HTTPException(
            404,
            f"Sample {sample_id} not found in cache. "
            "Run a detection first, and the sample must be an anomaly."
        )

    X_comb     = cache[sample_id].reshape(1, -1)
    feat_names = get_feature_names()

    try:
        explanation = explain_sample(
            X_comb, Models.classifier,
            Models.label_enc, feat_names
        )
        return JSONResponse(explanation)
    except Exception as e:
        raise HTTPException(500, f"SHAP explanation failed: {e}")


@app.get("/api/sample-csv")
def get_sample_csv():
    """Return a sample CSV snippet for testing the file upload."""
    # Return 5 rows from the test set as downloadable CSV
    try:
        X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
        feat_cols = Models.feat_cols
        df = pd.DataFrame(X_test[:5], columns=feat_cols)
        return JSONResponse({
            "message": "Use KDDTest+.txt format for file upload.",
            "columns": feat_cols[:10],
            "note": "Upload raw NSL-KDD format CSV with 41 or 43 columns."
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# FRONTEND ROUTES  (serve HTML pages)
# ─────────────────────────────────────────────
@app.get("/{page}.html")
def serve_page(page: str):
    path = os.path.join(FRONTEND_DIR, f"{page}.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(404, f"Page {page}.html not found.")