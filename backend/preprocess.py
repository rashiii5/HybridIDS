"""
1.
preprocess.py
-------------
Handles all data loading, cleaning, encoding, scaling, and saving
for the NSL-KDD dataset. Run this once before training anything.

Usage:
    python preprocess.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "backend", "models")

os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# COLUMN NAMES  (NSL-KDD has no header row)
# ─────────────────────────────────────────────
COLUMNS = [
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
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# ─────────────────────────────────────────────
# ATTACK → CATEGORY MAPPING
# ─────────────────────────────────────────────
ATTACK_MAP = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    # Normal
    "normal": "normal"
}

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_data(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, header=None, names=COLUMNS)
    print(f"  Loaded {filename}: {df.shape}")
    return df


# ─────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop(columns=["difficulty"])          # not a feature
    df["label"] = df["label"].str.strip().str.lower()
    df["category"] = df["label"].map(ATTACK_MAP)
    unknown = df["category"].isna().sum()
    if unknown > 0:
        print(f"  ⚠  {unknown} unknown labels → dropping")
        df = df.dropna(subset=["category"])
    print(f"  Clean: {before} → {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# ENCODE + SCALE
# ─────────────────────────────────────────────
def encode_and_scale(train: pd.DataFrame,
                     test:  pd.DataFrame,
                     fit_scaler: bool = True):
    """
    One-hot encode categoricals, then StandardScale numerics.
    Scaler is fit on train only and applied to both splits.
    Returns X_train, X_test as numpy arrays + category arrays.
    """

    # Separate labels
    y_train = train["category"].values
    y_test  = test["category"].values

    # Drop label columns before encoding
    train = train.drop(columns=["label", "category"])
    test  = test.drop(columns=["label", "category"])

    # One-hot encode
    train_enc = pd.get_dummies(train, columns=CATEGORICAL_COLS)
    test_enc  = pd.get_dummies(test,  columns=CATEGORICAL_COLS)

    # Align columns — test may be missing some dummy cols
    train_enc, test_enc = train_enc.align(
        test_enc, join="left", axis=1, fill_value=0
    )

    print(f"  Feature dim after encoding: {train_enc.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_enc).astype(np.float32)
    X_test  = scaler.transform(test_enc).astype(np.float32)

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    # Save column order (needed for inference)
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    joblib.dump(list(train_enc.columns), cols_path)
    print(f"  Feature columns saved → {cols_path}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# SAVE PROCESSED ARRAYS
# ─────────────────────────────────────────────
def save_arrays(X_train, X_test, y_train, y_test):
    np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_test)
    print(f"  Arrays saved to {PROC_DIR}")


# ─────────────────────────────────────────────
# CLASS DISTRIBUTION SUMMARY
# ─────────────────────────────────────────────
def print_distribution(y_train, y_test):
    print("\n  ── Train distribution ──")
    train_s = pd.Series(y_train).value_counts()
    for cls, cnt in train_s.items():
        print(f"     {cls:<10} {cnt:>6}")

    print("\n  ── Test distribution ──")
    test_s = pd.Series(y_test).value_counts()
    for cls, cnt in test_s.items():
        print(f"     {cls:<10} {cnt:>6}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_preprocessing():
    print("\n━━━ STEP 1 — Loading raw data ━━━")
    train_df = load_data("KDDTrain+.txt")
    test_df  = load_data("KDDTest+.txt")

    print("\n━━━ STEP 2 — Cleaning ━━━")
    train_df = clean(train_df)
    test_df  = clean(test_df)

    print("\n━━━ STEP 3 — Encoding & Scaling ━━━")
    X_train, X_test, y_train, y_test = encode_and_scale(train_df, test_df)

    print("\n━━━ STEP 4 — Saving processed arrays ━━━")
    save_arrays(X_train, X_test, y_train, y_test)

    print("\n━━━ STEP 5 — Class distribution ━━━")
    print_distribution(y_train, y_test)

    print("\n✅  Preprocessing complete!\n")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()