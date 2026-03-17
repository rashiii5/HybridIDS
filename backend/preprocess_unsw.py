import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os

# Map UNSW-NB15's 9 attack types down to your 4-class structure
ATTACK_MAP = {
    'Normal':        'normal',
    '':              'normal',
    'DoS':           'DoS',
    'Reconnaissance':'Probe',
    'Fuzzers':       'Probe',
    'Analysis':      'Probe',
    'Backdoors':     'R2L',
    'Exploits':      'R2L',
    'Shellcode':     'U2R',
    'Worms':         'U2R',
    'Generic':       'DoS',
}

CATEGORICAL_COLS = ['proto', 'service', 'state']

LABEL_MAP = {'normal': -1, 'DoS': 0, 'Probe': 1, 'R2L': 2, 'U2R': 3}

def load_unsw(path):
    df = pd.read_csv(path)

    # Drop ID and raw label cols if present
    df.drop(columns=['id', 'label'], errors='ignore', inplace=True)

    # Normalize attack_cat
    df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
    df['attack_cat'] = df['attack_cat'].map(ATTACK_MAP).fillna('DoS')

    # Separate labels
    y_str = df.pop('attack_cat')

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Fill any NaN/inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df, y_str

def preprocess_unsw(train_path, test_path, save_dir='data/processed_unsw'):
    os.makedirs(save_dir, exist_ok=True)

    df_train, y_train_str = load_unsw(train_path)
    df_test,  y_test_str  = load_unsw(test_path)

    # Align columns — train is the reference
    feature_cols = df_train.columns.tolist()
    df_test = df_test.reindex(columns=feature_cols, fill_value=0)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train).astype(np.float32)
    X_test  = scaler.transform(df_test).astype(np.float32)

    # Encode labels
    le = LabelEncoder()
    le.fit(list(LABEL_MAP.keys()))
    y_train = np.array([LABEL_MAP[l] for l in y_train_str])
    y_test  = np.array([LABEL_MAP[l] for l in y_test_str])

    # Save
    np.save(f'{save_dir}/X_train.npy', X_train)
    np.save(f'{save_dir}/X_test.npy',  X_test)
    np.save(f'{save_dir}/y_train.npy', y_train)
    np.save(f'{save_dir}/y_test.npy',  y_test)
    joblib.dump(scaler,       f'{save_dir}/scaler.pkl')
    joblib.dump(feature_cols, f'{save_dir}/feature_columns.pkl')
    joblib.dump(le,           f'{save_dir}/label_encoder.pkl')

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    return X_train, X_test, y_train, y_test, feature_cols