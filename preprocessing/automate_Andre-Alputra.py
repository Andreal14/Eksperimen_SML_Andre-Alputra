"""
automate_preprocessing_heart.py
================================
Script otomatis preprocessing Heart Disease Dataset.
Menghasilkan data siap latih dalam folder heart_preprocessing/.

Author  : Andre Alputra
Dataset : Heart Disease (UCI / Kaggle)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
RAW_FILE    = "heart.csv"
OUTPUT_DIR  = "heart_preprocessing"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari file CSV."""
    print(f"[INFO] Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Shape awal: {df.shape}")
    return df

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Periksa dan tangani missing values."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"[INFO] Missing values ditemukan:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"[INFO] Shape setelah drop NA: {df.shape}")
    else:
        print("[INFO] Tidak ada missing values.")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(f"[INFO] Menghapus {n_dup} baris duplikat.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[INFO] Tidak ada duplikat.")
    return df

def handle_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Hapus outlier menggunakan IQR method."""
    original_len = len(df)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    df = df.reset_index(drop=True)
    print(f"[INFO] Outlier dihapus: {original_len - len(df)} baris. Shape: {df.shape}")
    return df

def split_and_scale(df: pd.DataFrame, target_col: str):
    """Split data dan lakukan feature scaling."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    print(f"[INFO] Train size: {X_train_scaled.shape} | Test size: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def save_preprocessed(X_train, X_test, y_train, y_test, output_dir: str):
    """Simpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    print(f"[INFO] Data disimpan ke folder: {output_dir}/")
    print(f"       - X_train.csv: {X_train.shape}")
    print(f"       - X_test.csv : {X_test.shape}")
    print(f"       - y_train.csv: {y_train.shape}")
    print(f"       - y_test.csv : {y_test.shape}")

def run_preprocessing():
    """Fungsi utama pipeline preprocessing."""
    print("="*55)
    print("  PREPROCESSING - Heart Disease Dataset")
    print("="*55)

    # Kolom numerik untuk outlier handling
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    df = load_data(RAW_FILE)
    df = check_missing(df)
    df = remove_duplicates(df)
    df = handle_outliers(df, numeric_cols)

    X_train, X_test, y_train, y_test = split_and_scale(df, target_col='target')
    save_preprocessed(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    print("\n[SUKSES] Preprocessing selesai!")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()
