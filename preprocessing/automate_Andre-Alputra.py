"""
automate_Andre-Alputra.py
=========================
Script otomatisasi preprocessing Wine Quality Dataset.
Konversi dari notebook eksperimen ke script Python yang dapat dijalankan secara otomatis.

Author  : Andre Alputra
Dataset : Wine Quality (Red Wine) - UCI ML Repository
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath: Path ke file CSV dataset mentah.

    Returns:
        DataFrame berisi dataset yang sudah dimuat.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] File tidak ditemukan: {filepath}")

    # Wine Quality menggunakan separator ';'
    df = pd.read_csv(filepath, sep=';')
    print(f"[INFO] Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[INFO] Kolom: {list(df.columns)}")
    return df


def binarize_target(df: pd.DataFrame, target_col: str = 'quality',
                    threshold: int = 6) -> pd.DataFrame:
    """
    Mengubah target dari skor kontinyu menjadi label biner.
    quality >= threshold → 1 (good), quality < threshold → 0 (bad)

    Args:
        df        : DataFrame input.
        target_col: Nama kolom target.
        threshold : Batas pemisah kualitas.

    Returns:
        DataFrame dengan target yang sudah dibinarisasi.
    """
    df = df.copy()
    df[target_col] = (df[target_col] >= threshold).astype(int)
    counts = df[target_col].value_counts()
    print(f"[INFO] Binarisasi target selesai — Good(1): {counts.get(1, 0)}, Bad(0): {counts.get(0, 0)}")
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.

    Args:
        df: DataFrame input.

    Returns:
        DataFrame tanpa duplikat.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[INFO] Duplikat dihapus: {removed} baris (sisa: {len(df)} baris)")
    return df


def remove_outliers_iqr(df: pd.DataFrame, feature_cols: list,
                        factor: float = 3.0) -> pd.DataFrame:
    """
    Menghapus outlier menggunakan metode IQR (Interquartile Range).

    Args:
        df          : DataFrame input.
        feature_cols: Daftar kolom fitur yang akan dicek outlier-nya.
        factor      : Faktor pengali IQR (default 3.0 untuk konservatif).

    Returns:
        DataFrame tanpa outlier ekstrem.
    """
    df_clean = df.copy()
    total_removed = 0

    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        removed = before - len(df_clean)
        total_removed += removed
        if removed > 0:
            print(f"[INFO] Outlier [{col}]: {removed} baris dihapus (range: {lower:.3f} — {upper:.3f})")

    df_clean = df_clean.reset_index(drop=True)
    print(f"[INFO] Total outlier dihapus: {total_removed} baris (sisa: {len(df_clean)} baris)")
    return df_clean


def split_features_target(df: pd.DataFrame,
                           target_col: str = 'quality'):
    """
    Memisahkan fitur (X) dan target (y).

    Args:
        df        : DataFrame input.
        target_col: Nama kolom target.

    Returns:
        Tuple (X, y) — fitur dan target.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    print(f"[INFO] Fitur (X): {X.shape} | Target (y): {y.shape}")
    return X, y


def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Membagi data menjadi train set dan test set secara stratified.

    Args:
        X           : Fitur.
        y           : Target.
        test_size   : Proporsi data test (default 0.2 = 20%).
        random_state: Seed untuk reproducibility.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train set: {X_train.shape} | Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Melakukan standarisasi fitur menggunakan StandardScaler.
    Scaler di-fit pada data train saja, lalu di-transform ke test.

    Args:
        X_train: Fitur training set.
        X_test : Fitur test set.

    Returns:
        Tuple (X_train_scaled, X_test_scaled) sebagai DataFrame.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    print(f"[INFO] Standarisasi selesai - Mean train ~= {X_train_scaled.mean().mean():.4f}, Std ~= {X_train_scaled.std().mean():.4f}")
    return X_train_scaled, X_test_scaled


def save_results(X_train, X_test, y_train, y_test,
                 output_dir: str = 'winequality_preprocessing') -> None:
    """
    Menyimpan hasil preprocessing ke folder output dalam format CSV.

    Args:
        X_train   : Fitur training.
        X_test    : Fitur testing.
        y_train   : Label training.
        y_test    : Label testing.
        output_dir: Direktori output.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.reset_index(drop=True).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.reset_index(drop=True).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print(f"\n[INFO] Hasil preprocessing disimpan ke folder '{output_dir}':")
    for fname in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath)
        data = pd.read_csv(fpath)
        print(f"  [OK] {fname}: {data.shape[0]} baris x {data.shape[1]} kolom ({size:,} bytes)")


# ---------------------------------------------
# FUNGSI UTAMA (PIPELINE)
# ---------------------------------------------

def preprocess_pipeline(raw_data_path: str,
                         output_dir: str = 'winequality_preprocessing',
                         target_col: str = 'quality',
                         threshold: int = 6,
                         test_size: float = 0.2,
                         random_state: int = 42) -> None:
    """
    Pipeline preprocessing lengkap dari raw data hingga data siap latih.

    Args:
        raw_data_path: Path ke file CSV dataset mentah.
        output_dir   : Direktori untuk menyimpan hasil preprocessing.
        target_col   : Nama kolom target.
        threshold    : Ambang batas binarisasi target.
        test_size    : Proporsi data test.
        random_state : Seed untuk reproducibility.
    """
    print("=" * 55)
    print("  PIPELINE PREPROCESSING - Wine Quality Dataset")
    print("=" * 55)

    # Step 1: Load
    print("\n[STEP 1] Memuat dataset...")
    df = load_data(raw_data_path)

    # Step 2: Binarisasi target
    print("\n[STEP 2] Binarisasi target...")
    df = binarize_target(df, target_col=target_col, threshold=threshold)

    # Step 3: Hapus duplikat
    print("\n[STEP 3] Menghapus duplikat...")
    df = handle_duplicates(df)

    # Step 4: Hapus outlier
    print("\n[STEP 4] Menangani outlier...")
    feature_cols = [c for c in df.columns if c != target_col]
    df = remove_outliers_iqr(df, feature_cols, factor=3.0)

    # Step 5: Feature-target split
    print("\n[STEP 5] Memisahkan fitur dan target...")
    X, y = split_features_target(df, target_col=target_col)

    # Step 6: Train-test split
    print("\n[STEP 6] Membagi data train dan test...")
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 7: Standarisasi
    print("\n[STEP 7] Standarisasi fitur...")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Step 8: Simpan
    print("\n[STEP 8] Menyimpan hasil preprocessing...")
    save_results(X_train_scaled, X_test_scaled, y_train, y_test, output_dir)

    print("\n" + "=" * 55)
    print("  [SUKSES] Preprocessing selesai! Data siap dilatih.")
    print("=" * 55)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Path relatif dari folder preprocessing/ ke dataset raw
    RAW_DATA_PATH = '../winequality-red.csv'
    OUTPUT_DIR    = 'winequality_preprocessing'

    preprocess_pipeline(
        raw_data_path=RAW_DATA_PATH,
        output_dir=OUTPUT_DIR,
        target_col='quality',
        threshold=6,
        test_size=0.2,
        random_state=42
    )
