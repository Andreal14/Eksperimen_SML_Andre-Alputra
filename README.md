# Eksperimen_SML_Andre-Alputra

Repositori ini berisi eksperimen dan pipeline preprocessing untuk dataset **Wine Quality (Red Wine)** sebagai bagian dari Proyek Akhir kelas **Membangun Sistem Machine Learning (MSML)**.

## 📁 Struktur Repositori

```
Eksperimen_SML_Andre-Alputra/
├── .github/
│   └── workflows/
│       └── preprocessing.yml        ← GitHub Actions (otomatisasi preprocessing)
├── preprocessing/
│   ├── Eksperimen_Andre-Alputra.ipynb   ← Notebook eksperimen
│   ├── automate_Andre-Alputra.py        ← Script preprocessing otomatis
│   └── winequality_preprocessing/       ← Output: data siap latih
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── winequality-red.csv              ← Dataset mentah (raw)
└── README.md
```

## 🍷 Dataset

**Wine Quality (Red Wine)** dari UCI Machine Learning Repository.

- **Sumber**: https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Jumlah sampel**: 1.599 baris
- **Fitur**: 11 fitur physicochemical
- **Target**: Binary classification — `1` (good, quality ≥ 6) / `0` (bad, quality < 6)

## ⚙️ Pipeline Preprocessing

1. **Load data** dari CSV (separator `;`)
2. **Binarisasi target** — quality ≥ 6 = good (1), else bad (0)
3. **Hapus duplikat**
4. **Hapus outlier** dengan metode IQR (factor = 3.0)
5. **Feature-Target split**
6. **Train-Test split** — 80% train, 20% test (stratified)
7. **Standarisasi** dengan StandardScaler

## 🚀 Cara Menjalankan

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Jalankan dari folder preprocessing/
cd preprocessing
python automate_Andre-Alputra.py
```

## 🔄 GitHub Actions

Workflow `preprocessing.yml` otomatis berjalan setiap kali:
- File `winequality-red.csv` diperbarui, **atau**
- Script `automate_Andre-Alputra.py` diubah

Hasil preprocessing akan otomatis di-commit ke folder `preprocessing/winequality_preprocessing/`.
