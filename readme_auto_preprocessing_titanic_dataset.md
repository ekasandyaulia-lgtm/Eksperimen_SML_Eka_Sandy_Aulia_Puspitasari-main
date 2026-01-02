# Eksperimen SML – Auto Preprocessing Dataset Titanic

Repository ini merupakan bagian dari **Eksperimen Sistem Machine Learning (SML)** yang berfokus pada **otomatisasi data preprocessing** menggunakan **Python** dan **GitHub Actions (CI/CD)**. Dataset yang digunakan adalah **Titanic – Machine Learning from Disaster** dari Kaggle.

## 📌 Tujuan Proyek
Tujuan utama proyek ini adalah:
1. Melakukan eksplorasi dan preprocessing dataset Titanic secara terstruktur.
2. Mengotomatisasi proses preprocessing menggunakan script Python.
3. Mengintegrasikan preprocessing ke dalam **GitHub Actions** sehingga berjalan otomatis setiap ada perubahan kode.
4. Menyimpan hasil preprocessing (train–test split) secara otomatis ke repository.

Proyek ini memenuhi kriteria eksperimen dan otomatisasi pada pengembangan Sistem Machine Learning.

---

## 📊 Dataset
- **Nama Dataset**: Titanic – Machine Learning from Disaster  
- **Sumber**: Kaggle  
- **Link**: https://www.kaggle.com/competitions/titanic  
- **Jenis Data**: Tabular  
- **Jumlah Data**: 891 baris  
- **Target**: `Survived` (0 = Tidak selamat, 1 = Selamat)

Dataset `train.csv` diubah namanya menjadi **`titanic.csv`** untuk konsistensi dalam proyek.

---

## 📂 Struktur Folder

```
Eksperimen_SML_Eka_Sandy_Aulia_Puspitasari-main
│
├── .github/workflows/
│   └── preprocess.yml        # Workflow GitHub Actions
│
├── preprocessing/
│   ├── automate_Eka_Sandy_Aulia_Puspitasari.py  # Script preprocessing otomatis
│   └── processed_data/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── titanic.csv                # Dataset input
├── Eksperimen_SML_Eka_Sandy_Aulia_Puspitasari.ipynb  # Notebook EDA & preprocessing
├── Eksperimen_SML_Eka_Sandy_Aulia_Puspitasari.txt    # Dokumentasi eksperimen
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)
EDA dilakukan menggunakan **Jupyter Notebook** untuk memahami karakteristik data, meliputi:
- Histogram (`Age`, `Fare`)
- Boxplot untuk deteksi outlier
- Scatterplot antar fitur numerik
- Countplot fitur kategorikal (`Sex`, `Pclass`)
- Correlation Matrix

Notebook EDA tersedia pada file:
```
Eksperimen_SML_Eka_Sandy_Aulia_Puspitasari.ipynb
```

---

## ⚙️ Tahapan Data Preprocessing
Preprocessing dilakukan secara manual (di notebook) dan otomatis (melalui script Python), dengan tahapan:

1. **Handling Missing Values**
   - Menghapus kolom `Cabin` dan `Embarked`
   - Mengisi nilai kosong `Age` dengan median

2. **Menghapus Data Duplikat**

3. **Standarisasi Fitur Numerik**
   - `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`

4. **Outlier Removal**
   - Menggunakan metode **IQR** pada `Age` dan `Fare`

5. **Encoding Data Kategorikal**
   - `Sex` → `Sex_encoded` menggunakan `LabelEncoder`

6. **Feature Selection**
   - Mengambil hanya fitur numerik

7. **Train-Test Split**
   - 80% data latih
   - 20% data uji

8. **Menyimpan Hasil Preprocessing**
   - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

---

## 🤖 Automasi dengan GitHub Actions
Workflow GitHub Actions didefinisikan pada file:
```
.github/workflows/preprocess.yml
```

### Workflow ini akan berjalan ketika:
- Push ke branch `main`
- Manual trigger (`workflow_dispatch`)

### Tahapan Workflow:
1. Checkout repository
2. Setup Python 3.11
3. Install dependencies (`numpy`, `pandas`, `scikit-learn`)
4. Menjalankan script preprocessing otomatis
5. Commit & push hasil preprocessing ke repository

---

## ▶️ Cara Menjalankan Manual (Local)

```bash
pip install numpy pandas scikit-learn
python preprocessing/automate_Eka_Sandy_Aulia_Puspitasari.py
```

Hasil preprocessing akan tersimpan di folder:
```
preprocessing/processed_data/
```

---

## 📦 Output
Hasil akhir preprocessing berupa:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

File-file ini siap digunakan untuk tahap **training model Machine Learning**.

---

## 👤 Author
**Eka Sandy Aulia Puspitasari**  
Repository: https://github.com/ekasandyaulia-lgtm

---

## ✅ Kesimpulan
Proyek ini menunjukkan penerapan **data preprocessing terstruktur** serta **otomatisasi pipeline Machine Learning** menggunakan GitHub Actions. Dengan pendekatan ini, proses preprocessing menjadi konsisten, reproducible, dan siap diintegrasikan ke tahap pelatihan model selanjutnya.

