import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def automate_preprocessing(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # 1. Menghapus atau Menangani Data Kosong (Missing Values)
    data = data.drop(columns=['Cabin', 'Embarked'])
    data['Age'] = data['Age'].fillna(data['Age'].median())

    # 2. Menghapus Data Duplikat
    data = data.drop_duplicates()

    # 3. Normalisasi / Standarisasi
    scaler = StandardScaler()
    for col in ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']:
        data[col + '_scaled'] = scaler.fit_transform(data[[col]])

    # 4. Deteksi dan Penanganan Outlier
    for col in ['Age', 'Fare']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]

    # 5. Encoding Data Kategorikal
    label_encoder = LabelEncoder()
    data['Sex_encoded'] = label_encoder.fit_transform(data['Sex'])

    # 6. Binning (Pengelompokan Data)
    data['Age_binned'] = pd.cut(
        data['Age'],
        bins=[0, 12, 20, 40, 60, 80],
        labels=['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior']
    )

    data['Fare_binned'] = pd.qcut(
        data['Fare'],
        q=4,
        labels=['Low', 'Medium', 'High', 'Very_High']
    )

    # 7. Feature dan Target
    X = data.drop(columns=['Survived'])
    y = data['Survived']

    # 8. Splitting Data Latih dan Data Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

# Main execution
if __name__ == "__main__":
    import os

    INPUT_PATH = "titanic.csv"
    OUTPUT_DIR = "processed_data"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = automate_preprocessing(INPUT_PATH)

    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
