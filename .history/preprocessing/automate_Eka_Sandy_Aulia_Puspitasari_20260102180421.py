import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def automate_preprocessing(file_path):
    data = pd.read_csv(file_path)

    # missing values
    data = data.drop(columns=["Cabin", "Embarked"])
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # duplicate
    data = data.drop_duplicates()

    # scaling
    scaler = StandardScaler()
    for col in ["Age", "Fare", "SibSp", "Parch", "Pclass"]:
        data[col + "_scaled"] = scaler.fit_transform(data[[col]])

    # outlier removal
    for col in ["Age", "Fare"]:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[
            (data[col] >= Q1 - 1.5 * IQR) &
            (data[col] <= Q3 + 1.5 * IQR)
        ]

    # encoding
    encoder = LabelEncoder()
    data["Sex_encoded"] = encoder.fit_transform(data["Sex"])

    # feature & target
    X = data.drop(columns=["Survived"])
    y = data["Survived"]

    # keep numeric only (IMPORTANT)
    X = X.select_dtypes(include=["int64", "float64"])

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    INPUT_PATH = "titanic.csv"
    OUTPUT_DIR = "preprocessing/processed_data"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = automate_preprocessing(INPUT_PATH)

    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
