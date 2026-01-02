import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

data = pd.read_csv('train.csv')
data.head()

# HISTOGRAM
plt.figure(figsize=(8,5))
data['Age'].hist(bins=30)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,5))
data['Fare'].hist(bins=30)
plt.title('Histogram of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# BOXPLOT
plt.figure(figsize=(8,5))
sns.boxplot(x=data['Age'])
plt.title('Boxplot of Age')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=data['Fare'])
plt.title('Boxplot of Fare')
plt.show()

# SCATTERPLOT
plt.figure(figsize=(8,5))
sns.scatterplot(x=data['Age'], y=data['Fare'], hue=data['Survived'])
plt.title('Scatterplot of Age vs Fare (Survived)')
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=data['SibSp'], y=data['Parch'], hue=data['Survived'])
plt.title('Scatterplot of SibSp vs Parch (Survived)')
plt.show()

# COUNT PLOT (KATEGORIKAL)
plt.figure(figsize=(8,5))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Passenger Class vs Survived')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Sex vs Survived')
plt.show()

# Matriks Korelasi
numeric_data = data.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 1.Menghapus atau Menangani Data Kosong (Missing Values)
data = data.drop(columns=['Cabin', 'Embarked'])
data['Age'].fillna(data['Age'].median(), inplace=True)

# 2. Menghapus Data Duplikat
data.duplicated().sum() # untuk mengecek jumlah data duplikat
data.drop_duplicates(inplace=True)

#3. Normalisasi atau Standarisasi
# Standarisasi numerik
scaler = StandardScaler()
for col in ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']:
    data[col + '_scaled'] = scaler.fit_transform(data[[col]])

# 4. Deteksi dan Penanganan Outlier
for col in ['Age', 'Fare']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# 5. Encoding Data Kategorikal
label_encoder = LabelEncoder()
data['Sex_encoded'] = label_encoder.fit_transform(data['Sex'])

# 6. Binning (Pengelompokan Data)
data['Age_binned'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior'])
data['Fare_binned'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])     
