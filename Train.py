from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

house = pd.read_csv("Data_Harga_Rumah_Surabaya.csv")

print("Shape:", house.shape)
print("\nStatistik target Price:")
print(house["Price"].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
house["Price"].hist(bins=50, ax=axes[0], color="#ffa657")
axes[0].set_title("Distribusi Harga Rumah (Original)")

np.log1p(house["Price"]).hist(bins=50, ax=axes[1], color="#79c0ff")
axes[1].set_title("Distribusi Harga Rumah log(Price+1)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
num_cols = house.select_dtypes(include="number").columns
sns.heatmap(house[num_cols].corr(), annot=True, fmt=".1f",
            cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Korelasi Fitur Numerik")
plt.tight_layout()
plt.show()


house.columns = house.columns.str.strip()

house_clean = house.copy()


house_clean['Price'] = house_clean['Price'].replace('[^0-9]', '', regex=True).astype(float)
house_clean['Luas Bangunan'] = house_clean['Luas Bangunan'].replace('[^0-9]', '', regex=True).astype(float)
house_clean['Luas Tanah'] = house_clean['Luas Tanah'].replace('[^0-9]', '', regex=True).astype(float)

for col in ['Kamar Tidur', 'Kamar Mandi', 'Jumlah Lantai']:
    house_clean[col] = pd.to_numeric(house_clean[col], errors='coerce')


house_clean = house_clean.dropna(subset=['Price', 'Luas Bangunan', 'Luas Tanah', 'Kamar Tidur', 'Kamar Mandi', 'Jumlah Lantai'])


house_clean['Rasio_Bangunan_Tanah'] = house_clean['Luas Bangunan'] / house_clean['Luas Tanah']

features = ['Luas Bangunan', 'Luas Tanah', 'Kamar Tidur', 'Kamar Mandi', 'Jumlah Lantai', 'Rasio_Bangunan_Tanah']
target = 'Price'

X_h = house[features]
y_h = np.log1p(house["SalePrice"])

X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)
print(f"Train: {X_h_train.shape[0]} | Test: {X_h_test.shape[0]}")