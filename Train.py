#Import
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
house = pd.read_csv("Data_Harga_Rumah_Surabaya.csv")

#Preprocessing
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

X = house_clean[features]
y = np.log1p(house_clean["Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

#Training
RandomForest = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

XGBoost = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', XGBRegressor())
])

RandomForest.fit(X_train, y_train)
XGBoost.fit(X_train, y_train)

RandomForest_pred = RandomForest.predict(X_test)
XGBoost_pred = XGBoost.predict(X_test)

#Evaluasi 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


mae_log = mean_absolute_error(y_test, RandomForest_pred)
rmse_log = np.sqrt(mean_squared_error(y_test, RandomForest_pred))
r2_log = r2_score(y_test, RandomForest_pred)

print("Random Forest (Log)")
print(f"MAE: {mae_log:.4f}")
print(f"RMSE: {rmse_log:.4f}")
print(f"R2 Score: {r2_log:.4f}\n")

mae_log = mean_absolute_error(y_test, XGBoost_pred)
rmse_log = np.sqrt(mean_squared_error(y_test, XGBoost_pred))
r2_log = r2_score(y_test, XGBoost_pred)

print("XGBoost (Log)")
print(f"MAE: {mae_log:.4f}")
print(f"RMSE: {rmse_log:.4f}")
print(f"R2 Score: {r2_log:.4f}\n")

rf_importances = RandomForest.named_steps['model'].feature_importances_
xgb_importances = XGBoost.named_steps['model'].feature_importances_

rf_importance_series = pd.Series(rf_importances, index=features)
xgb_importance_series = pd.Series(xgb_importances, index=features)

print(f"Random Forest (Importance): \n{rf_importance_series}\n")
print(f"XGBoost (Importance): \n{xgb_importance_series}\n")

rf_importance_series.sort_values().plot(kind='barh')

plt.title('Feature Importances (Random Forest Model)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Importance Score', fontsize=12, labelpad=10)
plt.ylabel('Features', fontsize=12, labelpad=10)

xgb_importance_series.sort_values().plot(kind='barh')

#Visualisasi
plt.figure()
plt.title('Actual vs Predicted (Random Forest Model)', fontsize=14, fontweight='bold', pad=15)
plt.scatter(y_test, RandomForest_pred, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

plt.title('Actual vs Predicted (XGBoost Model)', fontsize=14, fontweight='bold', pad=15)
plt.scatter(y_test, XGBoost_pred, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

joblib.dump(RandomForest, "RandomForest_model.pkl")
print("Model disimpan k eRandomForesn_model.pkl")


joblib.dump(features, "feature_names.pkl")