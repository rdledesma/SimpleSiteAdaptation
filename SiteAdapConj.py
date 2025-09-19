import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from sklearn.linear_model import LinearRegression
import sys
import joblib  # 🔹 para guardar/cargar el modelo
from Geo import Geo
# ==============================
# Main
# ==============================

# Leer los códigos desde la consola
codigos = sys.argv[1:]  # lista con los códigos pasados por consola

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python script.py YU SA)")
    sys.exit(1)

# ==============================
# Construir datasets globales
# ==============================
train_list = []
test_list = []

for cod in codigos:
    site = Site(cod)

    # Train
    dfTrain = pd.read_csv(f"{site.cod}_Train_15.csv")
    dfTrain["lat"] = site.lat
    dfTrain["lon"] = site.long
    dfTrain["alt"] = site.alt
    dfTrain["site"] = site.cod
    dfTrain['N'] = pd.to_datetime(dfTrain['datetime']).dt.day_of_year
    
    train_list.append(dfTrain)

    # Test
    dfTest = pd.read_csv(f"{site.cod}_Test_15.csv")
    dfTest["lat"] = site.lat
    dfTest["lon"] = site.long
    dfTest["alt"] = site.alt
    dfTest["site"] = site.cod
    dfTest['N'] = pd.to_datetime(dfTest['datetime']).dt.day_of_year
    test_list.append(dfTest)

# Concatenar en un único DataFrame
dfTrain_all = pd.concat(train_list, ignore_index=True)
dfTest_all = pd.concat(test_list, ignore_index=True)

# ==============================
# Entrenamiento del modelo MLR
# ==============================
features = ["lsasaf", "lat", "lon", "alt","N"]
X_train = dfTrain_all[features].values
y_train = dfTrain_all["ghi"].values


X_test = dfTest_all[features].values
y_test = dfTest_all["ghi"].values

print("Entrenando modelo de Regresión Lineal Múltiple (MLR)...")
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# 🔹 Guardar modelo entrenado
joblib.dump(mlr_model, "mlr_model.pkl")
print("✅ Modelo guardado como 'mlr_model.pkl'")

# ==============================
# Predicciones
# ==============================
dfTest_all["AdapLsasafMLR"] = mlr_model.predict(X_test)

# ==============================
# Métricas globales
# ==============================
print("\n=== Resultados Globales (todos los sitios) ===")
print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_test, dfTest_all.lsasaf):.4f} | Adap: {ms.rmbe(y_test, dfTest_all.AdapLsasafMLR):.4f}")
print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_test, dfTest_all.lsasaf):.4f} | Adap: {ms.rmae(y_test, dfTest_all.AdapLsasafMLR):.4f}")
print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_test, dfTest_all.lsasaf):.4f} | Adap: {ms.rrmsd(y_test, dfTest_all.AdapLsasafMLR):.4f}")

# ==============================
# Métricas por sitio
# ==============================
print("\n=== Resultados por sitio ===")
for cod in codigos:
    df_site = dfTest_all[dfTest_all["site"] == cod]
    y_true = df_site["ghi"].values
    print(f"\nSitio {cod}:")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_true, df_site.lsasaf):.4f} | Adap: {ms.rmbe(y_true, df_site.AdapLsasafMLR):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_true, df_site.lsasaf):.4f} | Adap: {ms.rmae(y_true, df_site.AdapLsasafMLR):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_true, df_site.lsasaf):.4f} | Adap: {ms.rrmsd(y_true, df_site.AdapLsasafMLR):.4f}")
