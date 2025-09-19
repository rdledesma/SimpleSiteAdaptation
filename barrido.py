import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from itertools import combinations
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

SITES = ["YU", "SA", "SCA", "ERO", "LQ"]
FEATURES = ["lsasaf", "lat", "lon", "alt", "N"]

def maybe_use_gpu():
    return os.environ.get("XGB_USE_GPU", "0") == "1"

def fit_xgb(X_train, y_train, use_gpu=False):
    tree_method = "gpu_hist" if use_gpu else "hist"
    model = XGBRegressor(
        booster="gbtree",
        n_estimators=50,
        max_depth=8,
        learning_rate=0.01,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method=tree_method,
        n_jobs=os.cpu_count()
    )
    model.fit(X_train, y_train)
    return model

def prepare_df(sites_list):
    dfs = []
    for cod in sites_list:
        site = Site(cod)
        df = pd.read_csv(f"/home/inenco/Documentos/01_SiteAdaptation/{cod.lower()}15.csv")
        df["lat"] = site.lat
        df["lon"] = site.long
        df["alt"] = site.alt
        df["site"] = cod
        df["N"] = pd.to_datetime(df["datetime"]).dt.day_of_year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

USE_GPU = maybe_use_gpu()
print(f"✅ XGBoost tree_method = {'gpu_hist' if USE_GPU else 'hist'}")

# =============================
# Guardar métricas para graficar
# =============================
metrics_list = []

for n_train in range(1, len(SITES)):
    train_combos = list(combinations(SITES, n_train))
    
    for train_sites in train_combos:
        test_sites = [s for s in SITES if s not in train_sites]

        df_train = prepare_df(train_sites)
        df_test = prepare_df(test_sites)

        X_train = df_train[FEATURES].values.astype(float)
        y_train = df_train["ghi"].values
        X_test = df_test[FEATURES].values.astype(float)
        y_test = df_test["ghi"].values

        # Entrenar modelo
        model = fit_xgb(X_train, y_train, USE_GPU)

        # Predecir
        df_test["AdapLsasafXGB"] = model.predict(X_test)

        # Métricas globales
        rmse_glob = ms.rrmsd(y_test, df_test.AdapLsasafXGB)
        metrics_list.append({
            "train_sites": "+".join(train_sites),
            "test_sites": "+".join(test_sites),
            "site": "Global",
            "RMSE": rmse_glob
        })

        # Métricas por sitio
        for s in test_sites:
            df_s = df_test[df_test["site"] == s]
            rmse_site = ms.rrmsd(df_s["ghi"].values, df_s["AdapLsasafXGB"].values)
            metrics_list.append({
                "train_sites": "+".join(train_sites),
                "test_sites": "+".join(test_sites),
                "site": s,
                "RMSE": rmse_site
            })

# Convertir a DataFrame
df_metrics = pd.DataFrame(metrics_list)

# =============================
# Graficar RMSE
# =============================
plt.figure(figsize=(12,6))
sns.barplot(data=df_metrics, x="train_sites", y="RMSE", hue="site")
plt.title("RMSE por sitio según combinación de entrenamiento")
plt.ylabel("RMSE")
plt.xlabel("Sitios de entrenamiento")
plt.xticks(rotation=45)
plt.legend(title="Sitio de prueba")
plt.tight_layout()
plt.show()
