import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from itertools import combinations
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

SITES = ["YU", "SA", "SCA", "ERO", "LQ"]
FEATURES = ["lsasaf", "lat", "lon", "alt", "N"]

# ==============================
# Funciones XGBoost
# ==============================
def build_param_grid():
    """Grid de hiperparámetros XGBoost."""
    param_grid = {
        "n_estimators": list(range(1, 51, 10)),
        "max_depth": [2 ** x for x in range(2, 6)],
        "learning_rate": [10 ** x for x in [-3, -2, -1]]
    }
    return list(ParameterGrid(param_grid))


def fit_and_eval_xgb(X_train, y_train, X_val, y_val, params, use_gpu=False):
    tree_method = "gpu_hist" if use_gpu else "hist"
    model = XGBRegressor(
        booster="gbtree",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method=tree_method,
        n_jobs=os.cpu_count()
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return model, rmse


def maybe_use_gpu():
    return os.environ.get("XGB_USE_GPU", "0") == "1"


# ==============================
# Preparación de datos
# ==============================
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


# ==============================
# Main
# ==============================
USE_GPU = maybe_use_gpu()
print(f"✅ XGBoost tree_method = {'gpu_hist' if USE_GPU else 'hist'}")

param_grid = build_param_grid()
metrics_list = []

# Combinaciones de entrenamiento
for n_train in range(1, len(SITES)):
    train_combos = list(combinations(SITES, n_train))
    
    for train_sites in train_combos:
        test_sites = [s for s in SITES if s not in train_sites]

        df_train = prepare_df(train_sites)
        df_test = prepare_df(test_sites)

        X_train = df_train[FEATURES].values.astype(float)
        y_train = df_train["ghi"].values
        X_val = df_test[FEATURES].values.astype(float)
        y_val = df_test["ghi"].values

        # Grid search XGBoost
        best_rmse = float("inf")
        best_model = None
        for params in param_grid:
            model, rmse_val = fit_and_eval_xgb(X_train, y_train, X_val, y_val, params, USE_GPU)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_model = model

        # Predicciones
        df_test["AdapLsasafXGB"] = best_model.predict(X_val)

        # ==========
        # Métricas globales
        # ==========
        rmse_xgb = ms.rrmsd(y_val, df_test.AdapLsasafXGB)
        rmse_lsasaf = ms.rrmsd(y_val, df_test.lsasaf)

        metrics_list.append({
            "train_sites": "+".join(train_sites),
            "test_sites": "+".join(test_sites),
            "site": "Global",
            "model": "LSASAF",
            "RMSE": rmse_lsasaf
        })
        metrics_list.append({
            "train_sites": "+".join(train_sites),
            "test_sites": "+".join(test_sites),
            "site": "Global",
            "model": "XGB Adaptado",
            "RMSE": rmse_xgb
        })

        # ==========
        # Métricas por sitio
        # ==========
        for s in test_sites:
            df_s = df_test[df_test["site"] == s]
            y_true = df_s["ghi"].values
            rmse_site_xgb = ms.rrmsd(y_true, df_s["AdapLsasafXGB"].values)
            rmse_site_lsasaf = ms.rrmsd(y_true, df_s["lsasaf"].values)

            metrics_list.append({
                "train_sites": "+".join(train_sites),
                "test_sites": "+".join(test_sites),
                "site": s,
                "model": "LSASAF",
                "RMSE": rmse_site_lsasaf
            })
            metrics_list.append({
                "train_sites": "+".join(train_sites),
                "test_sites": "+".join(test_sites),
                "site": s,
                "model": "XGB Adaptado",
                "RMSE": rmse_site_xgb
            })





# =============================
# Gráficos: métricas globales y por sitio
# =============================

# Pasar a formato largo
# =============================
# Convertir a DataFrame
# =============================
df_metrics = pd.DataFrame(metrics_list)

# =============================
# Gráficos: métricas globales y por sitio
# =============================

# Pasar a formato largo (solo RMSE porque es lo que calculamos)
df_plot = df_metrics.melt(
    id_vars=["train_sites", "test_sites", "site", "model"],
    value_vars=["RMSE"],
    var_name="Métrica",
    value_name="Valor"
)

# Crear etiquetas de combinación
df_plot["combo_site"] = df_plot["train_sites"] + " | " + df_plot["site"]

# Ordenar para que Global aparezca primero dentro de cada combinación
df_plot["site_order"] = df_plot["site"].apply(lambda x: 0 if x == "Global" else 1)
df_plot = df_plot.sort_values(by=["train_sites", "site_order", "site", "Métrica"])

# Crear gráfico de barras
g = sns.catplot(
    data=df_plot,
    kind="bar",
    x="combo_site", y="Valor", hue="model",
    col="Métrica", col_wrap=1,
    ci=None, palette="Set2", sharey=False,
    height=5, aspect=1.5
)

g.set_titles("📊 {col_name}")
g.set_axis_labels("Combinación de entrenamiento | Sitio evaluado", "Valor")
for ax in g.axes.flatten():
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()