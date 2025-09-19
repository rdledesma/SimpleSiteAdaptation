import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from sklearn.linear_model import LinearRegression
from itertools import combinations
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

SITES = ["YU", "SA", "SCA", "ERO", "LQ"]
FEATURES = ["lsasaf", "lat", "lon", "alt", "N"]

def fit_mlr(X_train, y_train):
    model = LinearRegression(n_jobs=-1)
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

print("✅ Usando MLR (Regresión Lineal Múltiple)")

# =============================
# Generar todas las combinaciones de entrenamiento
# =============================
metrics_list = []

for n_train in range(1, len(SITES)):
    train_combos = list(combinations(SITES, n_train))
    
    for train_sites in train_combos:
        test_sites = [s for s in SITES if s not in train_sites]

        # Preparar datasets
        df_train = prepare_df(train_sites)
        df_test = prepare_df(test_sites)

        X_train = df_train[FEATURES].values.astype(float)
        y_train = df_train["ghi"].values
        X_test = df_test[FEATURES].values.astype(float)
        y_test = df_test["ghi"].values

        # Entrenar modelo
        model = fit_mlr(X_train, y_train)

        # Predecir
        df_test["AdapLsasafMLR"] = model.predict(X_test)

        # ==========================
        # Métricas globales
        # ==========================
        metrics_list.append({
            "train_sites": "+".join(train_sites),
            "test_sites": "+".join(test_sites),
            "site": "Global",
            "modelo": "LSASAF",
            "RMBE": ms.rmbe(y_test, df_test.lsasaf),
            "RMAE": ms.rmae(y_test, df_test.lsasaf),
            "RRMSE": ms.rrmsd(y_test, df_test.lsasaf)
        })
        metrics_list.append({
            "train_sites": "+".join(train_sites),
            "test_sites": "+".join(test_sites),
            "site": "Global",
            "modelo": "MLR Adaptado",
            "RMBE": ms.rmbe(y_test, df_test.AdapLsasafMLR),
            "RMAE": ms.rmae(y_test, df_test.AdapLsasafMLR),
            "RRMSE": ms.rrmsd(y_test, df_test.AdapLsasafMLR)
        })

        # ==========================
        # Métricas por sitio
        # ==========================
        for s in test_sites:
            df_s = df_test[df_test["site"] == s]
            y_true = df_s["ghi"].values

            metrics_list.append({
                "train_sites": "+".join(train_sites),
                "test_sites": "+".join(test_sites),
                "site": s,
                "modelo": "LSASAF",
                "RMBE": ms.rmbe(y_true, df_s.lsasaf),
                "RMAE": ms.rmae(y_true, df_s.lsasaf),
                "RRMSE": ms.rrmsd(y_true, df_s.lsasaf)
            })
            metrics_list.append({
                "train_sites": "+".join(train_sites),
                "test_sites": "+".join(test_sites),
                "site": s,
                "modelo": "MLR Adaptado",
                "RMBE": ms.rmbe(y_true, df_s.AdapLsasafMLR),
                "RMAE": ms.rmae(y_true, df_s.AdapLsasafMLR),
                "RRMSE": ms.rrmsd(y_true, df_s.AdapLsasafMLR)
            })

# =============================
# Convertir a DataFrame
# =============================
df_metrics = pd.DataFrame(metrics_list)

# =============================
# Mostrar resumen en consola
# =============================
print("\n📊 Métricas Globales (por combinación de entrenamiento):")
print(df_metrics[df_metrics["site"] == "Global"]
      .sort_values(["train_sites", "modelo"])
      .to_string(index=False))

print("\n📊 Métricas por sitio de prueba:")
print(df_metrics[df_metrics["site"] != "Global"]
      .sort_values(["site", "train_sites", "modelo"])
      .to_string(index=False))

# =============================
# Gráficos
# =============================





import matplotlib.pyplot as plt
import seaborn as sns

# Copiar dataframe y crear etiquetas
df_plot = df_metrics.copy()
df_plot["combo_site"] = df_plot["train_sites"] + " | " + df_plot["site"]

# Ordenar para que global aparezca primero dentro de cada combinación
df_plot["site_order"] = df_plot["site"].apply(lambda x: 0 if x=="Global" else 1)

# Crear un orden jerárquico de las barras
df_plot = df_plot.sort_values(by=["train_sites", "site_order", "site"])

# Gráfico vertical
plt.figure(figsize=(16,10))
sns.barplot(
    data=df_plot,
    x="combo_site", y="RRMSE", hue="modelo", ci=None, palette="Set2"
)

plt.title("📊 RRMSE Global y por sitio de prueba en cada combinación de entrenamiento")
plt.ylabel("RRMSE")
plt.xlabel("Combinación de entrenamiento | Sitio evaluado")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Modelo")
plt.tight_layout()
plt.show()




# =============================
# Gráficos: métricas globales y por sitio
# =============================

# Pasar a formato largo
df_plot = df_metrics.melt(
    id_vars=["train_sites", "test_sites", "site", "modelo"],
    value_vars=[ "RRMSE"],
    var_name="Métrica",
    value_name="Valor"
)

# Crear etiquetas de combinación
df_plot["combo_site"] = df_plot["train_sites"] + " | " + df_plot["site"]

# Ordenar para que Global aparezca primero dentro de cada combinación
df_plot["site_order"] = df_plot["site"].apply(lambda x: 0 if x=="Global" else 1)
df_plot = df_plot.sort_values(by=["train_sites", "site_order", "site", "Métrica"])

# Crear gráficos por métrica
g = sns.catplot(
    data=df_plot,
    kind="bar",
    x="combo_site", y="Valor", hue="modelo",
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
