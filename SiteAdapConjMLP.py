import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from sklearn.model_selection import ParameterGrid
from itertools import combinations
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

SITES = ["YU", "SA", "SCA", "ERO", "LQ"]
FEATURES = ["lsasaf", "lat", "lon", "alt", "N"]

# ==============================
# Definición del MLP en PyTorch
# ==============================
class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_nodes_exp, dropout):
        super(MLPRegressorTorch, self).__init__()
        layers = []
        hidden_dim = 2 ** hidden_nodes_exp  # como en XGB usamos potencias de 2

        # capa de entrada
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # capas ocultas
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # capa de salida (regresión)
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


# ==============================
# Grilla de hiperparámetros
# ==============================
def build_param_grid():
    """Grid de hiperparámetros para el MLP."""
    param_grid = {
        "hidden_layers": [1, 2, 3, 4],
        "hidden_nodes_exp": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate_exp": [-1, -2]  # 0.1 y 0.01
    }
    return list(ParameterGrid(param_grid))


# ==============================
# Entrenamiento y evaluación
# ==============================
def fit_and_eval_mlp(X_train, y_train, X_val, y_val, params, epochs=50, batch_size=256):
    input_dim = X_train.shape[1]
    model = MLPRegressorTorch(
        input_dim=input_dim,
        hidden_layers=params["hidden_layers"],
        hidden_nodes_exp=params["hidden_nodes_exp"],
        dropout=params["dropout"]
    )

    lr = 10 ** params["learning_rate_exp"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Entrenamiento mini-batch
    n = len(X_train_t)
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluación
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_t).numpy()
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    return model, rmse


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
param_grid = build_param_grid()
metrics_list = []

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

        # Grid search MLP
        best_rmse = float("inf")
        best_model = None
        for params in param_grid:
            model, rmse_val = fit_and_eval_mlp(X_train, y_train, X_val, y_val, params)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_model = model

        # Predicciones
        best_model.eval()
        with torch.no_grad():
            df_test["AdapLsasafMLP"] = best_model(torch.tensor(X_val, dtype=torch.float32)).numpy()

        # ==========
        # Métricas globales
        # ==========
        rmse_mlp = ms.rrmsd(y_val, df_test.AdapLsasafMLP)
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
            "model": "MLP Adaptado",
            "RMSE": rmse_mlp
        })

        # ==========
        # Métricas por sitio
        # ==========
        for s in test_sites:
            df_s = df_test[df_test["site"] == s]
            y_true = df_s["ghi"].values
            rmse_site_mlp = ms.rrmsd(y_true, df_s["AdapLsasafMLP"].values)
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
                "model": "MLP Adaptado",
                "RMSE": rmse_site_mlp
            })

# =============================
# Gráficos
# =============================
df_metrics = pd.DataFrame(metrics_list)

df_plot = df_metrics.melt(
    id_vars=["train_sites", "test_sites", "site", "model"],
    value_vars=["RMSE"],
    var_name="Métrica",
    value_name="Valor"
)

df_plot["combo_site"] = df_plot["train_sites"] + " | " + df_plot["site"]
df_plot["site_order"] = df_plot["site"].apply(lambda x: 0 if x == "Global" else 1)
df_plot = df_plot.sort_values(by=["train_sites", "site_order", "site", "Métrica"])

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
