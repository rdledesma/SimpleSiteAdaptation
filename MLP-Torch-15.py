import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Sites import Site
import Metrics as ms
from sklearn.model_selection import ParameterGrid
import os
import sys

# ==============================
# Funciones auxiliares
# ==============================

def build_param_grid():
    """Crea el grid de hiperparámetros según la tabla proporcionada."""
    param_grid = {
        "hidden_layers": [1, 2, 3],
        "hidden_nodes_exp": [1, 2, 3, 4],  # Se transformará a 2^x
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "learning_rate_exp": [-3, -2, -1]  # Se transformará a 10^x
    }
    return list(ParameterGrid(param_grid))


# ==============================
# Definición del modelo MLP en PyTorch
# ==============================
class MLPRegressorTorch(nn.Module):
    def __init__(self, hidden_layers, hidden_nodes_exp, dropout):
        super(MLPRegressorTorch, self).__init__()
        input_size = 1  # solo 1 feature (CAMS o LSA-SAF)
        hidden_size = 2 ** hidden_nodes_exp

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(hidden_layers - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))  # capa de salida
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==============================
# Entrenamiento del modelo
# ==============================
def train_model(X_train, y_train, X_val, y_val, params, epochs=100, batch_size=32):
    """
    Entrena el modelo en GPU (si está disponible) y devuelve el modelo entrenado y RMSE en validación.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convertir datos a tensores y mover a GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # Crear dataset y dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Crear modelo
    model = MLPRegressorTorch(
        hidden_layers=params['hidden_layers'],
        hidden_nodes_exp=params['hidden_nodes_exp'],
        dropout=params['dropout']
    ).to(device)

    # Optimizador y función de pérdida
    learning_rate = 10 ** params['learning_rate_exp']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluación en validación
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)
        rmse = torch.sqrt(loss_fn(y_pred_val, y_val)).item()

    return model, rmse


# ==============================
# Predicción
# ==============================
def predict(model, X):
    """
    Genera predicciones usando el modelo entrenado (en GPU si está disponible).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
    return y_pred


# ==============================
# Main
# ==============================
codigos = sys.argv[1:]  # lista con los códigos pasados por consola

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python script.py YU SA)")
    sys.exit(1)

for cod in codigos:
    site = Site(cod)

    # ==============================
    # Cargar datos
    # ==============================
    df = pd.read_csv(f"{site.cod.lower()}15.csv")
    df['datetime'] = pd.to_datetime(df.datetime)

    dfTrain = pd.read_csv(f'{site.cod}_Train_15.csv')
    dfVal = pd.read_csv(f'{site.cod}_Val_15.csv')
    dfTest = pd.read_csv(f'{site.cod}_Test_15.csv')

    # ==============================
    # Preparar features
    # ==============================
    X_train_cams = dfTrain[['cams']].values
    X_val_cams = dfVal[['cams']].values
    X_test_cams = dfTest[['cams']].values

    X_train_lsasaf = dfTrain[['lsasaf']].values
    X_val_lsasaf = dfVal[['lsasaf']].values
    X_test_lsasaf = dfTest[['lsasaf']].values

    y_train = dfTrain['ghi'].values
    y_val = dfVal['ghi'].values
    y_test = dfTest['ghi'].values

    # ==============================
    # Grid Search para CAMS
    # ==============================
    best_rmse_cams = float('inf')
    best_model_cams = None

    print(f"Buscando mejor MLP (GPU) para CAMS en {site.cod}...")
    for params in build_param_grid():
        model, rmse_val = train_model(X_train_cams, y_train, X_val_cams, y_val, params)
        if rmse_val < best_rmse_cams:
            best_rmse_cams = rmse_val
            best_model_cams = model

    print(f"Mejor RMSE CAMS (val): {best_rmse_cams:.4f}")

    # ==============================
    # Grid Search para LSA-SAF
    # ==============================
    best_rmse_lsasaf = float('inf')
    best_model_lsasaf = None

    print(f"Buscando mejor MLP (GPU) para LSA-SAF en {site.cod}...")
    for params in build_param_grid():
        model, rmse_val = train_model(X_train_lsasaf, y_train, X_val_lsasaf, y_val, params)
        if rmse_val < best_rmse_lsasaf:
            best_rmse_lsasaf = rmse_val
            best_model_lsasaf = model

    print(f"Mejor RMSE LSA-SAF (val): {best_rmse_lsasaf:.4f}")

    # ==============================
    # Predicciones en Test Set
    # ==============================
    dfTest['AdapCamsMLP'] = predict(best_model_cams, X_test_cams)
    dfTest['AdapLsasafMLP'] = predict(best_model_lsasaf, X_test_lsasaf)

    # ==============================
    # Resultados
    # ==============================
    print(f"\n=== Resultados para sitio {site.cod} ===")
    print(f"CAMS     -> RMBE:  {ms.rmbe(y_test, dfTest.cams):.4f}   | Adap: {ms.rmbe(y_test, dfTest.AdapCamsMLP):.4f}")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_test, dfTest.lsasaf):.4f} | Adap: {ms.rmbe(y_test, dfTest.AdapLsasafMLP):.4f}")
    print(f"CAMS     -> RMAE:  {ms.rmae(y_test, dfTest.cams):.4f}   | Adap: {ms.rmae(y_test, dfTest.AdapCamsMLP):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_test, dfTest.lsasaf):.4f} | Adap: {ms.rmae(y_test, dfTest.AdapLsasafMLP):.4f}")
    print(f"CAMS     -> RRMSE: {ms.rrmsd(y_test, dfTest.cams):.4f}  | Adap: {ms.rrmsd(y_test, dfTest.AdapCamsMLP):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_test, dfTest.lsasaf):.4f} | Adap: {ms.rrmsd(y_test, dfTest.AdapLsasafMLP):.4f}")
    print()
