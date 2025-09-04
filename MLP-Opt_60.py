import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Sites import Site
import Metrics as ms
from sklearn.model_selection import ParameterGrid
import sys
import os

# ==============================
# Configuración global
# ==============================
torch.backends.cudnn.benchmark = True  # optimiza kernels para la GPU
torch.set_float32_matmul_precision('high')  # Activa TensorFloat32 en RTX 30xx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")

# ==============================
# Funciones auxiliares
# ==============================
def build_param_grid():
    """Crea el grid de hiperparámetros según la tabla proporcionada."""
    param_grid = {
        "hidden_layers": [1, 2, 3],
        "hidden_nodes_exp": [1, 2],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate_exp": [-1, -2]
    }
    return list(ParameterGrid(param_grid))

# ==============================
# Definición del modelo
# ==============================
class MLPRegressorTorch(nn.Module):
    def __init__(self, hidden_layers, hidden_nodes_exp, dropout):
        super().__init__()
        input_size = 1
        hidden_size = 2 ** hidden_nodes_exp

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ==============================
# Entrenamiento con Early Stopping y Mixed Precision
# ==============================
def train_model(X_train, y_train, X_val, y_val, params, epochs=100, batch_size=32, patience=10):
    """
    Entrena el modelo en GPU con mixed precision y early stopping.
    """
    # Dataset y DataLoader para entrenamiento
    train_dataset = TensorDataset(X_train.cpu(), y_train.cpu())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Validación en GPU
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Modelo
    model = MLPRegressorTorch(
        hidden_layers=params['hidden_layers'],
        hidden_nodes_exp=params['hidden_nodes_exp'],
        dropout=params['dropout']
    ).to(device)

    # Optimización
    model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=10 ** params['learning_rate_exp'])
    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler('cuda')

    best_rmse = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # Movemos a GPU por batch
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validación
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            rmse = torch.sqrt(loss_fn(y_pred_val, y_val)).item()

        # Early stopping
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Cargar el mejor modelo
    model.load_state_dict(best_model_state)
    return model, best_rmse

# ==============================
# Predicción
# ==============================
@torch.inference_mode()
def predict(model, X):
    y_pred = model(X).cpu().numpy().flatten()
    return y_pred

# ==============================
# Preparar tensores
# ==============================
def prepare_tensors(df_train, df_val, df_test):
    """
    Prepara tensores para todas las fuentes: CAMS, LSA-SAF, ERA y MERRA.
    """
    # CPU para entrenamiento
    X_train_cams = torch.tensor(df_train[['cams']].values, dtype=torch.float32)
    X_train_lsasaf = torch.tensor(df_train[['lsasaf']].values, dtype=torch.float32)
    X_train_era = torch.tensor(df_train[['era']].values, dtype=torch.float32)
    X_train_merra = torch.tensor(df_train[['merra']].values, dtype=torch.float32)
    y_train = torch.tensor(df_train['ghi'].values, dtype=torch.float32).view(-1, 1)

    # GPU para validación
    X_val_cams = torch.tensor(df_val[['cams']].values, dtype=torch.float32, device=device)
    X_val_lsasaf = torch.tensor(df_val[['lsasaf']].values, dtype=torch.float32, device=device)
    X_val_era = torch.tensor(df_val[['era']].values, dtype=torch.float32, device=device)
    X_val_merra = torch.tensor(df_val[['merra']].values, dtype=torch.float32, device=device)
    y_val = torch.tensor(df_val['ghi'].values, dtype=torch.float32, device=device).view(-1, 1)

    # GPU para test
    X_test_cams = torch.tensor(df_test[['cams']].values, dtype=torch.float32, device=device)
    X_test_lsasaf = torch.tensor(df_test[['lsasaf']].values, dtype=torch.float32, device=device)
    X_test_era = torch.tensor(df_test[['era']].values, dtype=torch.float32, device=device)
    X_test_merra = torch.tensor(df_test[['merra']].values, dtype=torch.float32, device=device)
    y_test = df_test['ghi'].values  # numpy para métricas

    return (X_train_cams, X_val_cams, X_test_cams,
            X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
            X_train_era, X_val_era, X_test_era,
            X_train_merra, X_val_merra, X_test_merra,
            y_train, y_val, y_test)

# ==============================
# Main
# ==============================
codigos = sys.argv[1:]

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python script.py YU SA)")
    sys.exit(1)

for cod in codigos:
    site = Site(cod)

    # ==============================
    # Cargar datos
    # ==============================
    df_train = pd.read_csv(f'{site.cod}_Train_60.csv')
    df_val = pd.read_csv(f'{site.cod}_Val_60.csv')
    df_test = pd.read_csv(f'{site.cod}_Test_60.csv')

    # Preparar tensores
    (X_train_cams, X_val_cams, X_test_cams,
     X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
     X_train_era, X_val_era, X_test_era,
     X_train_merra, X_val_merra, X_test_merra,
     y_train, y_val, y_test) = prepare_tensors(df_train, df_val, df_test)

    # ==============================
    # Grid Search para cada fuente
    # ==============================
    def grid_search(X_train, X_val, name):
        best_rmse = float('inf')
        best_model = None
        print(f"Buscando mejor MLP (GPU) para {name} en {site.cod}...")
        for params in build_param_grid():
            model, rmse_val = train_model(X_train, y_train, X_val, y_val, params)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_model = model
        print(f"Mejor RMSE {name} (val): {best_rmse:.4f}")
        return best_model

    best_model_cams = grid_search(X_train_cams, X_val_cams, "CAMS")
    best_model_lsasaf = grid_search(X_train_lsasaf, X_val_lsasaf, "LSA-SAF")
    best_model_era = grid_search(X_train_era, X_val_era, "ERA-5")
    best_model_merra = grid_search(X_train_merra, X_val_merra, "MERRA-2")

    # ==============================
    # Predicciones
    # ==============================
    df_test['AdapCamsMLP'] = predict(best_model_cams, X_test_cams)
    df_test['AdapLsasafMLP'] = predict(best_model_lsasaf, X_test_lsasaf)
    df_test['AdapEraMLP'] = predict(best_model_era, X_test_era)
    df_test['AdapMerraMLP'] = predict(best_model_merra, X_test_merra)

    # ==============================
    # Resultados
    # ==============================
    print(f"\n=== Resultados para sitio {site.cod} ===")
    fuentes = [
        ("CAMS", "cams", "AdapCamsMLP"),
        ("LSA-SAF", "lsasaf", "AdapLsasafMLP"),
        ("ERA-5", "era", "AdapEraMLP"),
        ("MERRA-2", "merra", "AdapMerraMLP")
    ]

    for nombre, col_original, col_adapt in fuentes:
        print(f"{nombre:8} -> RMBE:  {ms.rmbe(y_test, df_test[col_original]):.4f} | Adap: {ms.rmbe(y_test, df_test[col_adapt]):.4f}")
    print()
    for nombre, col_original, col_adapt in fuentes:
        print(f"{nombre:8} -> RMAE:  {ms.rmae(y_test, df_test[col_original]):.4f} | Adap: {ms.rmae(y_test, df_test[col_adapt]):.4f}")
    print()
    for nombre, col_original, col_adapt in fuentes:
        print(f"{nombre:8} -> RRMSE: {ms.rrmsd(y_test, df_test[col_original]):.4f} | Adap: {ms.rrmsd(y_test, df_test[col_adapt]):.4f}")
    print()

    # Guardar resultados
    df_test.to_csv(f'{site.cod}_Test_60_SLRMLP.csv', index=False)
