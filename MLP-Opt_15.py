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
    # ✅ X_train y y_train en CPU aquí
    # ✅ DataLoader usará pin_memory=True para transferir lotes rápido a GPU
    train_dataset = TensorDataset(X_train.cpu(), y_train.cpu())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # X_val y y_val a GPU directamente porque no cambian
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Modelo
    model = MLPRegressorTorch(
        hidden_layers=params['hidden_layers'],
        hidden_nodes_exp=params['hidden_nodes_exp'],
        dropout=params['dropout']
    ).to(device)

    # Optimización y compilación
    model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=10 ** params['learning_rate_exp'])
    loss_fn = nn.MSELoss()

    # ✅ Cambiamos a la nueva forma recomendada
    scaler = torch.amp.GradScaler('cuda')

    best_rmse = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # ✅ Ahora movemos cada batch a GPU dentro del loop
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
# Preparar tensores GPU una sola vez
# ==============================
def prepare_tensors(df_train, df_val, df_test):
    # ✅ CPU para entrenamiento
    X_train_cams = torch.tensor(df_train[['cams']].values, dtype=torch.float32)  # CPU
    X_train_lsasaf = torch.tensor(df_train[['lsasaf']].values, dtype=torch.float32)  # CPU
    y_train = torch.tensor(df_train['ghi'].values, dtype=torch.float32).view(-1, 1)  # CPU

    # ✅ GPU para validación y test
    X_val_cams = torch.tensor(df_val[['cams']].values, dtype=torch.float32, device=device)
    X_val_lsasaf = torch.tensor(df_val[['lsasaf']].values, dtype=torch.float32, device=device)
    y_val = torch.tensor(df_val['ghi'].values, dtype=torch.float32, device=device).view(-1, 1)

    X_test_cams = torch.tensor(df_test[['cams']].values, dtype=torch.float32, device=device)
    X_test_lsasaf = torch.tensor(df_test[['lsasaf']].values, dtype=torch.float32, device=device)
    y_test = df_test['ghi'].values  # numpy para métricas

    return (X_train_cams, X_val_cams, X_test_cams,
            X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
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
    
    df_train = pd.read_csv(f'{site.cod}_Train_15.csv')
    df_val = pd.read_csv(f'{site.cod}_Val_15.csv')
    df_test = pd.read_csv(f'{site.cod}_Test_15.csv')

    # Preparar tensores
    (X_train_cams, X_val_cams, X_test_cams,
     X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
     y_train, y_val, y_test) = prepare_tensors(df_train, df_val, df_test)

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
    # Predicciones
    # ==============================
    df_test['AdapCamsMLP'] = predict(best_model_cams, X_test_cams)
    df_test['AdapLsasafMLP'] = predict(best_model_lsasaf, X_test_lsasaf)

    # ==============================
    # Resultados
    # ==============================
    print(f"\n=== Resultados para sitio {site.cod} ===")
    print(f"CAMS     -> RMBE:  {ms.rmbe(y_test, df_test.cams):.4f}   | Adap: {ms.rmbe(y_test, df_test.AdapCamsMLP):.4f}")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_test, df_test.lsasaf):.4f} | Adap: {ms.rmbe(y_test, df_test.AdapLsasafMLP):.4f}")
    print(f"CAMS     -> RMAE:  {ms.rmae(y_test, df_test.cams):.4f}   | Adap: {ms.rmae(y_test, df_test.AdapCamsMLP):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_test, df_test.lsasaf):.4f} | Adap: {ms.rmae(y_test, df_test.AdapLsasafMLP):.4f}")
    print(f"CAMS     -> RRMSE: {ms.rrmsd(y_test, df_test.cams):.4f}  | Adap: {ms.rrmsd(y_test, df_test.AdapCamsMLP):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_test, df_test.lsasaf):.4f} | Adap: {ms.rrmsd(y_test, df_test.AdapLsasafMLP):.4f}")
    print()

    df_test.to_csv(f'{site.cod}_Test_15_SLRMLP.csv')