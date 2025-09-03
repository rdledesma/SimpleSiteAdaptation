import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
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


def create_mlp(hidden_layers, hidden_nodes_exp, dropout, learning_rate_exp):
    """
    Crea un MLPRegressor con los hiperparámetros ajustados.
    Nota: sklearn no soporta dropout directamente, se ignora en esta versión.
    """
    hidden_layer_sizes = tuple([2 ** hidden_nodes_exp] * hidden_layers)
    learning_rate_init = 10 ** learning_rate_exp

    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate_init,
        max_iter=500,
        random_state=42
    )


def evaluate_model(model, X_val, y_val):
    """Evalúa el modelo usando RMSE."""
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))


# ==============================
# Main
# ==============================

# Leer los códigos desde la consola
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
    # Grid Search para CAMS y LSA-SAF
    # ==============================
    features = ['cams', 'lsasaf']
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
    # Buscar mejor modelo para CAMS
    # ==============================
    best_rmse_cams = float('inf')
    best_model_cams = None

    print(f"Buscando mejor MLP para CAMS en {site.cod}...")
    for params in build_param_grid():
        model = create_mlp(**params)
        model.fit(X_train_cams, y_train)
        rmse_val = evaluate_model(model, X_val_cams, y_val)

        if rmse_val < best_rmse_cams:
            best_rmse_cams = rmse_val
            best_model_cams = model

    print(f"Mejor RMSE CAMS (val): {best_rmse_cams:.4f}")

    # ==============================
    # Buscar mejor modelo para LSA-SAF
    # ==============================
    best_rmse_lsasaf = float('inf')
    best_model_lsasaf = None

    print(f"Buscando mejor MLP para LSA-SAF en {site.cod}...")
    for params in build_param_grid():
        model = create_mlp(**params)
        model.fit(X_train_lsasaf, y_train)
        rmse_val = evaluate_model(model, X_val_lsasaf, y_val)

        if rmse_val < best_rmse_lsasaf:
            best_rmse_lsasaf = rmse_val
            best_model_lsasaf = model

    print(f"Mejor RMSE LSA-SAF (val): {best_rmse_lsasaf:.4f}")

    # ==============================
    # Predicciones en Test Set
    # ==============================
    dfTest['AdapCamsMLP'] = best_model_cams.predict(X_test_cams).flatten()
    dfTest['AdapLsasafMLP'] = best_model_lsasaf.predict(X_test_lsasaf).flatten()

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

    