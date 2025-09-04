import pandas as pd
import numpy as np
from Sites import Site
import Metrics as ms
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# Configuración global
# ==============================
RANDOM_STATE = 42

def build_param_grid():
    """
    Crea el grid de hiperparámetros para XGBoost según la tabla:
      - Booster: gbtree (fijo)
      - Estimators: 1 → 50 con paso 10
      - Max depth: 2^x para x en {2,3,4,5} → {4,8,16,32}
      - Learning rate: 10^x para x en {-3,-2,-1} → {0.001, 0.01, 0.1}
    """
    param_grid = {
        "n_estimators": list(range(1, 51, 10)),             # 1, 11, 21, 31, 41
        "max_depth": [2 ** x for x in range(2, 6)],         # 4, 8, 16, 32
        "learning_rate": [10 ** x for x in [-3, -2, -1]]    # 0.001, 0.01, 0.1
    }
    return list(ParameterGrid(param_grid))


def fit_and_eval_xgb(X_train, y_train, X_val, y_val, params, use_gpu=False):
    """
    Entrena un XGBRegressor con los params dados y evalúa RMSE en validación.
    """
    tree_method = "gpu_hist" if use_gpu else "hist"

    model = XGBRegressor(
        booster="gbtree",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        objective="reg:squarederror",
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method=tree_method,
        n_jobs=os.cpu_count()
    )

    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    return model, rmse


def prepare_arrays(df_train, df_val, df_test):
    """
    Prepara matrices numpy para entrenamiento/validación/test para cada fuente.
    """
    # CAMS
    X_train_cams = df_train[["cams"]].values.astype(np.float32)
    X_val_cams = df_val[["cams"]].values.astype(np.float32)
    X_test_cams = df_test[["cams"]].values.astype(np.float32)

    # LSA-SAF
    X_train_lsasaf = df_train[["lsasaf"]].values.astype(np.float32)
    X_val_lsasaf = df_val[["lsasaf"]].values.astype(np.float32)
    X_test_lsasaf = df_test[["lsasaf"]].values.astype(np.float32)

    # ERA
    X_train_era = df_train[["era"]].values.astype(np.float32)
    X_val_era = df_val[["era"]].values.astype(np.float32)
    X_test_era = df_test[["era"]].values.astype(np.float32)

    # MERRA
    X_train_merra = df_train[["merra"]].values.astype(np.float32)
    X_val_merra = df_val[["merra"]].values.astype(np.float32)
    X_test_merra = df_test[["merra"]].values.astype(np.float32)

    # Target
    y_train = df_train["ghi"].values.astype(np.float32)
    y_val = df_val["ghi"].values.astype(np.float32)
    y_test = df_test["ghi"].values.astype(np.float32)

    return (X_train_cams, X_val_cams, X_test_cams,
            X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
            X_train_era, X_val_era, X_test_era,
            X_train_merra, X_val_merra, X_test_merra,
            y_train, y_val, y_test)


def maybe_use_gpu():
    """
    Heurística simple: usar GPU si el usuario lo habilita por variable de entorno XGB_USE_GPU=1.
    """
    return os.environ.get("XGB_USE_GPU", "0") == "1"


# ==============================
# Main
# ==============================
codigos = sys.argv[1:]

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python script.py YU SA)")
    sys.exit(1)

USE_GPU = maybe_use_gpu()
print(f"✅ XGBoost tree_method = {'gpu_hist' if USE_GPU else 'hist'}")

grid = build_param_grid()

for cod in codigos:
    site = Site(cod)

    # ==============================
    # Cargar datos
    # ==============================
    df_train = pd.read_csv(f"{site.cod}_Train_60.csv")
    df_val = pd.read_csv(f"{site.cod}_Val_60.csv")
    df_test = pd.read_csv(f"{site.cod}_Test_60_SLRMLP.csv")

    # Preparar arrays
    (X_train_cams, X_val_cams, X_test_cams,
     X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
     X_train_era, X_val_era, X_test_era,
     X_train_merra, X_val_merra, X_test_merra,
     y_train, y_val, y_test) = prepare_arrays(df_train, df_val, df_test)

    # ==============================
    # Grid Search para cada fuente
    # ==============================
    def grid_search_source(X_train, X_val, name):
        best_rmse = float("inf")
        best_model = None
        print(f"🔎 Buscando mejor XGBoost (gbtree) para {name} en {site.cod}...")
        for params in grid:
            model, rmse_val = fit_and_eval_xgb(X_train, y_train, X_val, y_val, params, use_gpu=USE_GPU)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_model = model
        print(f"✅ Mejor RMSE {name} (val): {best_rmse:.4f}")
        return best_model

    best_model_cams = grid_search_source(X_train_cams, X_val_cams, "CAMS")
    best_model_lsasaf = grid_search_source(X_train_lsasaf, X_val_lsasaf, "LSA-SAF")
    best_model_era = grid_search_source(X_train_era, X_val_era, "ERA-5")
    best_model_merra = grid_search_source(X_train_merra, X_val_merra, "MERRA-2")

    # ==============================
    # Predicciones
    # ==============================
    df_test["AdapCamsXGB"] = best_model_cams.predict(X_test_cams)
    df_test["AdapLsasafXGB"] = best_model_lsasaf.predict(X_test_lsasaf)
    df_test["AdapEraXGB"] = best_model_era.predict(X_test_era)
    df_test["AdapMerraXGB"] = best_model_merra.predict(X_test_merra)

    # ==============================
    # Resultados
    # ==============================
    print(f"=== Resultados para sitio {site.cod} ===\n")

    print(f"CAMS     -> RMBE:  {ms.rmbe(df_test.ghi, df_test.cams):.4f}   | Adap: {ms.rmbe(df_test.ghi, df_test.AdapCamsXGB):.4f}")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(df_test.ghi, df_test.lsasaf):.4f} | Adap: {ms.rmbe(df_test.ghi, df_test.AdapLsasafXGB):.4f}")
    print(f"ERA-5    -> RMBE:  {ms.rmbe(df_test.ghi, df_test.era):.4f}    | Adap: {ms.rmbe(df_test.ghi, df_test.AdapEraXGB):.4f}")
    print(f"MERRA-2  -> RMBE:  {ms.rmbe(df_test.ghi, df_test.merra):.4f}  | Adap: {ms.rmbe(df_test.ghi, df_test.AdapMerraXGB):.4f}\n")

    print(f"CAMS     -> RMAE:  {ms.rmae(df_test.ghi, df_test.cams):.4f}   | Adap: {ms.rmae(df_test.ghi, df_test.AdapCamsXGB):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(df_test.ghi, df_test.lsasaf):.4f} | Adap: {ms.rmae(df_test.ghi, df_test.AdapLsasafXGB):.4f}")
    print(f"ERA-5    -> RMAE:  {ms.rmae(df_test.ghi, df_test.era):.4f}    | Adap: {ms.rmae(df_test.ghi, df_test.AdapEraXGB):.4f}")
    print(f"MERRA-2  -> RMAE:  {ms.rmae(df_test.ghi, df_test.merra):.4f}  | Adap: {ms.rmae(df_test.ghi, df_test.AdapMerraXGB):.4f}\n")

    print(f"CAMS     -> RRMSE: {ms.rrmsd(df_test.ghi, df_test.cams):.4f}  | Adap: {ms.rrmsd(df_test.ghi, df_test.AdapCamsXGB):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(df_test.ghi, df_test.lsasaf):.4f} | Adap: {ms.rrmsd(df_test.ghi, df_test.AdapLsasafXGB):.4f}")
    print(f"ERA-5    -> RRMSE: {ms.rrmsd(df_test.ghi, df_test.era):.4f}    | Adap: {ms.rrmsd(df_test.ghi, df_test.AdapEraXGB):.4f}")
    print(f"MERRA-2  -> RRMSE: {ms.rrmsd(df_test.ghi, df_test.merra):.4f}  | Adap: {ms.rrmsd(df_test.ghi, df_test.AdapMerraXGB):.4f}\n")

    # Guardar resultados
    df_test.to_csv(f"{site.cod}_Test_60_SLRMLPXGB.csv", index=False)
