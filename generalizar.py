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
import joblib   # 👈 añadido

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
        "n_estimators": list(range(1, 40, 10)),             # 1, 11, 21, 31, 41
        "max_depth": [2 ** x for x in range(4, 20)],         # 4, 8, 16, 32
        "learning_rate": [10 ** x for x in [-2, -1]]    # 0.001, 0.01, 0.1
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
    Prepara matrices numpy para entrenamiento/validación/test.
    """
    X_train_cams = df_train[["cams",'lat','lon','alt']].values.astype(np.float32)
    X_train_lsasaf = df_train[["lsasaf", 'lat','lon','alt']].values.astype(np.float32)
    y_train = df_train["ghi"].values.astype(np.float32)

    X_val_cams = df_val[["cams",'lat','lon','alt']].values.astype(np.float32)
    X_val_lsasaf = df_val[["lsasaf",'lat','lon','alt']].values.astype(np.float32)
    y_val = df_val["ghi"].values.astype(np.float32)

    X_test_cams = df_test[["cams",'lat','lon','alt']].values.astype(np.float32)
    X_test_lsasaf = df_test[["lsasaf",'lat','lon','alt']].values.astype(np.float32)
    y_test = df_test["ghi"].values.astype(np.float32)

    return (X_train_cams, X_val_cams, X_test_cams,
            X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
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
    df_train = pd.read_csv(f"{site.cod}_Train_15.csv")
    df_val = pd.read_csv(f"{site.cod}_Val_15.csv")
    df_test = pd.read_csv(f"{site.cod}_Test_15_SLRMLP.csv")

    df_train['lat'] = site.lat
    df_train['lon'] = site.long
    df_train['alt'] = site.alt


    df_val['lat'] = site.lat
    df_val['lon'] = site.long
    df_val['alt'] = site.alt


    df_test['lat'] = site.lat
    df_test['lon'] = site.long
    df_test['alt'] = site.alt

    # Preparar arrays
    (X_train_cams, X_val_cams, X_test_cams,
     X_train_lsasaf, X_val_lsasaf, X_test_lsasaf,
     y_train, y_val, y_test) = prepare_arrays(df_train, df_val, df_test)

    # ==============================
    # Grid Search para CAMS
    # ==============================
    best_rmse_cams = float("inf")
    best_model_cams = None
    print(f"🔎 Buscando mejor XGBoost (gbtree) para CAMS en {site.cod}...")

    for params in grid:
        model, rmse_val = fit_and_eval_xgb(X_train_cams, y_train, X_val_cams, y_val, params, use_gpu=USE_GPU)
        if rmse_val < best_rmse_cams:
            best_rmse_cams = rmse_val
            best_model_cams = model

    print(f"✅ Mejor RMSE CAMS (val): {best_rmse_cams:.4f}")
    joblib.dump(best_model_cams, f"{site.cod}_BestModel_CAMS_XGB.pkl")   # 👈 guardar modelo

    # ==============================
    # Grid Search para LSA-SAF
    # ==============================
    best_rmse_lsasaf = float("inf")
    best_model_lsasaf = None
    print(f"🔎 Buscando mejor XGBoost (gbtree) para LSA-SAF en {site.cod}...")

    for params in grid:
        model, rmse_val = fit_and_eval_xgb(X_train_lsasaf, y_train, X_val_lsasaf, y_val, params, use_gpu=USE_GPU)
        if rmse_val < best_rmse_lsasaf:
            best_rmse_lsasaf = rmse_val
            best_model_lsasaf = model

    print(f"✅ Mejor RMSE LSA-SAF (val): {best_rmse_lsasaf:.4f}")
    joblib.dump(best_model_lsasaf, f"{site.cod}_BestModel_LSASAF_XGB.pkl")   # 👈 guardar modelo

    # ==============================
    # Predicciones
    # ==============================
    df_test["AdapCamsXGB"] = best_model_cams.predict(X_test_cams)
    df_test["AdapLsasafXGB"] = best_model_lsasaf.predict(X_test_lsasaf)

    # ==============================
    # Resultados
    # ==============================
    print(f"\n=== Resultados para sitio {site.cod} ===")
    print(f"CAMS     -> RMBE:  {ms.rmbe(y_test, df_test.cams):.1f}   | Adap: {ms.rmbe(y_test, df_test.AdapCamsXGB):.1f}")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_test, df_test.lsasaf):.1f} | Adap: {ms.rmbe(y_test, df_test.AdapLsasafXGB):.1f}")
    print(f"CAMS     -> RMAE:  {ms.rmae(y_test, df_test.cams):.1f}   | Adap: {ms.rmae(y_test, df_test.AdapCamsXGB):.1f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_test, df_test.lsasaf):.1f} | Adap: {ms.rmae(y_test, df_test.AdapLsasafXGB):.1f}")
    print(f"CAMS     -> RRMSE: {ms.rrmsd(y_test, df_test.cams):.1f}  | Adap: {ms.rrmsd(y_test, df_test.AdapCamsXGB):.1f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_test, df_test.lsasaf):.1f} | Adap: {ms.rrmsd(y_test, df_test.AdapLsasafXGB):.1f}")
    print()

    # Guardar test con predicciones
    df_test.to_csv(f"{site.cod}_Test_15_SLRMLPXGB.csv", index=False)
