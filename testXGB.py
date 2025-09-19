import sys
import joblib
import pandas as pd
from Sites import Site
import Metrics as ms

# Verificar que se pasaron sitios como argumentos
if len(sys.argv) < 2:
    print("Uso: python script.py <SITIO1> <SITIO2> ...")
    sys.exit(1)

# Lista de sitios desde consola
site_codes = sys.argv[1:]

# Cargar el modelo XGBoost previamente entrenado
xgb_model = joblib.load("xgb_lsasaf_model.pkl")

for code in site_codes:
    site = Site(code)

    # Ajusta el path según cómo se llamen tus archivos CSV
    csv_path = f"/home/inenco/Documentos/01_SiteAdaptation/{code.lower()}15.csv"
    df_new = pd.read_csv(csv_path)

    # Agregar coordenadas
    df_new['lat'] = site.lat
    df_new['lon'] = site.long
    df_new['alt'] = site.alt

    # Selección de features
    features = ["lsasaf", "lat", "lon", "alt", "N"]
    
    # Asegurarse de que la columna 'N' exista
    if 'N' not in df_new.columns:
        df_new['N'] = pd.to_datetime(df_new['datetime']).dt.day_of_year

    X_new = df_new[features].values.astype(float)

    # Predecir GHI adaptado con XGBoost
    df_new["AdapLsasafXGB"] = xgb_model.predict(X_new)

    y_true = df_new.ghi.values

    # Resultados
    print(f"\nSitio {site.cod}:")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_true, df_new.lsasaf):.4f} | Adap: {ms.rmbe(y_true, df_new.AdapLsasafXGB):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_true, df_new.lsasaf):.4f} | Adap: {ms.rmae(y_true, df_new.AdapLsasafXGB):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_true, df_new.lsasaf):.4f} | Adap: {ms.rrmsd(y_true, df_new.AdapLsasafXGB):.4f}")
