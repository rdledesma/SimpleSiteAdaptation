import pandas as pd
from Sites import Site
import Metrics as ms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import sys

# Leer los códigos desde la consola
# Ejemplo de uso:  python script.py YU SA SCA
codigos = sys.argv[1:]  # lista con los códigos pasados por consola

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python script.py YU SA)")
    sys.exit(1)

for cod in codigos:
    site = Site(cod)

    # Cargar datos
    df = pd.read_csv(f"{site.cod.lower()}15.csv")
    df['datetime'] = pd.to_datetime(df.datetime)

    # Train / Validation / Test
    dfTrain = df[df.datetime.dt.year == df.datetime.dt.year.unique()[0]]
    dfTrain, dfVal = train_test_split(dfTrain, test_size=0.2, random_state=42, shuffle=True)
    dfTest = df[df.datetime.dt.year != df.datetime.dt.year.unique()[0]]

    dfTrain.to_csv(f'{site.cod}_Train_15.csv', index=False)
    dfVal.to_csv(f'{site.cod}_Val_15.csv', index=False)
    

    # Modelos
    regCams = LinearRegression().fit(dfTrain.cams.values.reshape(-1,1), dfTrain.ghi)
    regLsasaf = LinearRegression().fit(dfTrain.lsasaf.values.reshape(-1,1), dfTrain.ghi)

    # Predicciones
    dfTest['AdapCamsSLR'] = regCams.predict(dfTest.cams.values.reshape(-1,1)).flatten()
    dfTest['AdapLsasafSLR'] = regLsasaf.predict(dfTest.lsasaf.values.reshape(-1,1)).flatten()
    dfTest.to_csv(f'{site.cod}_Test_15.csv', index=False)
    # Limpiar consola
    #os.system("cls" if os.name == "nt" else "clear")

    # Métricas
    print(f"=== Resultados para sitio {site.cod} ===")
    print(f"CAMS     -> RMBE:  {ms.rmbe(dfTest.ghi, dfTest.cams):.4f}   | Adap: {ms.rmbe(dfTest.ghi, dfTest.AdapCamsSLR):.4f}")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(dfTest.ghi, dfTest.lsasaf):.4f} | Adap: {ms.rmbe(dfTest.ghi, dfTest.AdapLsasafSLR):.4f}")
    print(f"CAMS     -> RMAE:  {ms.rmae(dfTest.ghi, dfTest.cams):.4f}   | Adap: {ms.rmae(dfTest.ghi, dfTest.AdapCamsSLR):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(dfTest.ghi, dfTest.lsasaf):.4f} | Adap: {ms.rmae(dfTest.ghi, dfTest.AdapLsasafSLR):.4f}")
    print(f"CAMS     -> RRMSE: {ms.rrmsd(dfTest.ghi, dfTest.cams):.4f}  | Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapCamsSLR):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(dfTest.ghi, dfTest.lsasaf):.4f}| Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapLsasafSLR):.4f}")
    print()
