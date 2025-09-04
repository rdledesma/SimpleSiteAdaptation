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
    df = pd.read_csv(f"{site.cod.lower()}60.csv")
    df['datetime'] = pd.to_datetime(df.datetime)

    # Train / Validation / Test
    dfTrain = df[df.datetime.dt.year == df.datetime.dt.year.unique()[1]]
    dfTrain, dfVal = train_test_split(dfTrain, test_size=0.2, random_state=42, shuffle=True)
    dfTest = df[df.datetime.dt.year != df.datetime.dt.year.unique()[1]]

    dfTrain.to_csv(f'{site.cod}_Train_60.csv', index=False)
    dfVal.to_csv(f'{site.cod}_Val_60.csv', index=False)
    

    # Modelos
    regCams = LinearRegression().fit(dfTrain.cams.values.reshape(-1,1), dfTrain.ghi)
    regLsasaf = LinearRegression().fit(dfTrain.lsasaf.values.reshape(-1,1), dfTrain.ghi)
    regEra = LinearRegression().fit(dfTrain.era.values.reshape(-1,1), dfTrain.ghi)
    regMerra = LinearRegression().fit(dfTrain.merra.values.reshape(-1,1), dfTrain.ghi)

    # Predicciones
    dfTest['AdapCamsSLR'] = regCams.predict(dfTest.cams.values.reshape(-1,1)).flatten()
    dfTest['AdapLsasafSLR'] = regLsasaf.predict(dfTest.lsasaf.values.reshape(-1,1)).flatten()
    dfTest['AdapEraSLR'] = regEra.predict(dfTest.era.values.reshape(-1,1)).flatten()
    dfTest['AdapMerraSLR'] = regMerra.predict(dfTest.merra.values.reshape(-1,1)).flatten()

    dfTest.to_csv(f'{site.cod}_Test_60.csv', index=False)
    # Limpiar consola
    #os.system("cls" if os.name == "nt" else "clear")

    # Métricas
    print(f"=== Resultados para sitio {site.cod} ===")
    
    print(f"CAMS     -> RMBE:  {ms.rmbe(dfTest.ghi, dfTest.cams):.4f}   | Adap: {ms.rmbe(dfTest.ghi, dfTest.AdapCamsSLR):.4f}")
    print(f"LSA-SAF  -> RMBE:  {ms.rmbe(dfTest.ghi, dfTest.lsasaf):.4f} | Adap: {ms.rmbe(dfTest.ghi, dfTest.AdapLsasafSLR):.4f}")
    print(f"ERA-5    -> RMBE:  {ms.rmbe(dfTest.ghi, dfTest.era):.4f}    | Adap: {ms.rmbe(dfTest.ghi, dfTest.AdapEraSLR):.4f}")
    print(f"MERRA-2  -> RMBE:  {ms.rmbe(dfTest.ghi, dfTest.merra):.4f}  | Adap: {ms.rmbe(dfTest.ghi, dfTest.AdapMerraSLR):.4f}")
    print()
    print(f"CAMS     -> RMAE:  {ms.rmae(dfTest.ghi, dfTest.cams):.4f}   | Adap: {ms.rmae(dfTest.ghi, dfTest.AdapCamsSLR):.4f}")
    print(f"LSA-SAF  -> RMAE:  {ms.rmae(dfTest.ghi, dfTest.lsasaf):.4f} | Adap: {ms.rmae(dfTest.ghi, dfTest.AdapLsasafSLR):.4f}")
    print(f"ERA-5    -> RMAE:  {ms.rmae(dfTest.ghi, dfTest.era):.4f}   | Adap: {ms.rmae(dfTest.ghi, dfTest.AdapEraSLR):.4f}")
    print(f"MERRA-2  -> RMAE:  {ms.rmae(dfTest.ghi, dfTest.merra):.4f} | Adap: {ms.rmae(dfTest.ghi, dfTest.AdapMerraSLR):.4f}")
    
    print()
    print(f"CAMS     -> RRMSE: {ms.rrmsd(dfTest.ghi, dfTest.cams):.4f}  | Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapCamsSLR):.4f}")
    print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(dfTest.ghi, dfTest.lsasaf):.4f}| Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapLsasafSLR):.4f}")
    print(f"ERA      -> RRMSE: {ms.rrmsd(dfTest.ghi, dfTest.era):.4f}   | Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapEraSLR):.4f}")
    print(f"MERRA    -> RRMSE: {ms.rrmsd(dfTest.ghi, dfTest.merra):.4f} | Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapMerraSLR):.4f}")
    
    print()
