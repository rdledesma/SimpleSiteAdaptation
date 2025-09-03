import pandas as pd
from Sites import Site
import Metrics as ms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys

# Leer los códigos desde la consola
codigos = sys.argv[1:]

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python script.py YU SA)")
    sys.exit(1)

# Diccionario para guardar resultados completos
resultados = {}

for cod in codigos:
    site = Site(cod)

    # Cargar datos
    df = pd.read_csv(f"{site.cod.lower()}15.csv")
    df['datetime'] = pd.to_datetime(df.datetime)

    # Listas para guardar los RRMSD de cada iteración
    rrmsd_cams = []
    rrmsd_adap_cams = []
    rrmsd_lsasaf = []
    rrmsd_adap_lsasaf = []

    for i in range(1000):
        # Train / Validation / Test
        dfTrain = df[df.datetime.dt.year == df.datetime.dt.year.unique()[0]]
        dfTrain, dfVal = train_test_split(dfTrain, test_size=0.2, random_state=i, shuffle=True)
        dfTest = df[df.datetime.dt.year != df.datetime.dt.year.unique()[0]]

        # Modelos
        regCams = LinearRegression().fit(dfTrain.cams.values.reshape(-1,1), dfTrain.ghi)
        regLsasaf = LinearRegression().fit(dfTrain.lsasaf.values.reshape(-1,1), dfTrain.ghi)

        # Predicciones
        dfTest['AdapCamsSLR'] = regCams.predict(dfTest.cams.values.reshape(-1,1)).flatten()
        dfTest['AdapLsasafSLR'] = regLsasaf.predict(dfTest.lsasaf.values.reshape(-1,1)).flatten()

        # Guardar métricas
        rrmsd_cams.append(ms.rrmsd(dfTest.ghi, dfTest.cams))
        rrmsd_adap_cams.append(ms.rrmsd(dfTest.ghi, dfTest.AdapCamsSLR))
        rrmsd_lsasaf.append(ms.rrmsd(dfTest.ghi, dfTest.lsasaf))
        rrmsd_adap_lsasaf.append(ms.rrmsd(dfTest.ghi, dfTest.AdapLsasafSLR))

    # Guardar resultados completos
    resultados[cod] = {
        'CAMS': rrmsd_cams,
        'AdapCams': rrmsd_adap_cams,
        'LSA-SAF': rrmsd_lsasaf,
        'AdapLsasaf': rrmsd_adap_lsasaf
    }

# Convertir resultados a DataFrame para análisis
filas = []
for cod, metrics in resultados.items():
    fila = {
        'Sitio': cod,
        'CAMS_mean': pd.Series(metrics['CAMS']).mean(),
        'CAMS_std': pd.Series(metrics['CAMS']).std(),
        'AdapCams_mean': pd.Series(metrics['AdapCams']).mean(),
        'AdapCams_std': pd.Series(metrics['AdapCams']).std(),
        'LSASAF_mean': pd.Series(metrics['LSA-SAF']).mean(),
        'LSASAF_std': pd.Series(metrics['LSA-SAF']).std(),
        'AdapLSASAF_mean': pd.Series(metrics['AdapLsasaf']).mean(),
        'AdapLSASAF_std': pd.Series(metrics['AdapLsasaf']).std(),
    }
    filas.append(fila)

df_resultados = pd.DataFrame(filas)

# Mostrar DataFrame final
print(df_resultados)

# Guardar DataFrame a CSV para análisis posterior
df_resultados.to_csv('RRMSD_Resultados.csv', index=False)
