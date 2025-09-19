import pandas as pd
from Sites import Site
from Geo import Geo
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import Metrics as ms  # módulo personalizado
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
# -----------------------------
# Lectura de datos
# -----------------------------
site = Site('TG')

meas = pd.read_csv('/home/inenco/Documentos/Medidas/tg_15.csv')
lsasaf = pd.read_csv('/home/inenco/Documentos/LSASAF/tg_15.csv')

meas['datetime'] = pd.to_datetime(meas.datetime)
lsasaf['datetime'] = pd.to_datetime(lsasaf.datetime)


plt.figure()
plt.plot(meas.datetime, meas.ghi, '-.r')
plt.show()




dfGeo = Geo(
    range_dates=lsasaf.datetime + timedelta(minutes=7.5),
    lat=site.lat,
    long=site.long,
    alt=site.alt,
    gmt=0,
    beta=0
).df

meas = (
    meas.set_index('datetime')
        .reindex(lsasaf.datetime)
        .rename_axis(['datetime'])
        .reset_index()
)

lsasaf['lat'] = site.lat
lsasaf['lon'] = site.long
lsasaf['alt'] = site.alt
lsasaf['ghi'] = meas.ghi.values
lsasaf['SZA'] = dfGeo.SZA.values
lsasaf['lsasaf'] = lsasaf.GHI.values
lsasaf['N'] = lsasaf.datetime.dt.day_of_year

df_new = lsasaf.dropna()
mlr_model = joblib.load("mlr_model.pkl")

# Selección de features
features = ["lsasaf", "lat", "lon", "alt",'N']
X_new = df_new[features].values

# Predecir GHI adaptado
df_new["AdapLsasafMLR"] = mlr_model.predict(X_new)

y_true = df_new.ghi.values

# Resultados
print(f"\nSitio {site.cod}:")
print(f"LSA-SAF  -> RMBE:  {ms.rmbe(y_true, df_new.lsasaf):.4f} | Adap: {ms.rmbe(y_true, df_new.AdapLsasafMLR):.4f}")
print(f"LSA-SAF  -> RMAE:  {ms.rmae(y_true, df_new.lsasaf):.4f} | Adap: {ms.rmae(y_true, df_new.AdapLsasafMLR):.4f}")
print(f"LSA-SAF  -> RRMSE: {ms.rrmsd(y_true, df_new.lsasaf):.4f} | Adap: {ms.rrmsd(y_true, df_new.AdapLsasafMLR):.4f}")