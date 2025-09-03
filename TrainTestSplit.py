import pandas as pd
from Sites import Site
import Metrics as ms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
site = Site('YU')


df = pd.read_csv(f"{site.cod.lower()}15.csv")
df['datetime'] = pd.to_datetime(df.datetime)

dfTrain = df[df.datetime.dt.year == df.datetime.dt.year.unique()[0]]
dfTrain, dfVal = train_test_split(dfTrain, test_size=0.2, random_state=42, shuffle=True)
dfTest = df[df.datetime.dt.year != df.datetime.dt.year.unique()[0]]


regCams = LinearRegression().fit(dfTrain.cams.values.reshape(-1,1), dfTrain.ghi)
regLsasaf = LinearRegression().fit(dfTrain.lsasaf.values.reshape(-1,1), dfTrain.ghi)


dfTest['AdapCams'] = regCams.predict(dfTest.cams.values.reshape(-1,1)).flatten()
dfTest['AdapLsasaf'] = regLsasaf.predict(dfTest.lsasaf.values.reshape(-1,1)).flatten()

#ms.rrmsd(dfTrain.ghi, dfTrain.cams)
#ms.rrmsd(dfVal.ghi, dfVal.cams)
os.system("clear")
print(f"CAMS: {ms.rrmsd(dfTest.ghi, dfTest.cams)}) Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapCams)}) ")

print(f"LSA-SAF: {ms.rrmsd(dfTest.ghi, dfTest.lsasaf)}) Adap: {ms.rrmsd(dfTest.ghi, dfTest.AdapLsasaf)}) ")