import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from Sites import Site
import Metrics as ms

# ==============================
# Configuración
# ==============================
sitios = ["YU", "SA", "SCA", "ERO", "LQ"]
features = {
    "CAMS": ["cams",'lat','lon','alt'],
    "LSASAF": ["lsasaf",'lat','lon','alt']
}

# Matrices de resultados
rrmse_cams_adap = pd.DataFrame(index=sitios, columns=sitios, dtype=float)
rrmse_lsasaf_adap = pd.DataFrame(index=sitios, columns=sitios, dtype=float)
rrmse_cams_orig = pd.Series(index=sitios, dtype=float)
rrmse_lsasaf_orig = pd.Series(index=sitios, dtype=float)

# ==============================
# Errores originales (baseline)
# ==============================
for site_code in sitios:
    site = Site(site_code)
    df_test = pd.read_csv(f"{site.cod}_Test_15_SLRMLP.csv")
    df_test['lat'] = site.lat
    df_test['lon'] = site.long
    df_test['alt'] = site.alt
    y_test = df_test["ghi"].values.astype(np.float32)

    rrmse_cams_orig[site_code] = ms.rrmsd(y_test, df_test["cams"].values)
    rrmse_lsasaf_orig[site_code] = ms.rrmsd(y_test, df_test["lsasaf"].values)

# ==============================
# Evaluación cruzada con modelos adaptados
# ==============================
for train_site in sitios:
    print(f"\n🔎 Evaluando modelos entrenados en {train_site}...")

    # Cargar modelos entrenados en train_site
    model_cams = joblib.load(f"{train_site}_BestModel_CAMS_XGB.pkl")
    model_lsasaf = joblib.load(f"{train_site}_BestModel_LSASAF_XGB.pkl")

    for test_site in sitios:
        site = Site(test_site)

        # Cargar dataset de test
        df_test = pd.read_csv(f"{site.cod}_Test_15_SLRMLP.csv")
        df_test['lat'] = site.lat
        df_test['lon'] = site.long
        df_test['alt'] = site.alt
        y_test = df_test["ghi"].values.astype(np.float32)

        # CAMS adaptado
        X_test_cams = df_test[features["CAMS"]].values.astype(np.float32)
        y_pred_cams = model_cams.predict(X_test_cams)
        rrmse_cams_adap.loc[train_site, test_site] = ms.rrmsd(y_test, y_pred_cams)

        # LSA-SAF adaptado
        X_test_lsasaf = df_test[features["LSASAF"]].values.astype(np.float32)
        y_pred_lsasaf = model_lsasaf.predict(X_test_lsasaf)
        rrmse_lsasaf_adap.loc[train_site, test_site] = ms.rrmsd(y_test, y_pred_lsasaf)

# ==============================
# Plot: Heatmaps de RRMSE
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# CAMS original
sns.heatmap(pd.DataFrame(rrmse_cams_orig).T, annot=True, fmt=".1f",
            cmap="magma", cbar=False, ax=axes[0, 0])
axes[0, 0].set_title("RRMSE - CAMS Original")
axes[0, 0].set_xlabel("Sitios")
axes[0, 0].set_ylabel("")

# CAMS adaptado
sns.heatmap(rrmse_cams_adap, annot=True, fmt=".1f",
            cmap="viridis", ax=axes[0, 1])
axes[0, 1].set_title("RRMSE - CAMS Adaptado (XGB)")
axes[0, 1].set_xlabel("Sitio de test")
axes[0, 1].set_ylabel("Sitio de entrenamiento")

# LSA-SAF original
sns.heatmap(pd.DataFrame(rrmse_lsasaf_orig).T, annot=True, fmt=".1f",
            cmap="magma", cbar=False, ax=axes[1, 0])
axes[1, 0].set_title("RRMSE - LSA-SAF Original")
axes[1, 0].set_xlabel("Sitios")
axes[1, 0].set_ylabel("")

# LSA-SAF adaptado
sns.heatmap(rrmse_lsasaf_adap, annot=True, fmt=".1f",
            cmap="viridis", ax=axes[1, 1])
axes[1, 1].set_title("RRMSE - LSA-SAF Adaptado (XGB)")
axes[1, 1].set_xlabel("Sitio de test")
axes[1, 1].set_ylabel("Sitio de entrenamiento")

plt.tight_layout()
plt.show()
