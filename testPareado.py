import pandas as pd
from Sites import Site
import Metrics as ms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
from scipy import stats

# Leer los códigos desde la consola
codigos = sys.argv[1:]

if not codigos:
    print("⚠️ Debes ingresar al menos un código de sitio (ejemplo: python testPareado.py YU SA)")
    sys.exit(1)

# Diccionario para guardar resultados completos
resultados = {}

for cod in codigos:
    site = Site(cod)

    # Cargar datos (ejemplo: yu15.csv, sa15.csv, etc.)
    df = pd.read_csv(f"{site.cod.lower()}15.csv")
    df['datetime'] = pd.to_datetime(df.datetime)
    

    # Verificar columnas necesarias
    if not all(col in df.columns for col in ["ghi", "cams", "lsasaf"]):
        print(f"❌ Faltan columnas en archivo {site.cod.lower()}15.csv")
        continue

    # Calcular errores
    df["err_cams"] = df["cams"] - df["ghi"]
    df["err_lsasaf"] = df["lsasaf"] - df["ghi"]

    # Diferencia de errores
    diff = df["err_cams"] - df["err_lsasaf"]

    # Test de normalidad (solo si N < 5000)
    if len(diff) < 5000:
        stat, p_normal = stats.shapiro(diff.dropna())
        p_normal_str = f"{p_normal:.4e}"
    else:
        p_normal_str = "omitido (N>5000)"

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(df["err_cams"], df["err_lsasaf"], nan_policy="omit")

    # Wilcoxon signed-rank test
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(df["err_cams"], df["err_lsasaf"])
    except ValueError:
        # Esto ocurre si todas las diferencias son cero
        w_stat, p_wilcoxon = None, 1.0

    # Guardar resultados
    resultados[cod] = {
        "N": len(df),
        "Shapiro p": p_normal_str,
        "t-test p": f"{p_ttest:.4e}",
        "Wilcoxon p": f"{p_wilcoxon:.4e}",
        "Conclusión": "Significativo" if (p_ttest < 0.05 or p_wilcoxon < 0.05) else "No significativo"
    }

# Mostrar resultados
print("\n=== Resultados de Pruebas Pareadas CAMS vs LSA-SAF ===")
for cod, res in resultados.items():
    print(f"\nSitio: {cod}")
    print(f"N = {res['N']}")
    print(f"Shapiro-Wilk p = {res['Shapiro p']}")
    print(f"Paired t-test p = {res['t-test p']}")
    print(f"Wilcoxon signed-rank p = {res['Wilcoxon p']}")
    print(f"Conclusión: {res['Conclusión']}")
