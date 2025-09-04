import pandas as pd
import Metrics as ms
import sys
import os

# ==============================
# Uso del script
# ==============================
# Ejemplo de ejecución:
#   python calcular_metricas.py YU SA SCA ERO LQ
# ==============================

def calcular_metricas_para_sitio(site):
    """Calcula métricas RMBE, RMAE y RRMSE para un sitio dado."""
    filename = f"{site}_Test_60_SLRMLPXGB.csv"
    
    if not os.path.exists(filename):
        print(f"[ERROR] Archivo no encontrado: {filename}")
        return None

    # Cargar datos
    df = pd.read_csv(filename)

    # Calcular métricas para CAMS y LSASAF
    resultados = {
        "Site": site,
        "CAMS_RMBE": ms.rmbe(df.ghi, df['AdapCamsSLR']),
        "LSASAF_RMBE": ms.rmbe(df.ghi, df['AdapLsasafSLR']),
        "ERA_RMBE": ms.rmbe(df.ghi, df['AdapEraSLR']),
        "MERRA_RMBE": ms.rmbe(df.ghi, df['AdapMerraSLR']),
        "CAMS_RMAE": ms.rmae(df.ghi, df['AdapCamsSLR']),
        "LSASAF_RMAE": ms.rmae(df.ghi, df['AdapLsasafSLR']),
        "ERA_RMAE": ms.rmae(df.ghi, df['AdapEraSLR']),
        "MERRA_RMAE": ms.rmae(df.ghi, df['AdapMerraSLR']),
        "CAMS_RRMSE": ms.rrmsd(df.ghi, df['AdapCamsSLR']),
        "LSASAF_RRMSE": ms.rrmsd(df.ghi, df['AdapLsasafSLR']),
        "ERA_RRMSE": ms.rrmsd(df.ghi, df['AdapEraSLR']),
        "MERRA_RRMSE": ms.rrmsd(df.ghi, df['AdapMerraSLR']),
    }

    return resultados

def main():
    # Leer los sitios desde la consola
    sitios = sys.argv[1:]
    
    if not sitios:
        print("Uso: python calcular_metricas.py YU SA SCA ERO LQ")
        sys.exit(1)

    print("Calculando métricas para los sitios:", ", ".join(sitios))
    print("=" * 80)

    resultados_totales = []

    for site in sitios:
        res = calcular_metricas_para_sitio(site)
        if res:
            resultados_totales.append(res)

    # Mostrar resultados en una tabla
    if resultados_totales:
        df_resultados = pd.DataFrame(resultados_totales)
        print(df_resultados.to_string(index=False))

if __name__ == "__main__":
    main()
