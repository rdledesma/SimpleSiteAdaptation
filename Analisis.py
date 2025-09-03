import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos (puedes usar tu DataFrame directamente)
df = pd.read_csv('RRMSD_Resultados.csv')

# Posiciones para las barras
x = np.arange(len(df['Sitio']))
width = 0.2  # ancho de las barras

# Crear la figura
fig, ax = plt.subplots(figsize=(12,6))

# Barras con desviación estándar
ax.bar(x - 1.5*width, df['CAMS_mean'], width, yerr=df['CAMS_std'], label='CAMS', capsize=4)
ax.bar(x - 0.5*width, df['AdapCams_mean'], width, yerr=df['AdapCams_std'], label='AdapCams', capsize=4)
ax.bar(x + 0.5*width, df['LSASAF_mean'], width, yerr=df['LSASAF_std'], label='LSASAF', capsize=4)
ax.bar(x + 1.5*width, df['AdapLSASAF_mean'], width, yerr=df['AdapLSASAF_std'], label='AdapLSASAF', capsize=4)

# Configuración del gráfico
ax.set_xlabel('Sitio')
ax.set_ylabel('RRMSD')
ax.set_title('Comparación de RRMSD por sitio y modelo')
ax.set_xticks(x)
ax.set_xticklabels(df['Sitio'])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


"""

Se evaluó el desempeño de los modelos CAMS y LSA-SAF, así como sus versiones adaptadas mediante regresión lineal, sobre los datos de test de cada sitio. Para cada sitio se realizaron 1000 iteraciones de partición aleatoria de entrenamiento y validación, calculando el RRMSD en cada iteración.

Los resultados se resumen en la Figura X, donde se muestran las medias y desviaciones estándar de RRMSD para cada modelo y sitio. Se observa que las versiones adaptadas (AdapCams y AdapLSASAF) generalmente reducen el error en comparación con las series originales, indicando que la regresión lineal logra una mejor aproximación al GHI medido, con mejoras más significativas en algunos sitios específicos (por ejemplo, YU y SA).

El valor de la desviación estándar (std) que se calculó no ha sido modificado y puede ser útil como referencia sobre la variabilidad de los resultados. Para obtener una estimación rápida del desempeño del modelo, podría resultar práctico emplear una validación cruzada simple (hold-out), aunque esta es menos robusta que la validación k-fold utilizada hasta ahora. La k-fold, al promediar el rendimiento sobre múltiples particiones, proporciona una evaluación más estable y confiable, pero a costa de una mayor complejidad computacional.
"""