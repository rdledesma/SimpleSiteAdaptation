import matplotlib.pyplot as plt
import numpy as np

# Simulación de días
dias = np.arange(1, 31)

# Datos "satélite" (sesgo + ruido)
satelite = 1.75 + 0.9*np.sin(dias/3) + np.random.normal(0, 0.1, len(dias))

# Datos medidos en sitio
sitio = 2 + 1.0*np.sin(dias/3) + np.random.normal(0, 0.1, len(dias))

# Corrección aplicada (ajuste de sesgo)
adaptado = 2 + 0.9*np.sin(dias/3) + np.random.normal(0, 0.1, len(dias))

# Crear figura
plt.figure(figsize=(12,6))

# Aumentar grosor de líneas
plt.plot(dias, sitio, label="Medición en sitio", linewidth=3)
plt.plot(dias, satelite, label="Satélite sin adaptar", linestyle="--", linewidth=3)
plt.plot(dias, adaptado, label="Satélite adaptado", linestyle=":", linewidth=3)

# Etiquetas y título con tamaño de fuente mayor
plt.xlabel("Días", fontsize=16)
plt.ylabel("GHI (kWh/m²)", fontsize=16)
# plt.title("Ejemplo de Adaptación al Sitio", fontsize=18)

# Leyenda más grande
plt.legend(fontsize=14)

# Aumentar tamaño de ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Simulación de datos
np.random.seed(42)
n = 100

# Mediciones en sitio (referencia)
sitio = np.linspace(2, 6, n) + np.random.normal(0, 0.1, n)

# Satélite sin adaptar (sesgo + dispersión)
satelite = sitio * 0.8 + 0.5 + np.random.normal(0, 0.2, n)

# Satélite adaptado (corregido)
adaptado = sitio * 1.0 + np.random.normal(0, 0.2, n)

plt.figure(figsize=(10,10))  # Figura más grande

# Dispersión antes (rojo) con puntos más grandes
plt.scatter(sitio, satelite, label="Satélite sin adaptar", 
            alpha=0.8, marker="o", color="red", s=150)

# Dispersión después (verde) con puntos más grandes
plt.scatter(sitio, adaptado, label="Satélite adaptado", 
            alpha=0.8, marker="s", color="green", s=150)

# Línea 1:1 (ideal) más gruesa
plt.plot([sitio.min(), sitio.max()], [sitio.min(), sitio.max()], 
         color="black", linestyle="--", linewidth=3, label="Línea 1:1 (ideal)")

# Anotaciones más grandes
plt.annotate("Sesgo evidente\n(está por debajo de la línea ideal)", 
             xy=(sitio[65], satelite[65]), xytext=(4.5, 2.5),
             arrowprops=dict(facecolor='red', shrink=0.05, width=3),
             fontsize=16, color="red")

plt.annotate("Después de la adaptación\nlos puntos se alinean con la línea ideal", 
             xy=(sitio[30], adaptado[25]), xytext=(3, 5.5),
             arrowprops=dict(facecolor='green', shrink=0.05, width=3),
             fontsize=16, color="green")

# Etiquetas más grandes
plt.xlabel("Medición en sitio (kWh/m²)", fontsize=20)
plt.ylabel("Estimación satelital (kWh/m²)", fontsize=20)

# Ticks más grandes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Leyenda más grande
plt.legend(fontsize=16)

# Rejilla
plt.grid(alpha=0.3)

# Título opcional
# plt.title("Adaptación al Sitio: Antes (rojo) vs Después (verde)", fontsize=22)

plt.tight_layout()
plt.show()
