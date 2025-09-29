import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Sites import Site

# Sitios
sites = ['YU', 'SA', 'SCA', 'ERO', 'LQ']

cams_models = ['cams', 'AdapCamsSLR', 'AdapCamsMLP']
lsasaf_models = ['lsasaf', 'AdapLsasafSLR', 'AdapLsasafMLP']

# Estilos de línea para las rectas de tendencia
line_styles = ['-', '--', '-.']

# Colores fijos por modelo (paleta de Matplotlib "tab10")
model_colors = {
    'cams': 'tab:blue',
    'AdapCamsSLR': 'tab:brown',
    'AdapCamsMLP': 'tab:red',
    'lsasaf': 'tab:blue',
    'AdapLsasafSLR': 'tab:brown',
    'AdapLsasafMLP': 'tab:red'
}

for site_code in sites:
    site = Site(site_code)
    df = pd.read_csv(f'{site.cod}_Test_15_SLRMLPXGB.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # -------- CAMS --------
    for i, model in enumerate(cams_models):
        if model in df.columns:
            # nube de puntos (color automático, semitransparente)
            sc = axes[0].scatter(df['ghi'], df[model], alpha=0.3, s=10, label=model)

            # Ajuste lineal con polyfit
            coef = np.polyfit(df['ghi'], df[model], 1)
            poly = np.poly1d(coef)

            # Recta con color fijo por modelo
            x_line = np.linspace(0, df['ghi'].max(), 100)
            y_line = poly(x_line)
            axes[0].plot(x_line, y_line, line_styles[i % len(line_styles)],
                         color=model_colors[model], lw=3,
                         label=f"Tendencia {model}")

    axes[0].plot([df['ghi'].min(), df['ghi'].max()],
                 [df['ghi'].min(), df['ghi'].max()], 'k:', lw=2, label="1:1")
    axes[0].set_title(f"{site.cod} - CAMS")
    axes[0].set_xlabel("GHI medida (W/m²)")
    axes[0].set_ylabel("Modelado (W/m²)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # -------- LSASAF --------
    for i, model in enumerate(lsasaf_models):
        if model in df.columns:
            sc = axes[1].scatter(df['ghi'], df[model], alpha=0.3, s=10, label=model)

            coef = np.polyfit(df['ghi'], df[model], 1)
            poly = np.poly1d(coef)

            x_line = np.linspace(0, df['ghi'].max(), 100)
            y_line = poly(x_line)
            axes[1].plot(x_line, y_line, line_styles[i % len(line_styles)],
                         color=model_colors[model], lw=3,
                         label=f"Tendencia {model}")

    axes[1].plot([df['ghi'].min(), df['ghi'].max()],
                 [df['ghi'].min(), df['ghi'].max()], 'k:', lw=2, label="1:1")
    axes[1].set_title(f"{site.cod} - LSASAF")
    axes[1].set_xlabel("GHI medida (W/m²)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # -------- Ajustar límites --------
    ghi_max = df['ghi'].max()
    pred_max = df[cams_models + lsasaf_models].max().max()
    limit = max(ghi_max, pred_max)
    for ax in axes:
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show(block=False)
