# -*- coding: utf-8 -*-
"""
Figuras para tesis: impacto de la site adaptation en métricas (MBE, MAE, RMSE)
sobre series de GHI a 15 minutos y 60 minutos.

Diseño:
1) Heatmaps por resolución (15 min y 60 min) que muestran la mejora relativa (%)
   en MAE y RMSE por sitio y dataset base, comparando el mejor adaptado vs. el original.
   (Verde = mejora; Rojo = empeora)
2) Barras comparativas (original vs mejor adaptado) por sitio y dataset base,
   con anotaciones de cambio absoluto (p.p.) y relativo (%).

Librerías: solo matplotlib/numpy/pandas (sin seaborn).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# ----------------------------
# 1) DATOS (copiados de las tablas del usuario)
# ----------------------------
SITES = ["YU", "SA", "SCA", "ERO", "LQ"]
DATASETS_15 = ["CAMS", "LSA-SAF"]
DATASETS_60 = ["CAMS", "LSA-SAF", "ERA-5", "MERRA-2"]

# Métricas originales (sin site adaptation) -------------- #
# Resolución: 15 minutos
orig_15 = {
    "CAMS": {
        "YU":  {"MBE": -0.2, "MAE": 17.4, "RMSE": 27.2},
        "SA":  {"MBE":  3.9, "MAE": 23.4, "RMSE": 33.7},
        "SCA": {"MBE":  2.6, "MAE": 21.8, "RMSE": 29.8},
        "ERO": {"MBE": -23.6,"MAE": 27.6, "RMSE": 40.9},
        "LQ":  {"MBE": -6.9, "MAE": 16.5, "RMSE": 25.8},
    },
    "LSA-SAF": {
        "YU":  {"MBE":  7.6, "MAE": 16.5, "RMSE": 25.5},
        "SA":  {"MBE": 17.9, "MAE": 27.4, "RMSE": 39.4},
        "SCA": {"MBE": 13.1, "MAE": 22.2, "RMSE": 30.9},
        "ERO": {"MBE": -7.7, "MAE": 16.2, "RMSE": 26.5},
        "LQ":  {"MBE":  4.0, "MAE": 12.6, "RMSE": 22.6},
    },
}

# Resolución: horaria (60 minutos)
orig_60 = {
    "CAMS": {
        "YU":  {"MBE": -0.2, "MAE": 15.3, "RMSE": 23.5},
        "SA":  {"MBE":  4.0, "MAE": 20.9, "RMSE": 29.3},
        "SCA": {"MBE":  3.0, "MAE": 19.7, "RMSE": 26.1},
        "ERO": {"MBE": -23.6,"MAE": 26.6, "RMSE": 39.3},
        "LQ":  {"MBE": -4.8, "MAE": 14.5, "RMSE": 21.6},
    },
    "LSA-SAF": {
        "YU":  {"MBE":  7.6, "MAE": 14.5, "RMSE": 21.9},
        "SA":  {"MBE": 17.9, "MAE": 25.3, "RMSE": 35.7},
        "SCA": {"MBE": 13.3, "MAE": 20.3, "RMSE": 27.1},
        "ERO": {"MBE": -7.7, "MAE": 14.8, "RMSE": 24.0},
        "LQ":  {"MBE":  4.8, "MAE": 10.9, "RMSE": 18.2},
    },
    "ERA-5": {
        "YU":  {"MBE": -4.0, "MAE": 43.5, "RMSE": 60.2},
        "SA":  {"MBE":  9.4, "MAE": 27.1, "RMSE": 37.7},
        "SCA": {"MBE":  1.9, "MAE": 20.2, "RMSE": 29.7},
        "ERO": {"MBE": -14.0,"MAE": 19.3, "RMSE": 25.6},
        "LQ":  {"MBE": -0.8, "MAE": 11.3, "RMSE": 18.4},
    },
    "MERRA-2": {
        "YU":  {"MBE": 25.0, "MAE": 33.6, "RMSE": 51.2},
        "SA":  {"MBE": 43.4, "MAE": 48.3, "RMSE": 65.0},
        "SCA": {"MBE": 10.9, "MAE": 21.7, "RMSE": 30.3},
        "ERO": {"MBE": -3.8, "MAE": 13.1, "RMSE": 20.4},
        "LQ":  {"MBE":  1.4, "MAE": 13.4, "RMSE": 20.8},
    },
}

# Métricas adaptadas (site adaptation) -------------- #
# 15 minutos
adapt_15 = {
    "CAMS": {
        "SLR": {
            "YU":  {"MBE": -0.9, "MAE": 17.1, "RMSE": 26.4},
            "SA":  {"MBE":  3.8, "MAE": 21.3, "RMSE": 31.5},
            "SCA": {"MBE": -1.5, "MAE": 18.4, "RMSE": 26.0},
            "ERO": {"MBE":  2.5, "MAE": 24.8, "RMSE": 31.5},
            "LQ":  {"MBE":  2.1, "MAE": 15.9, "RMSE": 23.5},
        },
        "MLP": {
            "YU":  {"MBE": -4.4, "MAE": 17.7, "RMSE": 26.2},
            "SA":  {"MBE":  4.8, "MAE": 21.4, "RMSE": 31.6},
            "SCA": {"MBE":  7.7, "MAE": 17.8, "RMSE": 27.7},
            "ERO": {"MBE":  0.5, "MAE": 24.1, "RMSE": 31.2},
            "LQ":  {"MBE":  9.1, "MAE": 19.4, "RMSE": 25.6},
        },
        "XGB": {
            "YU":  {"MBE": -1.3, "MAE": 17.1, "RMSE": 26.0},
            "SA":  {"MBE":  3.8, "MAE": 21.4, "RMSE": 31.4},
            "SCA": {"MBE": -1.8, "MAE": 18.9, "RMSE": 26.2},
            "ERO": {"MBE":  2.5, "MAE": 23.9, "RMSE": 30.9},
            "LQ":  {"MBE":  2.2, "MAE": 15.9, "RMSE": 23.5},
        },
    },
    "LSA-SAF": {
        "SLR": {
            "YU":  {"MBE": -5.5, "MAE": 18.4, "RMSE": 25.0},
            "SA":  {"MBE":  4.4, "MAE": 23.9, "RMSE": 34.6},
            "SCA": {"MBE":  1.0, "MAE": 18.0, "RMSE": 26.7},
            "ERO": {"MBE":  2.7, "MAE": 17.1, "RMSE": 25.4},
            "LQ":  {"MBE":  2.2, "MAE": 12.9, "RMSE": 22.3},
        },
        "MLP": {
            "YU":  {"MBE": -7.1, "MAE": 18.6, "RMSE": 25.0},
            "SA":  {"MBE":  4.5, "MAE": 23.6, "RMSE": 34.5},
            "SCA": {"MBE":  5.9, "MAE": 18.4, "RMSE": 26.6},
            "ERO": {"MBE":  2.5, "MAE": 17.0, "RMSE": 25.3},
            "LQ":  {"MBE": -0.4, "MAE": 13.6, "RMSE": 22.1},
        },
        "XGB": {
            "YU":  {"MBE": -6.0, "MAE": 18.2, "RMSE": 24.9},
            "SA":  {"MBE":  4.3, "MAE": 23.9, "RMSE": 34.6},
            "SCA": {"MBE":  0.5, "MAE": 18.6, "RMSE": 27.0},
            "ERO": {"MBE":  2.4, "MAE": 17.2, "RMSE": 25.1},
            "LQ":  {"MBE":  2.3, "MAE": 13.1, "RMSE": 22.3},
        },
    },
}

# 60 minutos
adapt_60 = {
    "CAMS": {
        "SLR": {
            "YU":  {"MBE": -1.3, "MAE": 14.7, "RMSE": 22.7},
            "SA":  {"MBE":  3.5, "MAE": 18.5, "RMSE": 27.0},
            "SCA": {"MBE": -1.7, "MAE": 15.9, "RMSE": 22.0},
            "ERO": {"MBE":  2.1, "MAE": 23.4, "RMSE": 29.6},
            "LQ":  {"MBE":  2.6, "MAE": 13.8, "RMSE": 19.4},
        },
        "MLP": {
            "YU":  {"MBE": -4.9, "MAE": 16.2, "RMSE": 23.0},
            "SA":  {"MBE":  4.3, "MAE": 18.6, "RMSE": 27.2},
            "SCA": {"MBE": -4.0, "MAE": 16.6, "RMSE": 22.4},
            "ERO": {"MBE":  3.8, "MAE": 23.5, "RMSE": 29.5},
            "LQ":  {"MBE": -0.5, "MAE": 13.0, "RMSE": 19.2},
        },
        "XGB": {
            "YU":  {"MBE": -1.6, "MAE": 14.8, "RMSE": 22.2},
            "SA":  {"MBE":  3.4, "MAE": 18.9, "RMSE": 27.1},
            "SCA": {"MBE": -2.4, "MAE": 16.6, "RMSE": 22.4},
            "ERO": {"MBE":  2.1, "MAE": 22.8, "RMSE": 29.2},
            "LQ":  {"MBE":  2.7, "MAE": 13.9, "RMSE": 19.6},
        },
    },
    "LSA-SAF": {
        "SLR": {
            "YU":  {"MBE": -5.5, "MAE": 16.0, "RMSE": 21.2},
            "SA":  {"MBE":  4.3, "MAE": 21.2, "RMSE": 30.5},
            "SCA": {"MBE":  1.1, "MAE": 15.6, "RMSE": 22.7},
            "ERO": {"MBE":  2.5, "MAE": 15.7, "RMSE": 22.8},
            "LQ":  {"MBE":  0.3, "MAE": 10.6, "RMSE": 17.4},
        },
        "MLP": {
            "YU":  {"MBE": -5.4, "MAE": 15.4, "RMSE": 20.9},
            "SA":  {"MBE":  2.5, "MAE": 21.4, "RMSE": 30.2},
            "SCA": {"MBE":  8.3, "MAE": 16.1, "RMSE": 24.1},
            "ERO": {"MBE":  9.6, "MAE": 19.4, "RMSE": 24.6},
            "LQ":  {"MBE": -3.8, "MAE": 12.1, "RMSE": 17.8},
        },
        "XGB": {
            "YU":  {"MBE": -6.1, "MAE": 16.0, "RMSE": 21.3},
            "SA":  {"MBE":  4.1, "MAE": 21.4, "RMSE": 30.6},
            "SCA": {"MBE":  0.0, "MAE": 16.9, "RMSE": 23.4},
            "ERO": {"MBE":  2.2, "MAE": 15.9, "RMSE": 22.6},
            "LQ":  {"MBE":  0.2, "MAE": 10.9, "RMSE": 17.4},
        },
    },
    "ERA-5": {
        "SLR": {
            "YU":  {"MBE":  1.2, "MAE": 44.0, "RMSE": 55.1},
            "SA":  {"MBE":  6.4, "MAE": 26.5, "RMSE": 36.9},
            "SCA": {"MBE": -6.4, "MAE": 20.1, "RMSE": 29.1},
            "ERO": {"MBE": -0.6, "MAE": 15.4, "RMSE": 21.4},
            "LQ":  {"MBE":  2.1, "MAE": 11.6, "RMSE": 18.4},
        },
        "MLP": {
            "YU":  {"MBE":  0.2, "MAE": 43.7, "RMSE": 55.1},
            "SA":  {"MBE":  9.6, "MAE": 27.1, "RMSE": 37.6},
            "SCA": {"MBE": -1.8, "MAE": 19.0, "RMSE": 28.5},
            "ERO": {"MBE": -7.5, "MAE": 16.7, "RMSE": 23.1},
            "LQ":  {"MBE":  0.1, "MAE": 11.3, "RMSE": 18.3},
        },
        "XGB": {
            "YU":  {"MBE":  1.4, "MAE": 44.6, "RMSE": 55.4},
            "SA":  {"MBE":  6.3, "MAE": 27.3, "RMSE": 37.4},
            "SCA": {"MBE": -7.3, "MAE": 20.7, "RMSE": 29.3},
            "ERO": {"MBE": -0.5, "MAE": 15.2, "RMSE": 21.2},
            "LQ":  {"MBE":  2.0, "MAE": 11.8, "RMSE": 18.6},
        },
    },
    "MERRA-2": {
        "SLR": {
            "YU":  {"MBE": -3.6, "MAE": 34.7, "RMSE": 44.1},
            "SA":  {"MBE":  7.1, "MAE": 35.2, "RMSE": 46.0},
            "SCA": {"MBE": -2.2, "MAE": 20.3, "RMSE": 27.7},
            "ERO": {"MBE":  0.7, "MAE": 12.9, "RMSE": 20.1},
            "LQ":  {"MBE":  0.3, "MAE": 13.1, "RMSE": 20.4},
        },
        "MLP": {
            "YU":  {"MBE":  0.7, "MAE": 33.7, "RMSE": 43.9},
            "SA":  {"MBE":  0.8, "MAE": 36.3, "RMSE": 45.4},
            "SCA": {"MBE": -0.3, "MAE": 19.5, "RMSE": 27.5},
            "ERO": {"MBE": 11.7, "MAE": 17.5, "RMSE": 23.4},
            "LQ":  {"MBE": -2.8, "MAE": 13.4, "RMSE": 20.6},
        },
        "XGB": {
            "YU":  {"MBE": -4.9, "MAE": 35.3, "RMSE": 44.5},
            "SA":  {"MBE":  6.9, "MAE": 35.6, "RMSE": 46.1},
            "SCA": {"MBE": -2.8, "MAE": 20.8, "RMSE": 28.2},
            "ERO": {"MBE":  0.8, "MAE": 13.3, "RMSE": 20.3},
            "LQ":  {"MBE":  0.2, "MAE": 13.5, "RMSE": 20.6},
        },
    },
}

# ----------------------------
# 2) UTILIDADES
# ----------------------------
def dict_to_df(original: dict) -> pd.DataFrame:
    """Convierte dict anidado a DataFrame (índice: sitio, columnas: (dataset, métrica))."""
    frames = []
    for ds, by_site in original.items():
        recs = {site: metrics for site, metrics in by_site.items()}
        df = pd.DataFrame.from_dict(recs, orient="index")
        df.columns = pd.MultiIndex.from_product([[ds], df.columns])
        frames.append(df)
    out = pd.concat(frames, axis=1).loc[SITES]
    return out

def best_adapted_df(adapted: dict, metric: str) -> pd.DataFrame:
    """
    Retorna DataFrame con el *mejor* valor por (dataset base x sitio) para una métrica dada,
    tomando el mínimo para MAE/RMSE y el mínimo valor absoluto para MBE.
    """
    best = pd.DataFrame(index=SITES)
    for ds, methods in adapted.items():
        # recolectar por método
        stacks = []
        for mname, by_site in methods.items():
            vals = pd.Series({site: (abs(m["MBE"]) if metric=="|MBE|" else m[metric]) for site, m in by_site.items()})
            vals.name = mname
            stacks.append(vals)
        mat = pd.concat(stacks, axis=1)
        best_vals = mat.min(axis=1)
        best[ds] = best_vals
    return best[DATASETS_15] if set(best.columns)==set(DATASETS_15) else best[DATASETS_60]

def original_metric_df(original: dict, metric: str) -> pd.DataFrame:
    """Extrae DataFrame para una métrica del original. Para MBE usa valor absoluto."""
    df = dict_to_df(original)
    if metric == "|MBE|":
        out = df.xs("MBE", axis=1, level=1).abs()
    else:
        out = df.xs(metric, axis=1, level=1)
    # ordenar columnas por lista esperada
    cols = DATASETS_15 if set(out.columns)==set(DATASETS_15) else DATASETS_60
    return out[cols]

def compute_improvement(original: dict, adapted: dict, metric: str):
    """
    Calcula:
      - best_adapt: mejor adaptado por (sitio,dataset)
      - delta_pp: diferencia absoluta (puntos porcentuales) best - original
      - delta_rel: cambio relativo (%) = (best - original) / original * 100
    NOTA: valores negativos implican mejora (reducción del error).
    """
    base = original_metric_df(original, metric)
    best = best_adapted_df(adapted, metric)
    aligned = base.copy()
    aligned[:] = best.values
    delta_pp = aligned - base
    delta_rel = (delta_pp / base) * 100.0
    return base, aligned, delta_pp, delta_rel

# ----------------------------
# 3) HEATMAPS (mejora relativa) por resolución y métrica
# ----------------------------
def plot_heatmaps():
    plt.figure(figsize=(12, 9))

    # 15 min: MAE y RMSE
    base15_mae, best15_mae, dpp15_mae, drel15_mae = compute_improvement(orig_15, adapt_15, "MAE")
    base15_rmse, best15_rmse, dpp15_rmse, drel15_rmse = compute_improvement(orig_15, adapt_15, "RMSE")

    # 60 min: MAE y RMSE
    base60_mae, best60_mae, dpp60_mae, drel60_mae = compute_improvement(orig_60, adapt_60, "MAE")
    base60_rmse, best60_rmse, dpp60_rmse, drel60_rmse = compute_improvement(orig_60, adapt_60, "RMSE")

    # helper de heatmap
    def draw_hm(ax, data, title):
        # escala simétrica centrada en 0 para destacar mejoras (valores negativos)
        vmax = np.nanmax(np.abs(data.values))
        vmax = max(vmax, 1)  # evitar saturación extrema
        im = ax.imshow(data.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(data.columns, rotation=0, fontsize=10)
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(data.index, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        # anotaciones
        for i, j in product(range(data.shape[0]), range(data.shape[1])):
            val = data.iat[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=9, color="black")
        return im

    gs = plt.GridSpec(2, 2, hspace=0.25, wspace=0.15)

    ax1 = plt.subplot(gs[0,0])
    im1 = draw_hm(ax1, drel15_mae, "15 min · MAE · cambio relativo (%)\n(mejor adaptado vs original)")

    ax2 = plt.subplot(gs[0,1])
    im2 = draw_hm(ax2, drel15_rmse, "15 min · RMSE · cambio relativo (%)\n(mejor adaptado vs original)")

    ax3 = plt.subplot(gs[1,0])
    im3 = draw_hm(ax3, drel60_mae, "60 min · MAE · cambio relativo (%)\n(mejor adaptado vs original)")

    ax4 = plt.subplot(gs[1,1])
    im4 = draw_hm(ax4, drel60_rmse, "60 min · RMSE · cambio relativo (%)\n(mejor adaptado vs original)")

    # barra de color compartida
    cbar = plt.colorbar(im4, ax=[ax1, ax2, ax3, ax4], fraction=0.02, pad=0.02)
    cbar.set_label("Cambio relativo (%)  —  negativo = mejora", fontsize=10)

    plt.suptitle("Impacto de la Site Adaptation en el Error (mejor variante por dataset y sitio)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

# ----------------------------
# 4) BARRAS comparativas (original vs mejor adaptado)
# ----------------------------
def plot_bars_resolution(resolution="15min", metric="MAE"):
    """
    Gráfico de barras por resolución y métrica.
    - resolution: "15min" o "60min"
    - metric: "MAE" | "RMSE" | "|MBE|" (para MBE se usa valor absoluto)
    """
    if resolution == "15min":
        original = orig_15
        adapted = adapt_15
        datasets = DATASETS_15
        title_res = "15 minutos"
    elif resolution == "60min":
        original = orig_60
        adapted = adapt_60
        datasets = DATASETS_60
        title_res = "60 minutos"
    else:
        raise ValueError("resolution debe ser '15min' o '60min'.")

    base, best, dpp, drel = compute_improvement(original, adapted, metric)

    # Barras por dataset (subplots en filas), sitios en eje x
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(12, 3.2*n_ds), sharex=True)

    if n_ds == 1:
        axes = [axes]

    x = np.arange(len(SITES))
    width = 0.35

    for ax, ds in zip(axes, datasets):
        bvals = base[ds].values
        avals = best[ds].values
        dpp_vals = dpp[ds].values
        drel_vals = drel[ds].values

        bars1 = ax.bar(x - width/2, bvals, width, label="Original", edgecolor="black", linewidth=0.7)
        bars2 = ax.bar(x + width/2, avals, width, label="Mejor Adaptado", edgecolor="black", linewidth=0.7)

        # Anotaciones: delta p.p. y delta %
        for i, (bx, axv, dpp_i, drel_i) in enumerate(zip(bvals, avals, dpp_vals, drel_vals)):
            y = max(bx, axv)
            ax.text(i + width/2, axv - 15.5, f"{dpp_i:+.1f} pp\n({drel_i:+.0f}%)",
                    ha="center", va="bottom", fontsize=12)

            # Flecha de mejora (si reduce error)
            if dpp_i < 0:
                ax.annotate("", xy=(i + width/2, axv), xytext=(i + width/2, bx),
                            arrowprops=dict(arrowstyle="-|>", lw=1.2))

        ax.set_title(f"{ds}", fontsize=12, fontweight="bold", pad=2)
        ax.set_ylabel(metric + " (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(SITES, fontsize=12)
    axes[0].legend(ncols=2, frameon=False, loc="upper right")

    plt.suptitle(f"{title_res} · {metric}: Original vs Mejor Site-Adapted", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

# ----------------------------
# 5) FIGURAS RESUMEN (opcional): tabla/fig con % de sitios que mejoran
# ----------------------------
def plot_summary_improvement():
    """
    Pequeño resumen: % de celdas (sitio,dataset) con mejora (drel<0) para MAE/RMSE y 15/60 min.
    """
    combos = [
        ("15min", orig_15, adapt_15),
        ("60min", orig_60, adapt_60),
    ]
    rows = []
    for label, original, adapted in combos:
        for metric in ["MAE", "RMSE"]:
            _, _, _, drel = compute_improvement(original, adapted, metric)
            total = drel.size
            improved = (drel.values < 0).sum()
            rows.append({"Resolución": label, "Métrica": metric,
                         "% celdas con mejora": 100.0 * improved / total})
    df = pd.DataFrame(rows)

    # Barra simple
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r['Resolución']} · {r['Métrica']}" for _, r in df.iterrows()]
    vals = df["% celdas con mejora"].values
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de combinaciones (sitio,dataset) con mejora")
    ax.set_title("Cobertura de mejora tras Site Adaptation", fontsize=12, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()


def plot_summary_improvement2():
    """
    Resumen de cobertura de mejora: % de celdas (sitio,dataset) con mejora (drel<0)
    para MBE, MAE y RMSE en resoluciones de 15 y 60 minutos.
    """
    combos = [
        ("15min", orig_15, adapt_15),
        ("60min", orig_60, adapt_60),
    ]
    rows = []
    for label, original, adapted in combos:
        for metric in ["MBE", "MAE", "RMSE"]:
            _, _, _, drel = compute_improvement(original, adapted, metric)
            total = drel.size
            improved = (drel.values < 0).sum()
            rows.append({
                "Resolución": label,
                "Métrica": metric,
                "% celdas con mejora": 100.0 * improved / total
            })
    df = pd.DataFrame(rows)

    palette = {"MBE": "#C44E52", "MAE": "#4C72B0", "RMSE": "#55A868"}

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['Resolución']} · {r['Métrica']}" for _, r in df.iterrows()]
    vals = df["% celdas con mejora"].values
    colors = [palette[m] for m in df["Métrica"]]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de combinaciones (sitio, dataset) con mejora")
    ax.set_title("Cobertura de mejora tras Site Adaptation", fontsize=12, fontweight="bold")

    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[m]) for m in palette]
    ax.legend(handles, palette.keys(), title="Métrica", loc="lower right")

    plt.tight_layout()
    plt.show()





# ----------------------------
# 6) OPCIONAL: MBE absoluto (tendencia) comparativa
# ----------------------------
def plot_mbe_abs_heatmaps():
    # |MBE|: cuanto más bajo mejor (cercano a 0)
    _, _, dpp15_mbe, drel15_mbe = compute_improvement(orig_15, adapt_15, "|MBE|")
    _, _, dpp60_mbe, drel60_mbe = compute_improvement(orig_60, adapt_60, "|MBE|")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title in [
        (axes[0], drel15_mbe, "15 min · |MBE| · cambio relativo (%)"),
        (axes[1], drel60_mbe, "60 min · |MBE| · cambio relativo (%)"),
    ]:
        vmax = np.nanmax(np.abs(data.values))
        vmax = max(vmax, 1)
        im = ax.imshow(data.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(data.columns, rotation=0)
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(data.index)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        for i, j in product(range(data.shape[0]), range(data.shape[1])):
            ax.text(j, i, f"{data.iat[i,j]:+.1f}%", ha="center", va="center", fontsize=9)
    cbar = plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Cambio relativo (%)  —  negativo = mejora", fontsize=10)
    plt.suptitle("Tendencia (|MBE|) tras Site Adaptation — mejor variante por dataset", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()




# ----------------------------
# 7) EJECUCIÓN: generar todas las figuras
# ----------------------------
if __name__ == "__main__":
    # Heatmaps principales (MAE y RMSE)
    #plot_heatmaps()

    # Barras comparativas (elige las que quieras incluir en la tesis)
    # plt.figure()
    # plot_bars_resolution("15min", "MAE")
    # plt.show()
    
    plt.figure()
    plot_bars_resolution("15min", "RMSE")
    plt.show(block=False)

    plt.figure()
    plot_bars_resolution("60min", "RMSE")
    plt.show(block=False)

    #plot_bars_resolution("60min", "MAE")

    #plt.figure()
    #plot_bars_resolution("60min", "RMSE")
    #plt.show()
    # Resumen de cobertura de mejora

    # plt.figure()
    # plot_summary_improvement()
    # plt.show()

    # plt.figure()
    # plot_summary_improvement2()
    # plt.show()

    # # Opcional: |MBE|

    # plt.figure()
    # plot_mbe_abs_heatmaps()

    # plt.show()
