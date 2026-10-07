import matplotlib

# solo guarda archivos
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import SEED, get_output_dir
from data_loader import load_data

# (Paleta de colores fueron generadas)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
BLUE = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDINAL = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])

def colors_for(col: str, levels: list) -> list:
    if col == "congestion_level" and len(levels) <= len(ORDINAL):
        return ORDINAL[:len(levels)]
    return [SERIES[i % len(SERIES)] for i in range(len(levels))]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})

# puntos que se dibujan en dispersion
SCATTER_SAMPLE = 10000   
DPI = 150

# Orden natural ordinales
ORDERS = {
    "congestion_level": ["Low", "Moderate", "High", "Severe"],
    "road_condition": ["Good", "Wet", "Poor"],
    "peak_period": ["Night", "Morning Peak", "Midday", "Evening Peak", "Off-Peak"],
}

# Listado de graficas
CHART_SPECS = [
    {"type": "pie", "x": "congestion_level", "title": "Distribucion de niveles de congestion"},
    {"type": "pie", "x": "road_type", "title": "Distribucion por tipo de via"},

    {"type": "bar", "x": "city_zone", "y": "congestion_score",
     "title": "Congestion promedio por zona"},
    {"type": "bar", "x": "weather_condition", "y": "average_speed",
     "title": "Velocidad promedio segun clima"},

    {"type": "hist", "x": "average_speed", "title": "Distribucion de la velocidad promedio"},
    {"type": "hist", "x": "vehicle_count", "title": "Distribucion del conteo de vehiculos"},
    {"type": "hist", "x": "congestion_score", "title": "Distribucion del puntaje de congestion"},
    {"type": "hist", "x": "queue_length", "log": True,
     "title": "Distribucion de la longitud de cola (escala log)"},

    {"type": "box", "x": "peak_period", "y": "congestion_score",
     "title": "Congestion por periodo del dia"},
    {"type": "box", "x": "congestion_level", "y": "average_speed",
     "title": "Velocidad promedio por nivel de congestion"},
    {"type": "box", "x": "road_type", "y": "vehicle_count",
     "title": "Conteo de vehiculos por tipo de via"},

    {"type": "scatter", "x": "vehicle_count", "y": "queue_length", "yscale": "symlog",
     "title": "Vehiculos vs. longitud de cola"},
    {"type": "scatter", "x": "traffic_density", "y": "congestion_score",
     "title": "Densidad de trafico vs. puntaje de congestion"},

    {"type": "line", "x": "hour", "y": "vehicle_count",
     "title": "Vehiculos promedio por hora del dia"},
    {"type": "line", "x": "hour", "y": "congestion_score", "by": "is_weekend",
     "labels": {0: "Entre semana", 1: "Fin de semana"},
     "title": "Congestion promedio por hora: entre semana vs. fin de semana"},

    {"type": "heatmap", "title": "Correlacion entre variables numericas"},
]

def space_words(name: str) -> str:
    return name.replace("_", " ")

def ordered_levels(df: pd.DataFrame, col: str) -> list:
    present = df[col].dropna().unique().tolist()
    if col in ORDERS:
        known = [v for v in ORDERS[col] if v in present]
        return known + [v for v in present if v not in known]
    if pd.api.types.is_numeric_dtype(df[col]):
        return sorted(present)
    return df[col].value_counts().index.tolist()

def missing_columns(df: pd.DataFrame, spec: dict) -> list:
    cols = [spec[k] for k in ("x", "y", "by") if k in spec]
    return [c for c in cols if c not in df.columns]



def draw_pie(df, spec):
    col = spec["x"]
    levels = ordered_levels(df, col)
    counts = df[col].value_counts().reindex(levels)
    colors = colors_for(col, levels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.grid(False)
    ax.pie(counts.values, colors=colors, startangle=90, counterclock=False,
           wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
           labels=None)
    ax.set_aspect("equal")
    total = counts.sum()
    labels = [f"{lv}: {v / total:.1%}" for lv, v in zip(levels, counts.values)]
    ax.legend(ax.patches, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(spec["title"])
    return fig

def draw_bar(df, spec):
    x, y = spec["x"], spec["y"]
    means = df.groupby(x)[y].mean()
    if x in ORDERS:
        means = means.reindex([v for v in ORDERS[x] if v in means.index])
    else:
        means = means.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(means) + 2))
    bars = ax.barh(means.index.astype(str), means.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {val:,.1f}",
                va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, means.max() * 1.12)
    ax.set_xlabel(f"{space_words(y)} (promedio)")
    ax.set_title(spec["title"])
    return fig

def draw_hist(df, spec):
    x = spec["x"]
    s = df[x].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    note = ""
    if spec.get("log"):
        positive = s[s > 0]
        note = f"  ({len(s) - len(positive):,} valores en 0 no se muestran)" if len(positive) < len(s) else ""
        bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 50)
        ax.hist(positive, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        ax.set_xscale("log")
    else:
        bins = int(np.ceil(1 + np.log2(len(s))))   # regla de Sturges, igual que en la practica 2
        ax.hist(s, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(s.median(), color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(s.median(), ax.get_ylim()[1], f" mediana = {s.median():,.1f}", va="top",
            color=INK_2, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(space_words(x) + note)
    ax.set_ylabel("frecuencia")
    ax.set_title(spec["title"])
    return fig

def draw_box(df, spec):
    x, y = spec["x"], spec["y"]
    levels = ordered_levels(df, x)
    data = [df.loc[df[x] == lv, y].dropna().values for lv in levels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=[str(lv) for lv in levels], patch_artist=True, widths=0.55,
                    medianprops={"color": INK, "linewidth": 2},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                    flierprops={"marker": "o", "markersize": 2, "markerfacecolor": INK_2,
                                "markeredgecolor": "none", "alpha": 0.25})
    for patch in bp["boxes"]:
        patch.set(facecolor="#9ec5f4", edgecolor=BLUE, linewidth=1.2)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(space_words(x))
    ax.set_ylabel(space_words(y))
    ax.set_title(spec["title"])
    plt.setp(ax.get_xticklabels(), rotation=20 if len(levels) > 4 else 0, ha="right" if len(levels) > 4 else "center")
    return fig

def draw_scatter(df, spec):
    x, y = spec["x"], spec["y"]
    sample = df[[x, y]].dropna().sample(min(SCATTER_SAMPLE, len(df)), random_state=SEED)
    r = df[[x, y]].corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(sample[x], sample[y], s=10, color=BLUE, alpha=0.35, edgecolors="none")
    if spec.get("yscale"):
        ax.set_yscale(spec["yscale"], linthresh=1) if spec["yscale"] == "symlog" else ax.set_yscale(spec["yscale"])
    ax.set_xlabel(space_words(x))
    ax.set_ylabel(space_words(y) + (" (escala symlog)" if spec.get("yscale") == "symlog" else ""))
    ax.set_title(spec["title"])
    ax.text(0.98, 0.04, f"r de Pearson = {r:.2f}\nmuestra: {len(sample):,} de {len(df):,}",
            transform=ax.transAxes, ha="right", va="bottom", color=INK_2, fontsize=9)
    return fig

def draw_line(df, spec):
    x, y = spec["x"], spec["y"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if "by" in spec:
        groups = sorted(df[spec["by"]].dropna().unique())
        for i, g in enumerate(groups):
            s = df[df[spec["by"]] == g].groupby(x)[y].mean()
            label = spec.get("labels", {}).get(g, str(g))
            ax.plot(s.index, s.values, color=SERIES[i], label=label)
        ax.legend(loc="upper left")
    else:
        s = df.groupby(x)[y].mean()
        ax.plot(s.index, s.values, color=BLUE)
        ax.scatter(s.index, s.values, s=24, color=BLUE, zorder=3)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel(space_words(x))
    ax.set_ylabel(space_words(y) + " (promedio)")
    ax.set_title(spec["title"])
    return fig

def draw_heatmap(df, spec):
    num = df.select_dtypes(include="number")
    is_int_id = [pd.api.types.is_integer_dtype(num[c]) and num[c].nunique() == len(num) for c in num.columns]
    num = num.loc[:, (num.nunique() > 2).values & ~np.array(is_int_id)]  # fuera binarias e ids enteros
    num = num.drop(columns=[c for c in ("hour", "day_of_week") if c in num.columns])
    # quita columnas identicas
    dup = [b for i, a in enumerate(num.columns) for b in num.columns[i + 1:] if num[a].equals(num[b])]
    num = num.drop(columns=sorted(set(dup)))
    corr = num.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(6.5, n * 0.5)))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(n), corr.columns)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n <= 22:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else INK_2)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlacion de Pearson")
    ax.set_title(spec["title"])
    return fig


DRAWERS = {"pie": draw_pie, "bar": draw_bar, "hist": draw_hist, "box": draw_box,
           "scatter": draw_scatter, "line": draw_line, "heatmap": draw_heatmap}

def main():
    out = get_output_dir("p03")
    df = load_data()
    
    print(f"Generando {len(CHART_SPECS)} graficas desde {len(df):,} filas")
    print("*" * 80)

    made, skipped = 0, 0
    for i, spec in enumerate(CHART_SPECS, start=1):
        kind = spec["type"]
        name = "_".join(str(spec[k]) for k in ("x", "y", "by") if k in spec) or "all"
        filename = f"{i:02d}_{kind}_{name}.png"

        miss = missing_columns(df, spec)
        if kind not in DRAWERS or miss:
            print(f"{filename}: se omite ({'tipo desconocido' if kind not in DRAWERS else f'faltan columnas {miss}'})")
            skipped += 1
            continue

        fig = DRAWERS[kind](df, spec)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{kind:<7}] {filename}")
        made += 1

    kinds = sorted({s["type"] for s in CHART_SPECS})
    print(f"\n{made} graficas generadas, {skipped} omitidas. Tipos distintos: {len(kinds)} ({', '.join(kinds)})")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()