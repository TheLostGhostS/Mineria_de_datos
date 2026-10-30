import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from config import SEED, get_output_dir
from data_loader import load_data

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

FEATURES = ["traffic_density", "average_speed", "queue_length", "green_light_duration"]
LOG_FEATURES = ["queue_length"]
ORDER_BY = "traffic_density"
K_FINAL = 4
K_RANGE = list(range(2, 11))
N_INIT = 10
SILHOUETTE_SAMPLE = 20000
STABILITY_SEEDS = 5
PLOT_SAMPLE = 5000


LEVEL_COLUMN = "congestion_level"
LEVEL_ORDER = ["Low", "Moderate", "High", "Severe"]
CONTEXT_COLUMNS = ["peak_period", "city_zone", "weather_condition", "is_weekend", "hour"]
EXTRA_DESCRIPTORS = ["vehicle_count", "lanes", "congestion_score"]   # promedios descriptivos (post-hoc)


SURFACE, INK, INK_2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6"
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
ORDINAL = ["#86b6ef", "#5598e7", "#256abf", "#104281"]          # Low -> Severe
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})

def preprocess(X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return StandardScaler().fit_transform(X)

def order_labels(labels: np.ndarray, values: pd.Series) -> np.ndarray:
    means = pd.Series(values.to_numpy()).groupby(labels).mean().sort_values()
    mapping = {old: new for new, old in enumerate(means.index)}
    return np.array([mapping[l] for l in labels])

def fit_kmeans(z: np.ndarray, k: int, seed: int = SEED) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(z)

def cramers_v(table: pd.DataFrame) -> float:
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))

def purity(table: pd.DataFrame) -> float:
    return float(table.max(axis=1).sum() / table.to_numpy().sum())

def k_selection(z: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        t0 = time.perf_counter()
        km = fit_kmeans(z, k)
        rows.append({
            "k": k, "inertia": km.inertia_,
            "silhouette": silhouette_score(z, km.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=SEED),
            "davies_bouldin": davies_bouldin_score(z, km.labels_),
        })
        print(f"    k = {k:>2} listo ({time.perf_counter() - t0:.1f} s)")
    return pd.DataFrame(rows)

def plot_k_selection(res: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    specs = [("inertia", "Inercia (codo)"),
             ("silhouette", "Coeficiente de silueta"),
             ("davies_bouldin", "Indice de Davies-Bouldin")]
    for ax, (col, title) in zip(axes, specs):
        ax.plot(res["k"], res[col], color=BLUE, marker="o", markersize=6)
        chosen = res[res["k"] == K_FINAL]
        ax.scatter(chosen["k"], chosen[col], s=110, color="#e34948", zorder=5, label=f"k = {K_FINAL} (elegido)")
        ax.set_title(f"{title}")
        ax.set_xlabel("k (numero de clusters)")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def plot_centroids(z_df: pd.DataFrame, labels: np.ndarray, path) -> pd.DataFrame:
    cent = z_df.groupby(labels).mean()
    cent.index = [f"C{i + 1}" for i in cent.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.grid(False)
    lim = max(1.0, np.abs(cent.to_numpy()).max())
    im = ax.imshow(cent.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(cent.shape[0]):
        for j in range(cent.shape[1]):
            v = cent.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(cent.shape[1]), cent.columns, rotation=20, ha="right")
    ax.set_yticks(range(cent.shape[0]), cent.index)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="desviaciones estandar respecto al promedio")
    ax.set_title("Centroides")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return cent

def plot_pca(z: np.ndarray, labels: np.ndarray, levels: pd.Series | None, path) -> PCA:
    pca = PCA(n_components=2, random_state=SEED).fit(z)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(z), min(PLOT_SAMPLE, len(z)), replace=False)
    p = pca.transform(z[idx])
    ncols = 2 if levels is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.6 * ncols, 5.4), sharex=True, sharey=True, squeeze=False)
    ax = axes[0, 0]
    for c in range(K_FINAL):
        m = labels[idx] == c
        ax.scatter(p[m, 0], p[m, 1], s=9, alpha=0.45, color=CLUSTER_COLORS[c], edgecolors="none",
                   label=f"C{c + 1}")
    ax.set_title("Clusters K-Means")
    ax.legend(markerscale=2.5, loc="best")
    if levels is not None:
        ax2 = axes[0, 1]
        lv = levels.to_numpy()[idx]
        for i, name in enumerate(LEVEL_ORDER):
            m = lv == name
            ax2.scatter(p[m, 0], p[m, 1], s=9, alpha=0.45, color=ORDINAL[i], edgecolors="none", label=name)
        ax2.set_title("Nivel de congestion")
        ax2.legend(markerscale=2.5, loc="best")
    ev = pca.explained_variance_ratio_
    for a in axes.ravel():
        a.set_xlabel(f"componente principal 1 ({ev[0]:.0%} de la varianza)")
    axes[0, 0].set_ylabel(f"componente principal 2 ({ev[1]:.0%})")
    fig.suptitle(f"Proyeccion PCA (muestra de {len(idx):,} puntos)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return pca

def plot_level_composition(pct: pd.DataFrame, sizes: pd.Series, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(pct) + 2))
    ax.grid(axis="y", visible=False)
    left = np.zeros(len(pct))
    ylabels = [f"{c}  (n = {sizes[c]:,})" for c in pct.index]
    for i, level in enumerate(pct.columns):
        vals = pct[level].to_numpy()
        ax.barh(ylabels, vals, left=left, color=ORDINAL[i], edgecolor=SURFACE, linewidth=2, label=level)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 4:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if i >= 2 else INK)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de observaciones")
    ax.set_title("Composicion de cluster por nivel de congestion")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def main():
    out = get_output_dir("p07")
    df = load_data()

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltante: {missing}")
    data = df.dropna(subset=FEATURES).reset_index(drop=True)
    X = data[FEATURES]
    z = preprocess(X)
    z_df = pd.DataFrame(z, columns=FEATURES)

    print(f"Observaciones: {len(data):,} | variables para agrupar: {FEATURES}")
    print(f"log1p en {LOG_FEATURES} + escalado estandar | k final = {K_FINAL} | semilla {SEED}")
    print("*" * 80)

    print("\n>> Silueta")
    sel = k_selection(z)
    sel.to_csv(out / "k_selection.csv", index=False, encoding="utf-8-sig")
    print()
    print(sel.to_string(index=False))
    best_sil = int(sel.loc[sel["silhouette"].idxmax(), "k"])
    
    print(f"\nMejor silueta: k = {best_sil}.")
    plot_k_selection(sel, out / "k_selection.png")

    km = fit_kmeans(z, K_FINAL)
    labels = order_labels(km.labels_, data[ORDER_BY])
    data["cluster"] = [f"C{l + 1}" for l in labels]
    sizes = data["cluster"].value_counts().sort_index()

    sil = silhouette_score(z, labels, sample_size=SILHOUETTE_SAMPLE, random_state=SEED)
    db = davies_bouldin_score(z, labels)
    aris = [adjusted_rand_score(labels, fit_kmeans(z, K_FINAL, SEED + s).labels_)
            for s in range(1, STABILITY_SEEDS + 1)]
    metrics = {"k": K_FINAL, "n": len(data), "inertia": km.inertia_, "silhouette": sil,
               "davies_bouldin": db, "stability_ari_mean": float(np.mean(aris)),
               "stability_ari_min": float(np.min(aris))}

    print(f">> Evaluacion del modelo k = {K_FINAL}")
    print("*" * 80)
    print("Tamaño de los clusters:")
    for c, n in sizes.items():
        print(f"  {c}: {n:>8,}  ({n / len(data):.1%})")
    print(f"\nInercia = {km.inertia_:,.1f}, Silueta (muestra de {SILHOUETTE_SAMPLE:,}) = {sil:.3f}"
          f", Davies-Bouldin = {db:.3f}")

    # Validacion externa con el nivel de congestion
    levels = None
    if LEVEL_COLUMN in data.columns:
        known = data[LEVEL_COLUMN].isin(LEVEL_ORDER)
        levels = data.loc[known, LEVEL_COLUMN]
        counts = pd.crosstab(data.loc[known, "cluster"], levels).reindex(columns=LEVEL_ORDER, fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        ari_ext = adjusted_rand_score(levels, data.loc[known, "cluster"])
        nmi = normalized_mutual_info_score(levels, data.loc[known, "cluster"])
        pur = purity(counts)
        metrics.update({"external_ari": ari_ext, "external_nmi": nmi, "external_purity": pur})
        counts.to_csv(out / "cluster_vs_level.csv", encoding="utf-8-sig")
        print(f"\n>> Validacion externa vs {LEVEL_COLUMN}")
        print("Conteos:")
        print(counts.to_string())
        print("\nPorcentaje dentro de cada cluster:")
        print(pct.round(1).to_string())
        print(f"\nRand ajustado = {ari_ext:.3f} | Informacion mutua normalizada = {nmi:.3f} | "
              f"Pureza = {pur:.3f}")
        plot_level_composition(pct, sizes, out / "cluster_vs_level.png")
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    extras = [c for c in EXTRA_DESCRIPTORS if c in data.columns]
    prof_cols = FEATURES + extras
    profile = data.groupby("cluster")[prof_cols].mean()
    profile.insert(0, "n", sizes)
    profile.insert(1, "pct", sizes / len(data) * 100)
    for c in LOG_FEATURES:
        profile[f"{c}_median"] = data.groupby("cluster")[c].median()
    profile.to_csv(out / "cluster_profiles.csv", encoding="utf-8-sig")
    print("\n>> Perfil de cluster")
    print(profile.to_string())

    cent = plot_centroids(z_df, labels, out / "centroids_heatmap.png")
    print("\nCentroides estandarizados (z):")
    print(cent.to_string())

    # Contexto: variables que no formaron los grupos
    print("\n>> Variables de contexto")
    assoc = []
    for col in [c for c in CONTEXT_COLUMNS if c in data.columns]:
        table = pd.crosstab(data["cluster"], data[col])
        assoc.append({"variable": col, "cramers_v": cramers_v(table)})
        (table.div(table.sum(axis=1), axis=0) * 100).to_csv(out / f"cluster_context_{col}.csv",
                                                              encoding="utf-8-sig")
    assoc_df = pd.DataFrame(assoc).sort_values("cramers_v", ascending=False)
    assoc_df.to_csv(out / "context_association.csv", index=False, encoding="utf-8-sig")
    print(assoc_df.to_string(index=False))
    for col in ("peak_period", "city_zone", "weather_condition"):
        if col in data.columns:
            t = pd.crosstab(data["cluster"], data[col], normalize="index") * 100
            print(f"\n% de cada cluster por {col}:")
            print(t.round(1).to_string())
    if "hour" in data.columns:
        print("\nHora promedio por cluster: " + ", ".join(
            f"{c} = {h:.1f}" for c, h in data.groupby("cluster")["hour"].mean().items()))

    plot_pca(z, labels, levels.reindex(data.index) if levels is not None else None, out / "pca_clusters.png")
    print(f"\nGuardado en: {out}")

if __name__ == "__main__":
    main()