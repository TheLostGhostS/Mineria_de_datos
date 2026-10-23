import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import SEED, get_output_dir
from data_loader import load_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

TARGET = "congestion_level"
LEVEL_ORDER = ["Low", "Moderate", "High", "Severe"]
FEATURES = ["vehicle_count", "average_speed", "traffic_density",
            "queue_length", "green_light_duration", "lanes"]
LOG_FEATURES = ["queue_length"] 
LEAKAGE_COLUMN = "congestion_score"

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  
K_VALUES = list(range(1, 32, 2))
WEIGHTS_OPTIONS = ["uniform", "distance"] 

# Paleta
SURFACE, INK, INK_2, GRID, BLUE, ORANGE, RED = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6", "#eb6834", "#e34948")
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f3f8fe", "#9ec5f4", "#2a78d6", "#104281"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})

def leakage_check(df: pd.DataFrame) -> pd.DataFrame | None:
    if LEAKAGE_COLUMN not in df.columns:
        return None
    table = (df.groupby(TARGET)[LEAKAGE_COLUMN].agg(["count", "min", "max", "mean"])
             .reindex([lv for lv in LEVEL_ORDER if lv in df[TARGET].unique()]))
    levels = list(table.index)
    overlaps = [table.loc[a, "max"] > table.loc[b, "min"] for a, b in zip(levels, levels[1:])]
    table["overlaps_next_level"] = overlaps + [False]
    print(f"\nRango de {LEAKAGE_COLUMN} por nivel")
    print(table.to_string())
    if not any(overlaps):
        print(f"No hay translape")
    else:
        print(f"Hay translape")
    return table

def log_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return X

def make_preprocessor() -> Pipeline:
    return Pipeline([("log", FunctionTransformer(log_features, validate=False)),
                     ("scale", StandardScaler())])

def make_model(k: int, weights: str) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()),
                     ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1))])

def search_k(X_sub: pd.DataFrame, y_sub: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
             n_classes: int) -> pd.DataFrame:
    k_max = max(K_VALUES)
    prep = make_preprocessor().fit(X_sub)
    z_sub, z_val = prep.transform(X_sub), prep.transform(X_val)

    dist, idx = NearestNeighbors(n_neighbors=k_max, n_jobs=-1).fit(z_sub).kneighbors(z_val)
    # (n_val, k_max, n_classes)
    onehot = np.eye(n_classes)[y_sub][idx]

    rows = []
    for weights in WEIGHTS_OPTIONS:
        w = np.ones_like(dist) if weights == "uniform" else 1.0 / (dist + 1e-9)
        # votos acumulados por vecino
        votes = np.cumsum(onehot * w[:, :, None], axis=1)     
        for k in K_VALUES:
            pred = votes[:, k - 1, :].argmax(axis=1)
            rows.append({"weights": weights, "k": k,
                         "accuracy": accuracy_score(y_val, pred),
                         "f1_macro": f1_score(y_val, pred, average="macro", zero_division=0)})
    return pd.DataFrame(rows)

def plot_k_search(results: pd.DataFrame, best: pd.Series, path) -> None:
    colors = {"uniform": BLUE, "distance": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, metric, title in zip(axes, ("f1_macro", "accuracy"), ("F1 macro", "Accuracy")):
        for weights, grp in results.groupby("weights"):
            ax.plot(grp["k"], grp[metric], color=colors.get(weights, INK_2), label=f"pesos: {weights}")
        ax.scatter([best["k"]], [best[metric]], s=70, color=RED, zorder=5,
                   label=f"mejor: k = {int(best['k'])} ({best['weights']})")
        ax.set_xlabel("k (numero de vecinos)")
        ax.set_ylabel(title + " en validacion")
        ax.set_title(f"{title} segun k")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

# ----------------------------------------------------------------------
# 5) Evaluacion
# ----------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, labels: list, path, title: str) -> None:
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.grid(False)
    im = ax.imshow(row_pct, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1%})", ha="center", va="center",
                    fontsize=9, color="white" if row_pct[i, j] > 0.5 else INK)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("nivel predicho")
    ax.set_ylabel("nivel real")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="% de cada nivel real (recall)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def main():
    out = get_output_dir("p06")
    df = load_data()

    # Filtrar
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"No se encontraron las siguientes columnas: {missing}")
        exit()
    unknown = sorted(set(df[TARGET].dropna().unique()) - set(LEVEL_ORDER))
    if unknown:
        print(f"Niveles desconocidos: {unknown}")
    labels = [lv for lv in LEVEL_ORDER if lv in df[TARGET].unique()]
    level_to_int = {lv: i for i, lv in enumerate(labels)}

    data = df[FEATURES + [TARGET]].dropna()
    data = data[data[TARGET].isin(labels)]
    X, y = data[FEATURES], data[TARGET].map(level_to_int).to_numpy()

    print("*" * 80)
    print(f"Datos: {len(data):,} filas | clases: {labels}")
    print(f"Variables explicativas: {FEATURES}  (log1p en {LOG_FEATURES})")

    # Fugas
    leak = leakage_check(df)
    if leak is not None:
        leak.to_csv(out / "leakage_check.csv", encoding="utf-8-sig")

    # Division estratificada
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    dist = pd.DataFrame({"train": pd.Series(y_tr).map(dict(enumerate(labels))).value_counts(normalize=True),
                         "test": pd.Series(y_te).map(dict(enumerate(labels))).value_counts(normalize=True)}
                        ).reindex(labels)
    print(f"\nDivision {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} estratificada "
          f"(entrenamiento = {len(X_tr):,}, prueba = {len(X_te):,})")
    print("Proporcion de cada nivel:")
    print((dist * 100).round(2).astype(str).add("%").to_string())

    # Busqueda de k con validacion dentro del entrenamiento
    X_sub, X_val, y_sub, y_val = train_test_split(X_tr, y_tr, test_size=VALIDATION_SIZE,
                                                  stratify=y_tr, random_state=SEED)
    print(f"\n>> Busqueda de k (entrenamiento = {len(X_sub):,} | validacion = {len(X_val):,})")
    t0 = time.perf_counter()
    results = search_k(X_sub, y_sub, X_val, y_val, n_classes=len(labels))
    print(f"    listo en {time.perf_counter() - t0:.1f} s")
    results.to_csv(out / "knn_k_search.csv", index=False, encoding="utf-8-sig")

    pivot = results.pivot(index="k", columns="weights", values="f1_macro")
    print("\nValidacion por k:")
    print(pivot.to_string())
    best = results.sort_values(["f1_macro", "k"], ascending=[False, False]).iloc[0]
    best_k, best_weights = int(best["k"]), best["weights"]
    print(f"\nMejor: k = {best_k}, pesos = {best_weights} "
          f"(validacion = {best['f1_macro']:.4f}, accuracy = {best['accuracy']:.4f})")
    plot_k_search(results, best, out / "knn_k_search.png")

    # Modelo final
    print("\n>> Producto final")
    t0 = time.perf_counter()
    model = make_model(best_k, best_weights).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"    entrenamiento + prediccion en {time.perf_counter() - t0:.1f} s")

    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
    metrics = {
        "k": best_k, "weights": best_weights, "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_te, pred, average="weighted", zero_division=0),
        "val_f1_macro": best["f1_macro"],
        "baseline_accuracy": accuracy_score(y_te, baseline),
        "baseline_f1_macro": f1_score(y_te, baseline, average="macro", zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "*" * 80)
    print(">> Resultados")
    print(f"Accuracy = {metrics['accuracy']:.4f} | Accuracy balanceada = {metrics['balanced_accuracy']:.4f}")
    print(f"F1 macro = {metrics['f1_macro']:.4f} | F1 ponderado = {metrics['f1_weighted']:.4f}")
    print(f"F1 macro en validacion = {metrics['val_f1_macro']:.4f} (si es parecido al de prueba, no hay sobreajuste al elegir k)")
    print(f"Modelo base: "
          f"accuracy = {metrics['baseline_accuracy']:.4f} | F1 macro = {metrics['baseline_f1_macro']:.4f}")

    report = pd.DataFrame(classification_report(y_te, pred, target_names=labels, output_dict=True,
                                                zero_division=0)).T
    report.to_csv(out / "classification_report.csv", encoding="utf-8-sig")
    print("\nReporte por clase:")
    print(report.to_string())

    cm = confusion_matrix(y_te, pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=[f"real_{lv}" for lv in labels], columns=[f"pred_{lv}" for lv in labels])
    cm_df.to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    print("\nMatriz de confusion (filas = real, columnas = predicho):")
    print(cm_df.to_string())
    plot_confusion(cm, labels, out / "confusion_matrix.png",
                   f"Matriz de confusion (KNN, k = {best_k}, pesos: {best_weights})")

    # Errores segun la distancia entre niveles (las clases son ordinales)
    gap = np.abs(y_te - pred)
    share = pd.Series(gap).value_counts(normalize=True).sort_index()
    print("\nDistancia entre nivel real y predicho:")
    for g, p in share.items():
        print(f"  {'acierto exacto' if g == 0 else f'error de {g} nivel(es)'}: {p:.2%}")
    errors = gap > 0
    if errors.any():
        print(f"  De los errores, {np.mean(gap[errors] == 1):.1%} son a un nivel vecino.")

    print(f"\nGuardado en: {out}")

if __name__ == "__main__":
    main()