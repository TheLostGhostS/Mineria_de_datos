import math

import numpy as np
import pandas as pd

from config import get_output_dir
from data_loader import load_data

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

GROUPED_VARIABLES = ["average_speed", "vehicle_count", "congestion_score", "queue_length"]

GROUP_COLUMNS = ["city_zone", "peak_period", "congestion_level"]
METRIC_COLUMNS = ["vehicle_count", "average_speed", "congestion_score", "queue_length"]

# Orden natural para variables ordinales (si no aparece, se ordena por nombre)
GROUP_ORDERS = {"congestion_level": ["Low", "Moderate", "High", "Severe"]}

def sturges_classes(n: int) -> int:
    """Número de clases según Sturges: k = 1 + log2(n)"""
    return math.ceil(1 + math.log2(n))

def build_grouped_table(s: pd.Series) -> tuple[pd.DataFrame, float]:
    """Tabla de frecuencias por intervalos de igual amplitud"""
    s = s.dropna()
    k = sturges_classes(len(s))
    edges = np.linspace(s.min(), s.max(), k + 1)
    width = edges[1] - edges[0]

    # Intervalos [L, U), el último cerrado en ambos extremos
    freq = pd.cut(s, bins=edges, include_lowest=True, right=False).value_counts(sort=False)
    # pd.cut con right=False deja fuera el máximo: se suma a la última clase
    freq.iloc[-1] += int((s == s.max()).sum())

    table = pd.DataFrame({
        "lower": edges[:-1],
        "upper": edges[1:],
        "freq": freq.values.astype(int),
    })
    table["class_mark"] = (table["lower"] + table["upper"]) / 2
    table["rel_freq"] = table["freq"] / table["freq"].sum()
    table["cum_freq"] = table["freq"].cumsum()
    table["cum_rel_freq"] = table["cum_freq"] / table["freq"].sum()
    return table, width

def grouped_stats(table: pd.DataFrame, width: float) -> dict:
    """Calcular media, varianza muestral, mediana y moda"""
    n = table["freq"].sum()
    f, x = table["freq"], table["class_mark"]

    mean = (f * x).sum() / n
    variance = (f * (x - mean) ** 2).sum() / (n - 1)
    
    # Formulas graciosas
    
    # Mediana
    i = int((table["cum_freq"] >= n / 2).idxmax())
    prev_cum = table["cum_freq"].iloc[i - 1] if i > 0 else 0
    median = table["lower"].iloc[i] + (n / 2 - prev_cum) / f.iloc[i] * width

    # Moda
    j = int(f.idxmax())
    d1 = f.iloc[j] - (f.iloc[j - 1] if j > 0 else 0)
    d2 = f.iloc[j] - (f.iloc[j + 1] if j < len(f) - 1 else 0)
    mode = table["lower"].iloc[j] + (d1 / (d1 + d2) * width if (d1 + d2) > 0 else width / 2)

    return {"mean": mean, "variance": variance, "std": math.sqrt(variance),
            "median": median, "mode": mode}

def exact_stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {"mean": s.mean(), "variance": s.var(), "std": s.std(),
            "median": s.median(), "mode": s.mode().iloc[0]}

def group_metrics(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    """Metricas descriptivas de cada variable dentro de cada grupo."""
    agg = df.groupby(group_col)[metrics].agg(["count", "mean", "median", "std", "min", "max"])
    agg.columns = [f"{var}_{stat}" for var, stat in agg.columns]
    
    # Ordenar correctamente
    if group_col in GROUP_ORDERS:
        order = [g for g in GROUP_ORDERS[group_col] if g in agg.index]
        agg = agg.reindex(order + [g for g in agg.index if g not in order])
    return agg

def main():
    out = get_output_dir("p02")
    df = load_data()

    for var in GROUPED_VARIABLES:
        if var not in df.columns:
            print(f"'{var}' no existe en el dataset")
            continue
        table, width = build_grouped_table(df[var])
        print("\n" + "*" * 80)
        print(f"Tabla agrupada: {var}  (n={len(df):,}, clases Sturges k={len(table)}, "
              f"amplitud={width:,.3f})")
        print("*" * 80)
        print(table.to_string(index=True))
        table.to_csv(out / f"grouped_freq_{var}.csv", index=False, encoding="utf-8-sig")

    metrics = [m for m in METRIC_COLUMNS if m in df.columns]
    for col in GROUP_COLUMNS:
        if col not in df.columns:
            print(f"'{col}' no existe en el dataset")
            continue
        gm = group_metrics(df, col, metrics)
        print(f">>> Metricas por {col}")
        print("*" * 80)
        
        shown = [c for c in gm.columns if c.endswith(("_mean", "_median", "_std"))]
        print(gm[["vehicle_count_count"] + shown].rename(
            columns={"vehicle_count_count": "n"}).to_string())
        gm.to_csv(out / f"group_metrics_{col}.csv", encoding="utf-8-sig")

    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()