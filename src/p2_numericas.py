import itertools

import numpy as np
import pandas as pd

from config import get_output_dir
from data_loader import load_data

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

# No promediar
EXCLUIR = {"hour", "day_of_week", "latitude", "longitude", "lanes"}

def get_numerics(df: pd.DataFrame) -> list[str]:
    """Numéricas no binarias y fuera de EXCLUIR."""
    cols = df.select_dtypes(include="number").columns
    return [c for c in cols if df[c].nunique() > 2 and c not in EXCLUIR]

def check_copies(df: pd.DataFrame, cols: list[str]) -> dict[str, str]:
    """Identificar copias"""
    copias = {}
    for a, b in itertools.combinations(cols, 2):
        if b not in copias and df[a].equals(df[b]):
            copias[b] = a
    return copias

def asymetric_form(g: float) -> str:
    if abs(g) < 0.5:
        return "sim"
    if abs(g) < 1:
        return "asim mod" + ("(der)" if g > 0 else "(izq)")
    return "asim tot" + ("(der)" if g > 0 else "(izq)")

def get_statistics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    filas = []
    for c in cols:
        s = df[c].dropna()
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lie, lse = q1 - 3 * iqr, q3 + 3 * iqr
        media, std = s.mean(), s.std()
        out = (s < li) | (s > ls)
        out_ext = (s < lie) | (s > lse)
        sk = s.skew()
        filas.append({
            "variable": c,
            "n": len(s),
            "media": media,
            "mediana": med,
            "moda": s.mode().iloc[0],
            "desv_std": std,
            "varianza": s.var(),
            "coef_var_pct": std / media * 100 if media != 0 else np.nan,
            "minimo": s.min(),
            "Q1": q1,
            "Q3": q3,
            "maximo": s.max(),
            "IQR": iqr,
            "asimetria": sk,
            "curtosis_exceso": s.kurt(),
            "forma": asymetric_form(sk),
            "limite_inf": li,
            "limite_sup": ls,
            "outliers": int(out.sum()),
            "pct_outliers": out.mean() * 100,
            "extr_outliers": int(out_ext.sum()),
        })
    return pd.DataFrame(filas)

def main():
    out = get_output_dir("p02")
    df = load_data()
    cols = get_numerics(df)

    print(f"Variables numéricas analizadas: {len(cols)}")
    print(f"Excluidas: "
          f"{sorted(set(df.select_dtypes('number').columns) - set(cols))}")
    
    print()

    res = get_statistics(df, cols)

    print("\n>>> Tendencia central y dispersion")
    print(res[["variable", "n", "media", "mediana", "moda", "desv_std",
               "varianza", "coef_var_pct", "minimo", "maximo"]].to_string(index=False))

    print("\n>>> Cuartiles, forma y outliers")
    print(res[["variable", "Q1", "Q3", "IQR", "asimetria", "curtosis_exceso", "forma",
               "outliers", "pct_outliers", "extr_outliers"]].to_string(index=False))

    res.to_csv(out / "descriptiva_numericas.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()