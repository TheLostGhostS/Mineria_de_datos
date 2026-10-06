import pandas as pd

from config import get_output_dir
from data_loader import load_data

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

MAX_TEXT_LEVELS = 20
MAX_NUMERIC_LEVELS = 24

ORDINAL_ORDERS = {
    "congestion_level": ["Low", "Moderate", "High", "Severe"],
    "road_condition": ["Good", "Wet", "Poor"],
}

RARE_LEVEL_PCT = 1.0

def select_categorical(df: pd.DataFrame) -> list[str]:
    """Detectar las columnas automaticamente"""
    cols = []
    for c in df.columns:
        s = df[c]
        n_levels = s.nunique(dropna=True)
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            is_integer = (s.dropna() % 1 == 0).all()
            if is_integer and n_levels <= MAX_NUMERIC_LEVELS:
                cols.append(c)
        elif n_levels <= MAX_TEXT_LEVELS:
            cols.append(c)
    return cols

def freq_table(s: pd.Series, name: str) -> tuple[pd.DataFrame, str]:
    """Tabla de frecuencias con acumuladas"""
    counts = s.value_counts(dropna=False)

    if name in ORDINAL_ORDERS:
        order = [v for v in ORDINAL_ORDERS[name] if v in counts.index]
        # Lo que falte
        order += [v for v in counts.index if v not in order]  
        counts = counts.reindex(order)
        order_kind = "ordinal"
    elif pd.api.types.is_numeric_dtype(s):
        counts = counts.sort_index()
        order_kind = "valor"
    else:
        # Default
        order_kind = "frecuencia"

    n = counts.sum()
    table = pd.DataFrame({
        "variable": name,
        "value": counts.index.astype(str),
        "abs_freq": counts.values,
        "rel_freq": counts.values / n,
        "cum_abs_freq": counts.cumsum().values,
        "cum_rel_freq": counts.cumsum().values / n,
    })
    return table, order_kind

def main():
    out = get_output_dir("p02")
    df = load_data()
    cols = select_categorical(df)

    print(f"Variables categóricas detectadas: {len(cols)}")
    print("*" * 80)

    tables, summary = [], []
    for c in cols:
        table, order_kind = freq_table(df[c], c)
        tables.append(table)

        # Moda y nivel
        by_freq = table.sort_values("abs_freq", ascending=False)
        mode_row, rare_row = by_freq.iloc[0], by_freq.iloc[-1]
        summary.append({
            "variable": c,
            "n_levels": len(table),
            "mode": mode_row["value"],
            "mode_abs": int(mode_row["abs_freq"]),
            "mode_pct": round(mode_row["rel_freq"] * 100, 2),
            "rarest_level": rare_row["value"],
            "rarest_pct": round(rare_row["rel_freq"] * 100, 3),
            "order_kind": order_kind,
        })

        shown = table.drop(columns="variable").copy()
        # Formato de porcentaje
        for col in ("rel_freq", "cum_rel_freq"):
            shown[col] = (shown[col] * 100).round(2).astype(str) + "%"
        print(f"\n>>> {c}  (orden: {order_kind}, moda: {mode_row['value']})")
        print(shown.to_string(index=False))

    freq_all = pd.concat(tables, ignore_index=True)
    summary_df = pd.DataFrame(summary)

    print("\n")
    print(">>> Moda por variable")
    print(summary_df.drop(columns="order_kind").to_string(index=False))

    freq_all.to_csv(out / "freq_tables.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out / "categorical_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nGuardado en: {out}")

if __name__ == "__main__":
    main()