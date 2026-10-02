import numpy as np
import pandas as pd

from config import get_output_dir
from data_loader import load_data

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 70)


KNOWN_DATATYPES = {
    "record_id": "identifier",
    "road_id": "identifier",
    "intersection_id": "identifier",
    
    "city_zone": "nominal",
    "road_type": "nominal",
    "peak_period": "nominal",
    "weather_condition": "nominal",
    
    "iot_sensor_health": "ordinal",
    "road_condition": "ordinal",
    "congestion_level": "ordinal",
    
    "is_weekend": "binary",
    "is_holiday": "binary",
    "rush_hour": "binary",
    "nearby_school": "binary",
    "nearby_hospital": "binary",
    "nearby_market": "binary",
    "construction_activity": "binary",
    "public_event": "binary",
    "accident_reported": "binary",
    "emergency_vehicle_detected": "binary",
    
    "hour": "cyclic_discrete",
    "day_of_week": "cyclic_discrete",
    
    "timestamp": "temporal",
    
    "latitude": "geografica",
    "longitude": "geografica",
}

CATEGORICOS = {"nominal", "ordinal", "binary"}
# máximo de valores en diccionario
MAX_VALORES_LISTADOS = 12   

def get_statistic_type(col: str, s: pd.Series) -> str:
    if col in KNOWN_DATATYPES:
        return KNOWN_DATATYPES[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return "temporal"
    if pd.api.types.is_numeric_dtype(s):
        no_nulos = s.dropna()
        if len(no_nulos) and (no_nulos % 1 == 0).all():
            return "discreta"
        return "continua"
    return "nominal"

def value_info(s: pd.Series, tipo: str) -> str:
    """Texto con los valores posibles (categorías con su %, o valores enteros)."""
    n_unicos = s.nunique(dropna=True)
    if tipo in ("identifier", "temporal", "continua", "geografica"):
        return ""
    
    # Solo categorias
    if n_unicos > MAX_VALORES_LISTADOS:
        return f"{n_unicos} valores distintos"
    value_count = s.value_counts(normalize=True, dropna=False).sort_index()
    return "; ".join(f"{k} ({v:.1%})" for k, v in value_count.items())

def create_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        s = df[col]
        tipo = get_statistic_type(col, s)
        es_num = pd.api.types.is_numeric_dtype(s)
        es_fecha = pd.api.types.is_datetime64_any_dtype(s)

        fila = {
            "columna": col,
            "dtype_pandas": str(s.dtype),
            "tipo_estadistico": tipo,
            "nulos": int(s.isna().sum()),
            "pct_nulos": round(s.isna().mean() * 100, 3),
            "valores_unicos": int(s.nunique(dropna=True)),
            "minimo": np.nan, "maximo": np.nan, "rango": np.nan, "media": np.nan,
        }
        if es_num:
            fila.update(minimo=s.min(), maximo=s.max(),
                        rango=s.max() - s.min(), media=round(s.mean(), 4))
        elif es_fecha:
            fila.update(minimo=str(s.min()), maximo=str(s.max()),
                        rango=str(s.max() - s.min()))
        else:
            # Para texto: primer y último valor alfabético (referencia rápida)
            fila.update(minimo=str(s.min()), maximo=str(s.max()))

        fila["categories"] = value_info(s, tipo)
        filas.append(fila)
    return pd.DataFrame(filas)

def main():
    out = get_output_dir("p02")
    df = load_data()

    print(f"DATASET: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print("*" * 80)

    # 1) Diccionario
    dic = create_dictionary(df)
    print("\n>>> Diccionario")
    print(dic.drop(columns=["categories"]).to_string(index=False))
    
    print("\n>>> Valores")
    for _, r in dic[dic["categories"] != ""].iterrows():
        print(f"  {r['columna']:<28} [{r['tipo_estadistico']}] {r['categories']}")
    
    dic.to_csv(out / "diccionario_datos.csv", index=False, encoding="utf-8-sig")

    # 3) Resumen por tipo estadístico
    print("\n>>> Cantidad de variables por tipo")
    print(dic["tipo_estadistico"].value_counts().to_string())

    print(f"\nArchivos guardados en: {out}")

if __name__ == "__main__":
    main()