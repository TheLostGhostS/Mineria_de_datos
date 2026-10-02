import sys
import pandas as pd
from config import DATA_FILE

def load_data(path=DATA_FILE) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"No se encontro el archivo: '{path}'")

    if "timestamp" in df.columns:
        # Formato:
        # (mes/día/año hora:minuto)
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
        except ValueError:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    data = load_data()
    print(f"{data.shape[0]:,} filas x {data.shape[1]} columnas")
    print(data.head())