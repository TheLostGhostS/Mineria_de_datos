from pathlib import Path

DATA_FILE = "../smart_city_traffic_mobility.csv"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# --- Reproducibilidad (para muestreos, KNN, K-Means, etc.) ---
SEED = 42

def get_output_dir(practica: str) -> Path:
    """Carpeta de salida"""
    carpeta = OUTPUT_DIR / practica
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta