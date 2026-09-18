
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ^\s*(#+)\s*(-+|\d+\..*)\n

ARCHIVO = "smart_city_traffic_mobility.csv"

SALIDA = Path("resultados_cityflow")
GRAFICAS = SALIDA / "graficas"

SALIDA.mkdir(exist_ok=True)
GRAFICAS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(ARCHIVO)

# Eliminar espacios de los nombres de columnas
df.columns = df.columns.str.strip()

# Eliminar filas completamente vacias
df = df.dropna(how="all")

# Eliminar duplicados
duplicados = df.duplicated().sum()
df = df.drop_duplicates()

# Intentar convertir columnas de texto a numeros
for col in df.select_dtypes(include="object").columns:
    convertido = pd.to_numeric(df[col], errors="coerce")

    if convertido.notna().mean() > 0.8:
        df[col] = convertido

# Intentar identificar fechas
for col in df.select_dtypes(include="object").columns:
    if any(word in col.lower() for word in
           ["date", "time", "timestamp"]):
        df[col] = pd.to_datetime(
            df[col], errors="coerce"
        )

# Identificar tipos de variables
numericas = df.select_dtypes(
    include=np.number
).columns.tolist()

categoricas = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

fechas = df.select_dtypes(
    include=["datetime"]
).columns.tolist()

print("Dimensiones:", df.shape)
print("Duplicados eliminados:", duplicados)
print("\nColumnas:")
print(df.dtypes)

# Estadisticas numericas
estadisticas = df[numericas].describe().T

estadisticas["mediana"] = df[numericas].median()
estadisticas["moda"] = df[numericas].mode().iloc[0]
estadisticas["varianza"] = df[numericas].var()
estadisticas["asimetria"] = df[numericas].skew()
estadisticas["rango"] = (
    df[numericas].max() - df[numericas].min()
)

estadisticas["rango_intercuartil"] = (
    df[numericas].quantile(0.75)
    - df[numericas].quantile(0.25)
)

estadisticas.to_csv(
    SALIDA / "estadistica_numerica.csv"
)

# Estadisticas categoricas
if categoricas:
    resumen_cat = []

    for col in categoricas:
        resumen_cat.append({
            "Variable": col,
            "Valores_unicos": df[col].nunique(),
            "Moda": df[col].mode().iloc[0]
                if not df[col].mode().empty else None,
            "Frecuencia_moda": df[col].value_counts().max()
                if not df[col].value_counts().empty else 0,
            "Valores_nulos": df[col].isna().sum()
        })

    resumen_cat = pd.DataFrame(resumen_cat)

    resumen_cat.to_csv(
        SALIDA / "estadistica_categorica.csv",
        index=False
    )

# Calidad de datos
calidad = pd.DataFrame({
    "Tipo": df.dtypes.astype(str),
    "Nulos": df.isna().sum(),
    "Porcentaje_nulos": df.isna().mean() * 100,
    "Unicos": df.nunique()
})

calidad.to_csv(SALIDA / "calidad_datos.csv")

# Posibles identificadores
identificadores = [
    col for col in df.columns
    if col.lower().endswith("_id")
    or col.lower() == "id"
]

print("\nPosibles identificadores:")
print(identificadores)

# Exportar columnas y tipos
diccionario = pd.DataFrame({
    "Variable": df.columns,
    "Tipo": df.dtypes.astype(str),
    "Valores_unicos": [
        df[col].nunique() for col in df.columns
    ],
    "Nulos": df.isna().sum().values
})

diccionario.to_csv(
    SALIDA / "diccionario_datos.csv",
    index=False
)

# Cada columna se representa como una variable.
# Las flechas indican posibles asociaciones
# estadisticas, no relaciones causales.

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")

variables = list(df.columns)
n = len(variables)

if n > 0:
    # Distribuir variables en una circunferencia
    angulos = np.linspace(0, 2 * np.pi, n,
                          endpoint=False)

    posiciones = {
        col: (
            0.5 + 0.38 * np.cos(a),
            0.5 + 0.38 * np.sin(a)
        )
        for col, a in zip(variables, angulos)
    }

    for col, (x, y) in posiciones.items():
        ax.text(
            x, y, col,
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightblue",
                edgecolor="black"
            ),
            fontsize=8
        )

    # Relacionar variables numericas
    if len(numericas) >= 2:
        corr = df[numericas].corr()

        for i, col1 in enumerate(numericas):
            for col2 in numericas[i + 1:]:
                if abs(corr.loc[col1, col2]) >= 0.5:
                    x1, y1 = posiciones[col1]
                    x2, y2 = posiciones[col2]

                    ax.annotate(
                        "",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="<->",
                            color="gray",
                            alpha=0.6
                        )
                    )

    ax.set_title(
        "Diagrama de variables y asociaciones",
        fontsize=14
    )

    plt.savefig(
        GRAFICAS / "diagrama_variables.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

# Agrupar por cada variable categorica.
# Para cada grupo calcular cantidad, media,
# mediana y desviacion estandar de variables numericas.

for cat in categoricas:
    if df[cat].nunique() > 50:
        continue

    if not numericas:
        continue

    agrupado = (
        df.groupby(cat, observed=True)[numericas]
        .agg(["count", "mean", "median", "std"])
    )

    agrupado.to_csv(
        SALIDA / f"agrupado_{cat}.csv"
    )

# Seleccionar variables con datos validos
num_plot = [
    col for col in numericas
    if df[col].notna().sum() > 0
]

cat_plot = [
    col for col in categoricas
    if 1 < df[col].nunique() <= 20
]
for col in num_plot:
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=col,
        kde=True,
        bins=30
    )

    plt.title(f"Distribucion de {col}")
    plt.tight_layout()

    plt.savefig(
        GRAFICAS / f"histograma_{col}.png",
        dpi=150
    )

    plt.close()
for col in num_plot:
    plt.figure(figsize=(8, 5))

    sns.boxplot(data=df, x=col)

    plt.title(f"Diagrama de caja: {col}")
    plt.tight_layout()

    plt.savefig(
        GRAFICAS / f"caja_{col}.png",
        dpi=150
    )

    plt.close()
if len(num_plot) >= 2:
    for i, x in enumerate(num_plot):
        for y in num_plot[i + 1:]:
            plt.figure(figsize=(8, 5))

            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                alpha=0.5
            )

            plt.title(f"{x} vs {y}")
            plt.tight_layout()

            plt.savefig(
                GRAFICAS / f"dispersion_{x}_{y}.png",
                dpi=150
            )

            plt.close()
for col in cat_plot:
    frecuencias = df[col].value_counts().head(10)

    plt.figure(figsize=(7, 7))

    frecuencias.plot.pie(
        autopct="%1.1f%%",
        ylabel=""
    )

    plt.title(f"Proporcion de {col}")
    plt.tight_layout()

    plt.savefig(
        GRAFICAS / f"pastel_{col}.png",
        dpi=150
    )

    plt.close()
if len(num_plot) >= 2:
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        df[num_plot].corr(),
        annot=True,
        cmap="coolwarm",
        center=0
    )

    plt.title("Matriz de correlacion")
    plt.tight_layout()

    plt.savefig(
        GRAFICAS / "correlacion.png",
        dpi=200
    )

    plt.close()
for cat in cat_plot:
    for num in num_plot:
        tabla = df.groupby(
            cat, observed=True
        )[num].mean().sort_values(ascending=False)

        plt.figure(figsize=(10, 5))

        tabla.head(15).plot.bar()

        plt.title(f"Media de {num} por {cat}")
        plt.ylabel(f"Media de {num}")
        plt.tight_layout()

        plt.savefig(
            GRAFICAS / f"barras_{cat}_{num}.png",
            dpi=150
        )

        plt.close()

with pd.ExcelWriter(
    SALIDA / "reporte_cityflow.xlsx"
) as writer:

    calidad.to_excel(
        writer, sheet_name="Calidad"
    )

    diccionario.to_excel(
        writer, sheet_name="Variables",
        index=False
    )

    if not estadisticas.empty:
        estadisticas.to_excel(
            writer, sheet_name="Estadistica"
        )

    if categoricas:
        resumen_cat.to_excel(
            writer, sheet_name="Categoricas",
            index=False
        )

print("\nAnalisis terminado.")
print("Resultados guardados en:", SALIDA.resolve())