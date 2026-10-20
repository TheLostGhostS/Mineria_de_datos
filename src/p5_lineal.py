import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan

from config import SEED, get_output_dir
from data_loader import load_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4g}")

TEST_SIZE = 0.20
PLOT_SAMPLE = 5000
NEAR_DUPLICATE_R = 0.98
TOP_PAIRS_SHOWN = 15

MODELS = [
    {"id": "M1", "x": "traffic_density", "y": "congestion_score",
     "title": "Densidad de trafico -> puntaje de congestion"},
    # congestion_score topa en 100
    # variante excluye esas filas
    {"id": "M1_uncapped", "x": "traffic_density", "y": "congestion_score",
     "where": "congestion_score < 100",
     "title": "Densidad de trafico -> puntaje de congestion (purgado)"},
    # log cambia a la variable 'y' con log(y+1)
    {"id": "M2", "x": "vehicle_count", "y": "queue_length",
     "title": "Conteo de vehiculos -> longitud de cola"},
    {"id": "M2_log", "x": "vehicle_count", "y": "queue_length", "log_y": True,
     "title": "Conteo de vehiculos -> log(1 + longitud de cola)"},
    # hipotesis: la cola depende de la demanda por carril (vehiculos / capacidad de la via)
    {"id": "M3", "x": "vehicles_per_lane", "y": "queue_length",
     "title": "Vehiculos por carril -> longitud de cola"},
    {"id": "M3_log", "x": "vehicles_per_lane", "y": "queue_length", "log_y": True,
     "title": "Vehiculos por carril -> log(1 + longitud de cola)"},
]

# Columnas derivadas (nombre: expresion de DataFrame.eval). Se crean despues del
# ranking de correlaciones para no contaminarlo.
DERIVED = {"vehicles_per_lane": "vehicle_count / lanes"}

# misma paleta
SURFACE, INK, INK_2, GRID, BLUE, RED = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})

# Pruebas de correlacion
def numeric_columns(df: pd.DataFrame) -> list:
    num = df.select_dtypes(include="number")
    cols = [c for c in num.columns
            if num[c].nunique() > 2
            and not (pd.api.types.is_integer_dtype(num[c]) and num[c].nunique() == len(num))]
    # Duplicadas
    dup = {b for i, a in enumerate(cols) for b in cols[i + 1:]
           if np.array_equal(num[a].to_numpy(), num[b].to_numpy())}
    if dup:
        print(f"Columnas duplicadas excluidas: {sorted(dup)}")
    return [c for c in cols if c not in dup]

def correlation_ranking(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = pearson.loc[a, b]
            rows.append({"var_a": a, "var_b": b, "pearson_r": r, "spearman_rho": spearman.loc[a, b],
                         "r_squared": r ** 2, "near_duplicate": abs(r) >= NEAR_DUPLICATE_R})
    out = pd.DataFrame(rows)
    return out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)

# Ajuste de modelo
def fit_model(df: pd.DataFrame, spec: dict) -> dict:
    x_col, y_col, log_y = spec["x"], spec["y"], spec.get("log_y", False)
    source = df.query(spec["where"]) if spec.get("where") else df   # filtro opcional de filas
    data = source[[x_col, y_col]].dropna()
    x = data[[x_col]].to_numpy()
    y_raw = data[y_col].to_numpy()
    y = np.log1p(y_raw) if log_y else y_raw

    x_tr, x_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        x, y, y_raw, test_size=TEST_SIZE, random_state=SEED)

    model = LinearRegression().fit(x_tr, y_tr)
    pred_tr, pred_te = model.predict(x_tr), model.predict(x_te)

    ols = sm.OLS(y_tr, sm.add_constant(x_tr)).fit()
    coef_table = pd.DataFrame({
        "term": ["intercept", x_col], "coef": ols.params, "std_err": ols.bse,
        "t": ols.tvalues, "p_value": ols.pvalues,
        "ci95_low": ols.conf_int()[:, 0], "ci95_high": ols.conf_int()[:, 1],
    })
    resid_te = y_te - pred_te
    bp_p = het_breuschpagan(ols.resid, sm.add_constant(x_tr))[1]

    metrics = {
        "model": spec["id"], "x": x_col, "y": y_col + (" (log1p)" if log_y else ""),
        "n_train": len(x_tr), "n_test": len(x_te),
        "intercept": model.intercept_, "slope": model.coef_[0],
        "r2_train": r2_score(y_tr, pred_tr), "r2_test": r2_score(y_te, pred_te),
        "rmse_test": np.sqrt(mean_squared_error(y_te, pred_te)),
        "mae_test": mean_absolute_error(y_te, pred_te),
        "pearson_r": np.corrcoef(x[:, 0], y)[0, 1],
        "spearman_rho": pd.Series(x[:, 0]).corr(pd.Series(y), method="spearman"),
        "slope_p_value": ols.pvalues[1], "f_p_value": ols.f_pvalue,
        "resid_skew": pd.Series(resid_te).skew(), "resid_excess_kurtosis": pd.Series(resid_te).kurt(),
        "breusch_pagan_p": bp_p,
    }
    if log_y:
        metrics["r2_test_original_scale"] = r2_score(raw_te, np.expm1(pred_te))
    else:
        metrics["r2_test_original_scale"] = metrics["r2_test"]

    return {"metrics": metrics, "coef_table": coef_table,
            "x_te": x_te[:, 0], "y_te": y_te, "pred_te": pred_te, "resid_te": resid_te,
            "model": model}

# Grafiquita
def plot_model(spec: dict, res: dict, path) -> None:
    m, log_y = res["metrics"], spec.get("log_y", False)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(res["x_te"]), min(PLOT_SAMPLE, len(res["x_te"])), replace=False)
    x, y, pred, resid = (res[k][idx] for k in ("x_te", "y_te", "pred_te", "resid_te"))
    y_label = f"log(1 + {spec['y']})" if log_y else spec["y"]
    dots = {"s": 9, "color": BLUE, "alpha": 0.3, "edgecolors": "none"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.scatter(x, y, **dots)
    xs = np.linspace(res["x_te"].min(), res["x_te"].max(), 100)
    ax1.plot(xs, res["model"].predict(xs.reshape(-1, 1)), color=RED, linewidth=2,
             label=f"y = {m['intercept']:.3g} + {m['slope']:.3g}·x")
    ax1.set_xlabel(spec["x"]); ax1.set_ylabel(y_label)
    ax1.set_title("Datos de prueba y recta ajustada"); ax1.legend(loc="upper left")

    ax2.scatter(pred, y, **dots)
    lims = [min(pred.min(), y.min()), max(pred.max(), y.max())]
    ax2.plot(lims, lims, color=RED, linewidth=1.5, linestyle="--", label="prediccion perfecta")
    ax2.set_xlabel("valor predicho"); ax2.set_ylabel("valor real")
    ax2.set_title("Predicho vs. real"); ax2.legend(loc="upper left")

    ax3.scatter(pred, resid, **dots)
    ax3.axhline(0, color=RED, linewidth=1.5)
    ax3.set_xlabel("valor predicho"); ax3.set_ylabel("residuo (real - predicho)")
    ax3.set_title("Residuos vs. predicho")

    ax4.hist(resid, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax4.axvline(0, color=RED, linewidth=1.5)
    ax4.set_xlabel("residuo"); ax4.set_ylabel("frecuencia")
    ax4.set_title("Distribucion de los residuos")

    fig.suptitle(f"{spec['id']}: {spec['title']}   |   R^2 prueba = {m['r2_test']:.3f}   "
                 f"(entrenamiento = {m['r2_train']:.3f})", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)

def main():
    out = get_output_dir("p05")
    df = load_data()
    
    print(f"Division {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}")
    print("*" * 80)

    # Correlacion
    cols = numeric_columns(df)
    ranking = correlation_ranking(df, cols)
    ranking.to_csv(out / "correlation_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"Primeros {TOP_PAIRS_SHOWN} pares de {len(ranking)} pares; hubo {int(ranking['near_duplicate'].sum())} pares casi identicos (|r| >= {NEAR_DUPLICATE_R})")
    print(ranking.head(TOP_PAIRS_SHOWN).to_string(index=False))

    for name, expr in DERIVED.items():
        try:
            df[name] = df.eval(expr)
            print(f"\nColumna derivada creada: {name} = {expr}")
        except Exception as e:
            print(f"\nNo se pudo crear {name} ({expr}): {e}")

    # Modelos
    summary = []
    for spec in MODELS:
        miss = [c for c in (spec["x"], spec["y"]) if c not in df.columns]
        if miss:
            print(f"Falta el par en {spec['id']} faltante: {miss}")
            continue
        res = fit_model(df, spec)
        m = res["metrics"]
        summary.append(m)

        print("\n" + "*" * 80)
        print(f"{spec['id']}: {spec['title']}")
        print()
        print(f"Ecuacion: {m['y']} = {m['intercept']:.4g} + {m['slope']:.4g} * {m['x']}")
        print(f"Correlacion: Pearson r = {m['pearson_r']:.3f} Spearman rho = {m['spearman_rho']:.3f}")
        print(f"R^2  entrenamiento = {m['r2_train']:.4f}  prueba = {m['r2_test']:.4f}"
              + (f"  prueba en escala original = {m['r2_test_original_scale']:.4f}" if spec.get("log_y") else ""))
        print(f"Error en prueba: RMSE = {m['rmse_test']:.4g}  MAE = {m['mae_test']:.4g}")
        print(res["coef_table"].to_string(index=False))
        print(f"Residuos (prueba): asimetria = {m['resid_skew']:.3f}, curtosis = {m['resid_excess_kurtosis']:.3f}, "
              f"Breusch-Pagan p = {m['breusch_pagan_p']:.3g} "
              f"({'varianza no constante' if m['breusch_pagan_p'] < 0.05 else 'varianza constante'})")

        res["coef_table"].to_csv(out / f"coefficients_{spec['id']}.csv", index=False, encoding="utf-8-sig")
        plot_model(spec, res, out / f"fig_{spec['id']}.png")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "model_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "*" * 80)
    print("Resumen general")
    print(summary_df[["model", "x", "y", "r2_train", "r2_test", "r2_test_original_scale",
                      "rmse_test", "mae_test", "pearson_r"]].to_string(index=False))
    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()