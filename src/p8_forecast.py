import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

from config import get_output_dir
from data_loader import load_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

# agregar "congestion_score" tambien
TARGETS = ["vehicle_count"]
TEST_DAYS = 14
FUTURE_DAYS = 7
HW_PERIODS = [24, 168]

FUTURE_MODEL = "auto"
HOLIDAY_COLUMN = "is_holiday"
ACF_LAGS = 200

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#e34948"
MODEL_COLORS = {"Ingenuo semanal": GRAY, "Ingenuo diario": "#b9b8b2", "Regresion calendario": BLUE,
                "HW 24h": ORANGE, "HW 168h": AQUA}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.6,
})

def build_series(df: pd.DataFrame, target: str) -> pd.DataFrame:
    agg = {target: "mean"}
    if HOLIDAY_COLUMN in df.columns:
        agg[HOLIDAY_COLUMN] = "max"
    s = df.groupby("timestamp").agg(agg).rename(columns={target: "y", HOLIDAY_COLUMN: "is_holiday"})
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    missing = len(full) - len(s)
    s = s.reindex(full)
    if missing:
        s["y"] = s["y"].interpolate()
    s["is_holiday"] = s["is_holiday"].fillna(0).astype(int) if "is_holiday" in s else 0
    s.index.name = "timestamp"
    return s.asfreq("h")

def add_calendar(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    frame = frame.copy()
    frame["hour"] = frame.index.hour
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    frame["trend"] = (frame.index - origin) / pd.Timedelta(weeks=1)
    return frame

def seasonal_naive(train_y: pd.Series, n: int, period: int) -> np.ndarray:
    return np.tile(train_y.to_numpy()[-period:], int(np.ceil(n / period)))[:n]

def fit_regression(train: pd.DataFrame):
    return smf.ols("y ~ C(hour) * is_weekend + is_holiday + trend", data=train).fit()

def predict_regression(res, frame: pd.DataFrame) -> pd.DataFrame:
    sf = res.get_prediction(frame).summary_frame(alpha=0.05)
    return pd.DataFrame({"mean": sf["mean"].to_numpy(), "lower": sf["obs_ci_lower"].to_numpy(),
                         "upper": sf["obs_ci_upper"].to_numpy()}, index=frame.index)

def fit_hw(y: pd.Series, period: int):
    return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                                seasonal_periods=period, initialization_method="heuristic").fit()

def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "R2": r2_score(y_true, y_pred)}

def with_band(point: np.ndarray, index, resid_std: float) -> pd.DataFrame:
    return pd.DataFrame({"mean": point, "lower": point - 1.96 * resid_std,
                         "upper": point + 1.96 * resid_std}, index=index)

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def plot_overview(s, split_at, target, path):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(s.index, s["y"], color=BLUE, linewidth=1)
    ax.axvline(split_at, color=RED, linestyle="--")
    ax.text(split_at, ax.get_ylim()[1], " prueba ->", color=RED, va="top", fontsize=9)
    ax.set_ylabel(f"{target} (promedio por hora)")
    ax.set_title("Serie horaria completa")
    save(fig, path)

def plot_profile(s, target, path):
    prof = s.assign(hour=s.index.hour, wk=np.where(s.index.dayofweek >= 5, "Fin de semana", "Entre semana"))
    prof = prof.groupby(["wk", "hour"])["y"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col, color in zip(prof.columns, (BLUE, ORANGE)):
        ax.plot(prof.index, prof[col], color=color, marker="o", markersize=4, label=col)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hora del dia")
    ax.set_ylabel(f"{target} (promedio)")
    ax.set_title("Perfil promedio por hora")
    ax.legend()
    save(fig, path)
    return prof

def plot_decomposition(res, target, path):
    comps = [("Observada", res.observed), ("Tendencia", res.trend),
             ("Estacional", res.seasonal), ("Residuo", res.resid)]
    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)
    for ax, (name, ser) in zip(axes, comps):
        ax.plot(ser.index, ser.to_numpy(), color=BLUE, linewidth=0.9)
        ax.set_ylabel(name)
    axes[0].set_title(f"Descomposicion STL de {target}")
    save(fig, path)

def plot_acf(values, n, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    lags = np.arange(len(values))
    ax.vlines(lags, 0, values, color=BLUE, linewidth=1.2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ci = 1.96 / np.sqrt(n)
    ax.axhspan(-ci, ci, color=GRID, alpha=0.7)
    for lag in (24, 48, 72, 168):
        if lag < len(values):
            ax.axvline(lag, color=RED, linestyle=":", linewidth=1)
            ax.text(lag, ax.get_ylim()[1], f" {lag}", color=RED, va="top", fontsize=9)
    ax.set_xlabel("rezago (horas)")
    ax.set_ylabel("autocorrelacion")
    ax.set_title("Autocorrelacion de la serie de entrenamiento")
    save(fig, path)

def plot_test(test_y, preds, target, path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in ("Ingenuo semanal", "Regresion calendario", "HW 24h", "HW 168h"):
        if name in preds:
            ax.plot(test_y.index, preds[name], color=MODEL_COLORS[name], label=name, linewidth=1.4)
    ax.plot(test_y.index, test_y.to_numpy(), color=INK, label="Real", linewidth=2)
    ax.set_ylabel(f"{target} (promedio por hora)")
    ax.set_title(f"Pronostico vs. Real")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, path)

def plot_comparison(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = table.sort_values("RMSE")
    colors = [MODEL_COLORS.get(m, BLUE) for m in order.index]
    for ax, col in zip(axes, ("RMSE", "MAE")):
        ax.barh(order.index, order[col], color=colors, height=0.6)
        for y, v in enumerate(order[col]):
            ax.text(v, y, f" {v:,.1f}", va="center", fontsize=9, color=INK_2)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, order[col].max() * 1.18)
        ax.set_title(f"{col} en prueba")
    save(fig, path)

def plot_residuals(resid: pd.Series, best: str, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(resid.index, resid.to_numpy(), color=BLUE, linewidth=0.9)
    axes[0].axhline(0, color=RED, linewidth=1.2)
    axes[0].set_title(f"Residuos en prueba: {best}")
    axes[0].set_ylabel("real - pronostico")
    axes[0].tick_params(axis="x", rotation=30)
    vals = acf(resid.to_numpy(), nlags=72, fft=True)
    axes[1].vlines(range(len(vals)), 0, vals, color=BLUE, linewidth=1.2)
    ci = 1.96 / np.sqrt(len(resid))
    axes[1].axhspan(-ci, ci, color=GRID, alpha=0.7)
    axes[1].set_xlabel("rezago (horas)")
    axes[1].set_title("Autocorrelacion de residuos")
    save(fig, path)

def plot_future(s, fut: pd.DataFrame, best: str, target, path):
    hist = s["y"].iloc[-TEST_DAYS * 24:]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(hist.index, hist.to_numpy(), color=INK, linewidth=1.6, label="Observado")
    ax.fill_between(fut.index, fut["lower"], fut["upper"], color=BLUE, alpha=0.18, label="Intervalo 95%")
    ax.plot(fut.index, fut["mean"], color=BLUE, linewidth=2, label=f"Pronostico ({best})")
    ax.axvline(fut.index[0], color=RED, linestyle="--")
    ax.set_ylabel(f"{target} (promedio por hora)")
    ax.set_title(f"Pronostico a futuro: {FUTURE_DAYS} dias despues del ultimo dato")
    ax.legend(loc="upper left", ncol=3)
    save(fig, path)

def run_target(df: pd.DataFrame, target: str, out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s = build_series(df, target)
    origin = s.index[0]
    n_test, n_fut = TEST_DAYS * 24, FUTURE_DAYS * 24
    s_cal = add_calendar(s, origin)
    train, test = s_cal.iloc[:-n_test], s_cal.iloc[-n_test:]
    s.to_csv(out / "hourly_series.csv", encoding="utf-8-sig")

    print("\n")
    print(f"Promedio por hora de {target}")
    print("*" * 80)
    print(f"{len(s):,} horas ({s.index[0]} a {s.index[-1]}) | entrenamiento = {len(train):,}"
          f"({len(train) / 24:.0f} dias) | prueba = {len(test):,} h (14 dias)")
    print(f"Resumen: media = {s['y'].mean():,.1f}, minimo = {s['y'].min():,.1f}, maximo = {s['y'].max():,.1f}")

    plot_overview(s, test.index[0], target, out / "series_overview.png")
    prof = plot_profile(train, target, out / "profile_by_hour.png")
    print(f"\n>> Hora pico entre semana: {prof['Entre semana'].idxmax()}"
          f"({prof['Entre semana'].max():,.0f}); fin de semana: {prof['Fin de semana'].idxmax()}:00 "
          f"({prof['Fin de semana'].max():,.0f})")

    stl = STL(train["y"], period=24, robust=True).fit()
    var_r, var_s, var_t = stl.resid.var(), (stl.seasonal + stl.resid).var(), (stl.trend + stl.resid).var()
    print(f"\n>> Descomposicion stl: fuerza estacional = {max(0, 1 - var_r / var_s):.2f}, "
          f"fuerza de tendencia = {max(0, 1 - var_r / var_t):.2f}")
    plot_decomposition(stl, target, out / "decomposition.png")

    ac = acf(train["y"], nlags=ACF_LAGS, fft=True)
    print(f">> Autocorrelacion: rezago 24 = {ac[24]:.2f}, 48 = {ac[48]:.2f}, 168 = {ac[168]:.2f}")
    plot_acf(ac, len(train), out / "acf.png")
    for label, series in (("serie", train["y"]), ("serie con diferencia de 24 h", train["y"].diff(24).dropna())):
        stat, p = adfuller(series, autolag="AIC")[:2]
        print(f">> ADF ({label}): estadistico = {stat:.2f}, p = {p:.4f} "
              f"{'estacionaria' if p < 0.05 else 'NO estacionaria'} (alpha = 5%)")

    preds = {"Ingenuo diario": seasonal_naive(train["y"], n_test, 24),
             "Ingenuo semanal": seasonal_naive(train["y"], n_test, 168)}
    reg = fit_regression(train)
    preds["Regresion calendario"] = predict_regression(reg, test)["mean"].to_numpy()
    hw_models = {}
    for period in HW_PERIODS:
        name = f"HW {period}h"
        hw_models[name] = fit_hw(train["y"], period)
        preds[name] = hw_models[name].forecast(n_test).to_numpy()
        print(f"    {name}: alpha = {hw_models[name].params['smoothing_level']:.3f}, "
              f"gamma = {hw_models[name].params['smoothing_seasonal']:.3f}")
    print(f"    Regresion calendario: = {reg.rsquared:.3f}, "
          f"pendiente de tendencia = {reg.params['trend']:.2f} por semana (p = {reg.pvalues['trend']:.3g})")

    table = pd.DataFrame({name: metrics(test["y"], p) for name, p in preds.items()}).T.sort_values("RMSE")
    table.to_csv(out / "model_comparison.csv", encoding="utf-8-sig")
    fitted = [m for m in table.index if not m.startswith("Ingenuo")]
    best = fitted[0] if FUTURE_MODEL == "auto" or FUTURE_MODEL not in fitted else FUTURE_MODEL
    
    print("\n")
    print(f">> Comparacion de los dias de prueba")
    print("*" * 80)
    print(table.to_string())
    print(f"\nMejor modelo en general: {table.index[0]}")
    
    base_rmse = table.loc["Ingenuo semanal", "RMSE"]
    
    plot_test(test["y"], preds, target, out / "forecast_test.png")
    plot_comparison(table, out / "model_comparison.png")
    plot_residuals(pd.Series(test["y"].to_numpy() - preds[best], index=test.index), best,
                   out / "residuals_best.png")

    fut_index = pd.date_range(s.index[-1] + pd.Timedelta(hours=1), periods=n_fut, freq="h")
    fut_cal = add_calendar(pd.DataFrame({"is_holiday": 0}, index=fut_index), origin)
    future, bands = {}, {}
    reg_all = fit_regression(s_cal)
    bands["Regresion calendario"] = predict_regression(reg_all, fut_cal)
    for period in HW_PERIODS:
        name = f"HW {period}h"
        fit = fit_hw(s["y"], period)
        bands[name] = with_band(fit.forecast(n_fut).to_numpy(), fut_index, float(np.std(fit.resid)))
    for name, period in (("Ingenuo diario", 24), ("Ingenuo semanal", 168)):
        point = seasonal_naive(s["y"], n_fut, period)
        bands[name] = pd.DataFrame({"mean": point, "lower": np.nan, "upper": np.nan}, index=fut_index)
    out_future = pd.DataFrame({f"{n}": b["mean"] for n, b in bands.items()})
    out_future[f"{best}_lower95"], out_future[f"{best}_upper95"] = bands[best]["lower"], bands[best]["upper"]
    out_future.to_csv(out / "future_forecast.csv", encoding="utf-8-sig")
    plot_future(s, bands[best], best, target, out / "forecast_future.png")

    daily = bands[best]["mean"].resample("D").agg(["mean", "max"])
    
    print("\n")
    print(f">> Pronostico a futuro de 7 dias")
    print("*" * 80)
    print(daily.rename(columns={"mean": "promedio_dia", "max": "maximo_hora"}).to_string())

def main():
    out = get_output_dir("p08")
    df = load_data()
    if "timestamp" not in df.columns:
        raise SystemExit("No hay timestamp")
    for target in TARGETS:
        if target not in df.columns:
            print(f"'{target}' no se encuentra")
            continue
        run_target(df, target, out / target)
    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()