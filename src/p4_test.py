import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import SEED, get_output_dir
from data_loader import load_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4g}")

ALPHA = 0.05
SAMPLE_SIZE = 5000 # maximo recomendado para Shapiro-Wilk
QQ_SAMPLE = 1500 # puntos por grupo en el Q-Q plot
MIN_GROUP_SIZE = 30 # grupos mas pequeños se excluyen
SKEW_TOL = 0.5 # menor a esto es casi simetrica
KURT_TOL = 1.0 # menor a esto es casi normal

HYPOTHESES = [
    {"id": "H1", "group": "peak_period", "value": "congestion_score", "compare_both": True,
     "title": "Cambio de congestion segun el dia"},
    {"id": "H2", "group": "weather_condition", "value": "average_speed", "compare_both": False,
     "title": "Velocidad promedio segun el clima"},
    {"id": "H3", "group": "road_type", "value": "emission_estimate", "compare_both": False,
     "title": "Emisiones segun la via"},
    {"id": "H4", "group": "city_zone", "value": "queue_length", "compare_both": False,
     "title": "Longitud de cola segun la zona"},
    {"id": "H5", "group": "is_weekend", "value": "vehicle_count", "compare_both": True,
     "labels": {0: "Entre semana", 1: "Fin de semana"},
     "title": "Relacion entre cantidad de vehiculos y momento de la semana"},
    {"id": "H6", "group": "accident_reported", "value": "average_wait_time", "compare_both": False,
     "labels": {0: "Sin accidente", 1: "Con accidente"},
     "title": "Relacion entre tiempo de espera y accidentes"},
]

TEST_NAMES = {
    "anova": "ANOVA de un factor",
    "welch_anova": "ANOVA de Welch",
    "kruskal": "Kruskal-Wallis",
    "student_t": "Prueba t de Student",
    "welch_t": "Prueba t de Welch",
    "mannwhitney": "U de Mann-Whitney",
}
PARAMETRIC = {"anova", "welch_anova", "student_t", "welch_t"}

# Diagnosis
def check_normality(groups: dict) -> pd.DataFrame:
    rows = []
    for name, s in groups.items():
        sample = s.sample(min(SAMPLE_SIZE, len(s)), random_state=SEED)
        p = stats.shapiro(sample).pvalue
        skew, kurt = s.skew(), s.kurt()
        # se acepta normalidad si shapiro no rechaza 
        # o si la forma tiene poca asimetria y curtosis
        shape_ok = abs(skew) < SKEW_TOL and abs(kurt) < KURT_TOL
        rows.append({"group": name, "n": len(s), "shapiro_p": p, "skew": skew,
                     "excess_kurtosis": kurt, "shape_ok": shape_ok,
                     "normal": bool(p >= ALPHA or shape_ok)})
    return pd.DataFrame(rows)

def check_equal_variance(groups: dict) -> float:
    return stats.levene(*groups.values(), center="median").pvalue

def choose_test(k: int, normal: bool, equal_var: bool) -> str:
    if not normal:
        return "mannwhitney" if k == 2 else "kruskal"
    if k == 2:
        return "student_t" if equal_var else "welch_t"
    return "anova" if equal_var else "welch_anova"

# Pruebas
def eta_squared(arrays: list) -> float:
    allv = np.concatenate(arrays)
    grand = allv.mean()
    ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
    return ssb / ((allv - grand) ** 2).sum()

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled

def welch_anova(arrays: list) -> tuple:
    k = len(arrays)
    n = np.array([len(a) for a in arrays])
    m = np.array([a.mean() for a in arrays])
    v = np.array([a.var(ddof=1) for a in arrays])
    w = n / v
    mean_w = (w * m).sum() / w.sum()
    num = (w * (m - mean_w) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    den = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    f_stat = num / den
    df2 = (k ** 2 - 1) / (3 * tmp)
    return f_stat, stats.f.sf(f_stat, k - 1, df2)

def label_effect(kind: str, value: float) -> str:
    v = abs(value)
    cuts = {"eta2": (0.01, 0.06, 0.14), "epsilon2": (0.01, 0.06, 0.14),
            "d": (0.2, 0.5, 0.8), "rank_biserial": (0.1, 0.3, 0.5)}[kind]
    if v < cuts[0]:
        return "insignificante"
    return "pequeño" if v < cuts[1] else "mediano" if v < cuts[2] else "grande"

def run_test(test: str, arrays: list) -> dict:
    # No puntero a funcion
    if test == "anova":
        stat, p = stats.f_oneway(*arrays)
        eff, kind = eta_squared(arrays), "eta2"
        
    elif test == "welch_anova":
        stat, p = welch_anova(arrays)
        eff, kind = eta_squared(arrays), "eta2"
        
    elif test == "kruskal":
        stat, p = stats.kruskal(*arrays)
        eff, kind = stat / (sum(len(a) for a in arrays) - 1), "epsilon2"
        
    elif test in ("student_t", "welch_t"):
        res = stats.ttest_ind(arrays[0], arrays[1], equal_var=(test == "student_t"))
        stat, p = res.statistic, res.pvalue
        eff, kind = cohen_d(arrays[0], arrays[1]), "d"
        
    elif test == "mannwhitney":
        res = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        stat, p = res.statistic, res.pvalue
        eff, kind = 2 * res.statistic / (len(arrays[0]) * len(arrays[1])) - 1, "rank_biserial"
    
    else:
        raise ValueError(test)
    
    return {"test": test, "statistic": stat, "p_value": p, "effect_size": eff,
            "effect_kind": kind, "effect_label": label_effect(kind, eff)}

def pairwise(groups: dict, parametric: bool) -> pd.DataFrame:
    names = list(groups)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rows = []
    for a, b in pairs:
        x, y = groups[a].values, groups[b].values
        if parametric:
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            eff, kind = cohen_d(x, y), "d"
        else:
            u = stats.mannwhitneyu(x, y, alternative="two-sided")
            p = u.pvalue
            eff, kind = 2 * u.statistic / (len(x) * len(y)) - 1, "rank_biserial"
        rows.append({"group_a": a, "group_b": b, "n_a": len(x), "n_b": len(y),
                     "mean_a": x.mean(), "mean_b": y.mean(),
                     "median_a": np.median(x), "median_b": np.median(y),
                     "p_raw": p, "effect_size": eff, "effect_kind": kind,
                     "effect_label": label_effect(kind, eff)})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p_raw"] * len(pairs)).clip(upper=1.0)
    out["significant"] = out["p_bonferroni"] < ALPHA
    out["method"] = "t de Welch" if parametric else "Mann-Whitney"
    return out

# Grafica
def save_qq(groups: dict, title: str, path) -> None:
    k = len(groups)
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax, (name, s) in zip(axes.ravel(), groups.items()):
        sample = s.sample(min(QQ_SAMPLE, len(s)), random_state=SEED)
        (osm, osr), (slope, intercept, _) = stats.probplot(sample, dist="norm")
        ax.scatter(osm, osr, s=6, color="#2a78d6", alpha=0.5, edgecolors="none")
        ax.plot(osm, slope * osm + intercept, color="#e34948", linewidth=1.5)
        ax.set_title(str(name), fontsize=10, loc="left")
        ax.set_xlabel("cuantiles teoricos", fontsize=8)
        ax.set_ylabel("cuantiles observados", fontsize=8)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    fig.suptitle(f"Q-Q plot: {title}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

# main
def analyze(df: pd.DataFrame, h: dict, out) -> list:
    gcol, vcol = h["group"], h["value"]
    labels = h.get("labels", {})
    print("\n" + "*" * 80)
    print(f"{h['id']}: {h['title']}")
    print(f"variable = {vcol}, grupos = {gcol}")
    print()

    data = df[[gcol, vcol]].dropna()
    groups = {}
    for key, s in data.groupby(gcol)[vcol]:
        name = labels.get(key, str(key))
        if len(s) < MIN_GROUP_SIZE:
            print(f"Grupo '{name}' no tiene suficientes datos.")
            continue
        groups[name] = s.reset_index(drop=True)
    if len(groups) < 2:
        print("Menos de 2 grupos validos, se omite la hipotesis.")
        return []
    k = len(groups)

    print("\nDescripcion por grupo")
    for name, val in groups.items():
        print(f"{name}: "
              f"longitud es {len(val)} "
              f"media es {val.mean()}")

    norm = check_normality(groups)
    lev_p = check_equal_variance(groups)
    normal_all = bool(norm["normal"].all())
    equal_var = lev_p >= ALPHA
    print("\nDatos")
    print(norm.to_string(index=False))
    print(f"Varianzas {'iguales' if equal_var else 'diferentes'}")

    main_test = choose_test(k, normal_all, equal_var)
    reason = (f"{f'No t' if not normal_all else f'T'}odos los grupos son normales, "
              f"con varianzas {'iguales' if equal_var else 'distintas'}")
    print(f"\nSe eligio {TEST_NAMES[main_test]}  ({reason})")

    arrays = [s.values for s in groups.values()]
    tests = [("principal", run_test(main_test, arrays))]
    if h.get("compare_both"):
        # Si la prueba principal es no parametrica, corre la parametrica y viceversa
        other = choose_test(k, main_test not in PARAMETRIC, equal_var)
        tests.append(("comparacion", run_test(other, arrays)))

    rows = []
    for role, res in tests:
        sig = res["p_value"] < ALPHA
        print(f"\n{TEST_NAMES[res['test']]} [{role}]")
        print(f"    estadistico = {res['statistic']:,.4g} | p = {res['p_value']:.4g} "
              f"-> {'significativa' if sig else 'no significativa'} (alfa = {ALPHA})")
        print(f"    tamaño de efecto ({res['effect_kind']}) = {res['effect_size']:.4f} "
              f"-> {res['effect_label']}")
        rows.append({"hypothesis": h["id"], "group_col": gcol, "value_col": vcol,
                     "n_groups": k, "n_total": sum(len(a) for a in arrays),
                     "normal_all_groups": normal_all, "levene_p": lev_p, "role": role,
                     "test_name": TEST_NAMES[res["test"]],
                     "p_value": res["p_value"], "significant": sig,
                     "effect_kind": res["effect_kind"], "effect_size": res["effect_size"],
                     "effect_label": res["effect_label"]})

    main_res = tests[0][1]
    if k >= 3 and main_res["p_value"] < ALPHA:
        post = pairwise(groups, parametric=main_test in PARAMETRIC)
        print(f"\n Comparaciones por pares ({post['method'].iloc[0]}, Bonferroni, "
              f"{len(post)} pares)")
        print(post[["group_a", "group_b", "median_a", "median_b", "p_bonferroni",
                    "significant", "effect_label"]].to_string(index=False))
        post.to_csv(out / f"posthoc_{h['id']}.csv", index=False, encoding="utf-8-sig")

    save_qq(groups, f"{h['id']} {vcol} por {gcol}", out / f"qq_{h['id']}.png")
    return rows

def main():
    out = get_output_dir("p04")
    df = load_data()
    print(f"Datos: {len(df):,} filas, alfa = {ALPHA}")

    summary = []
    for h in HYPOTHESES:
        missing = [c for c in (h["group"], h["value"]) if c not in df.columns]
        if missing:
            print(f"\n{h['id']}: faltan columnas {missing}")
            continue
        summary.extend(analyze(df, h, out))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "test_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "*" * 80)
    print(summary_df[["hypothesis", "role", "test_name", "p_value", "significant",
                      "effect_kind", "effect_size", "effect_label"]].to_string(index=False))
    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()
 