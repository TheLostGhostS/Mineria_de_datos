import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from wordcloud import WordCloud

from config import SEED, get_output_dir
from data_loader import load_data


MAX_LEVELS = 20
MIN_PCT = 1.0
DOMINANT_PCT = 85.0
EXCLUDE_COLUMNS = []

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FONT_PATH = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
DPI = 150

def select_text_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de texto con pocos niveles"""
    cols = []
    for c in df.columns:
        s = df[c]
        if c in EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        if s.nunique(dropna=True) <= MAX_LEVELS:
            cols.append(c)
    return cols


def build_frequencies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts = []
    for c in cols:
        counts = df[c].value_counts()
        parts.append(pd.DataFrame({"variable": c, "value": counts.index.astype(str),
                                   "abs_freq": counts.values}))
    freq = pd.concat(parts, ignore_index=True)
    freq["pct"] = freq["abs_freq"] / len(df) * 100

    freq["status"] = "incluida"
    freq["reason"] = ""
    dominant = freq["pct"] > DOMINANT_PCT
    rare = freq["pct"] < MIN_PCT
    freq.loc[dominant, "status"] = "omitida"
    freq.loc[dominant, "reason"] = f"dominante (> {DOMINANT_PCT:g}%)"
    freq.loc[rare, "status"] = "omitida"
    freq.loc[rare, "reason"] = f"rara (< {MIN_PCT:g}%)"

    duplicated = freq["value"].duplicated(keep=False)
    freq["label"] = np.where(duplicated, freq["variable"] + ": " + freq["value"], freq["value"])
    return freq


def draw_cloud(freq: pd.DataFrame, out_dir) -> None:
    kept = freq[freq["status"] == "incluida"]
    variables = list(dict.fromkeys(kept["variable"]))
    color_of = {v: SERIES[i % len(SERIES)] for i, v in enumerate(variables)}
    label_color = dict(zip(kept["label"], kept["variable"].map(color_of)))
    
    frequencies = dict(zip(kept["label"], np.sqrt(kept["abs_freq"])))

    cloud = WordCloud(
        width=1800, height=900, background_color=SURFACE, font_path=FONT_PATH,
        prefer_horizontal=0.8,
        relative_scaling=1.0,
        min_font_size=10, max_font_size=170, margin=8, random_state=SEED,
        color_func=lambda word, **kw: label_color[word],
    ).generate_from_frequencies(frequencies)

    placed = {item[0][0] for item in cloud.layout_}
    missing = set(frequencies) - placed
    if missing:
        print(f"No se pudo poner: {sorted(missing)}")

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cloud.to_array(), interpolation="bilinear")
    ax.axis("off")

    handles = [Patch(facecolor=color_of[v], label=v) for v in variables]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.045))

    omitted = freq[freq["status"] == "omitida"]
    
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.14)
    path = out_dir / "wordcloud.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Nube guardada en: {path}")


def main():
    out = get_output_dir("p09")
    df = load_data()
    cols = select_text_columns(df)

    print(f"Columnas de texto usadas ({len(cols)}): {', '.join(cols)}")
    print(f"Se omiten valores > {DOMINANT_PCT:g}% y < {MIN_PCT:g}%")
    print("*" * 80)

    freq = build_frequencies(df, cols)
    show = freq.sort_values("abs_freq", ascending=False)

    kept = show[show["status"] == "incluida"]
    print(f"\n>> Palabras en la nube ({len(kept)})")
    print(kept[["label", "variable", "abs_freq", "pct"]].assign(pct=lambda d: d["pct"].round(2))
          .to_string(index=False))

    omitted = show[show["status"] == "omitida"]
    print(f"\n>> Palabras omitidas ({len(omitted)})")
    if len(omitted):
        print(omitted[["label", "variable", "abs_freq", "pct", "reason"]]
              .assign(pct=lambda d: d["pct"].round(2)).to_string(index=False))
    else:
        print("ninguna")

    freq.to_csv(out / "word_frequencies.csv", index=False, encoding="utf-8-sig")
    draw_cloud(freq, out)
    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()