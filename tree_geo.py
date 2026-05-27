"""Эксперимент: одно решающее дерево на (lat, lon) → price_per_sqm.

Идея: дерево разобьёт географию Бишкека на прямоугольные зоны со своей
медианой $/м². Получим дискретное поле, которое можно подать в линейку как
OHE категориальный признак `geo_leaf` (альтернатива KMeans-кластеру).

Дерево намеренно подрезано (`max_leaf_nodes=10`, `min_samples_leaf=150`) — нам
нужны крупные стабильные зоны, а не overfit на отдельные адреса.

Сохраняет:
- figs/24_tree_geo_map.png — цветовая карта предсказаний.
- figs/25_tree_geo_structure.png — структура самого дерева.
- geo_leaf_meta.json — медианы $/м² по листьям + параметры дерева.
- tree_geo.joblib — сериализованное дерево (для применения на тесте).
- Секция §14 в report.md под рубрикой «Подумать потом».
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree

sns.set_theme(style="whitegrid")
FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)

MAX_LEAF_NODES = 10
MIN_SAMPLES_LEAF = 150


def load() -> pd.DataFrame:
    df = pd.read_csv("train_filled.csv")
    df = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()
    df["ppsqm"] = df["usd_price"] / df["area_total"]
    lo, hi = df["ppsqm"].quantile([0.01, 0.99])
    df = df[(df["ppsqm"] >= lo) & (df["ppsqm"] <= hi)].copy()
    return df.reset_index(drop=True)


def fit_tree(df: pd.DataFrame) -> tuple[DecisionTreeRegressor, dict]:
    X = df[["lat", "lon"]].values
    y = df["ppsqm"].values

    tree = DecisionTreeRegressor(
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=0,
    )
    tree.fit(X, y)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_r2 = cross_val_score(
        DecisionTreeRegressor(max_leaf_nodes=MAX_LEAF_NODES,
                              min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0),
        X, y, cv=cv, scoring="r2", n_jobs=-1)

    pred = tree.predict(X)
    train_r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))

    info = {
        "n_leaves": int(tree.get_n_leaves()),
        "depth": int(tree.get_depth()),
        "train_r2": round(train_r2, 4),
        "cv_r2_mean": round(float(cv_r2.mean()), 4),
        "cv_r2_std": round(float(cv_r2.std()), 4),
        "n_train": int(len(df)),
    }
    return tree, info


def plot_map(df: pd.DataFrame, tree: DecisionTreeRegressor, info: dict) -> None:
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    pad_lat = (lat_max - lat_min) * 0.03
    pad_lon = (lon_max - lon_min) * 0.03
    lat_grid = np.linspace(lat_min - pad_lat, lat_max + pad_lat, 400)
    lon_grid = np.linspace(lon_min - pad_lon, lon_max + pad_lon, 400)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    grid_X = np.column_stack([LAT.ravel(), LON.ravel()])
    grid_pred = tree.predict(grid_X).reshape(LAT.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1) heatmap предсказаний
    im = axes[0].pcolormesh(LON, LAT, grid_pred, shading="auto", cmap="viridis")
    axes[0].scatter(df["lon"], df["lat"], c="white", s=2, alpha=0.15, linewidths=0)
    plt.colorbar(im, ax=axes[0], label="predicted $/м²")
    axes[0].set_xlabel("lon"); axes[0].set_ylabel("lat")
    axes[0].set_title(f"Дерево на (lat, lon) → $/м²\n"
                      f"{info['n_leaves']} листьев, depth={info['depth']}, "
                      f"train R²={info['train_r2']}, CV R²={info['cv_r2_mean']}")

    # 2) точки, окрашенные по leaf
    leaves = tree.apply(df[["lat", "lon"]].values)
    n_leaves = info["n_leaves"]
    cmap_leaf = plt.get_cmap("tab10", n_leaves)
    sc = axes[1].scatter(df["lon"], df["lat"], c=leaves, s=6, alpha=0.7,
                         cmap=cmap_leaf, edgecolors="none")
    plt.colorbar(sc, ax=axes[1], label="leaf id", ticks=np.unique(leaves))
    axes[1].set_xlabel("lon"); axes[1].set_ylabel("lat")
    axes[1].set_title("Точки, окрашенные по leaf id (категория для OHE)")

    fig.tight_layout()
    fig.savefig(FIGS / "24_tree_geo_map.png", dpi=110)
    plt.close(fig)


def plot_tree_structure(tree: DecisionTreeRegressor) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree, feature_names=["lat", "lon"], filled=True, rounded=True,
              precision=0, ax=ax, fontsize=9, impurity=False)
    ax.set_title("Структура дерева (значение в листе — медианная $/м²)")
    fig.tight_layout()
    fig.savefig(FIGS / "25_tree_geo_structure.png", dpi=110)
    plt.close(fig)


def leaf_summary(df: pd.DataFrame, tree: DecisionTreeRegressor) -> pd.DataFrame:
    df = df.copy()
    df["leaf"] = tree.apply(df[["lat", "lon"]].values)
    summary = df.groupby("leaf").agg(
        n=("ppsqm", "size"),
        median_ppsqm=("ppsqm", "median"),
        mean_ppsqm=("ppsqm", "mean"),
        std_ppsqm=("ppsqm", "std"),
        lat_mean=("lat", "mean"),
        lon_mean=("lon", "mean"),
    ).round(1).sort_values("median_ppsqm")
    return summary


def update_report(info: dict, summary: pd.DataFrame) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker = "## 14. Подумать потом: дерево на координатах"

    sum_lines = ["| leaf | n | median $/м² | mean $/м² | std | lat_mean | lon_mean |",
                 "|---:|---:|---:|---:|---:|---:|---:|"]
    for leaf, row in summary.iterrows():
        sum_lines.append(
            f"| {leaf} | {int(row['n'])} | {row['median_ppsqm']:.0f} | "
            f"{row['mean_ppsqm']:.0f} | {row['std_ppsqm']:.0f} | "
            f"{row['lat_mean']:.4f} | {row['lon_mean']:.4f} |"
        )

    spread = float(summary["median_ppsqm"].max() - summary["median_ppsqm"].min())

    block = (
        f"\n---\n\n{marker}\n\n"
        "**Статус:** эксперимент, не интегрирован в основной пайплайн. "
        "Скрипт: [`tree_geo.py`](tree_geo.py).\n\n"
        "### 14.1 Что сделали\n\n"
        "Обучили одно решающее дерево **только на `(lat, lon)`** для предсказания "
        f"`price_per_sqm`. Дерево подрезали: `max_leaf_nodes={MAX_LEAF_NODES}`, "
        f"`min_samples_leaf={MIN_SAMPLES_LEAF}` — нужны крупные стабильные зоны "
        "вместо overfit на единичные адреса.\n\n"
        f"- Получили {info['n_leaves']} листьев, глубина {info['depth']}.\n"
        f"- Train R² = {info['train_r2']}, "
        f"CV R² (5-fold) = {info['cv_r2_mean']} ± {info['cv_r2_std']}.\n"
        f"- Разброс медианных $/м² между листьями: **${spread:.0f}/м²**.\n\n"
        "### 14.2 Цветовая карта\n\n"
        "![Tree geo map](figs/24_tree_geo_map.png)\n\n"
        "Слева — непрерывное поле предсказанных $/м² на сетке lat/lon. "
        "Видны прямоугольные блоки — каждая зона это один лист. Справа — "
        "реальные точки, окрашенные по `leaf_id` (это и есть новый категориальный "
        "признак).\n\n"
        "![Tree structure](figs/25_tree_geo_structure.png)\n\n"
        "Дерево разбивает Бишкек серией порогов по широте/долготе. Каждый "
        "лист содержит ≥150 объектов, в листе хранится средняя $/м².\n\n"
        "### 14.3 Лист → статистика\n\n"
        + "\n".join(sum_lines) + "\n\n"
        "### 14.4 Идеи к использованию (подумать потом)\n\n"
        "1. **OHE `geo_leaf` в линейку.** Готовый categorical с интерпретируемыми "
        "границами. Сравнить R² SGDRegressor с `geo_leaf` vs без него — потенциально "
        "конкурирует с `address` HashingVectorizer, но интерпретируемее.\n"
        "2. **`tree_price_ppsqm` как numeric признак.** Подать предсказание дерева "
        "как одно число (target encoding по геозоне). Минус — высокая корреляция "
        "с таргетом → утечка при naïve использовании. Безопасно только через "
        "out-of-fold predict.\n"
        "3. **Стек с `address`.** Возможно, дерево ловит «среднюю цену района», "
        "а адресный текст ловит «улицу/ЖК внутри района». Если так — оба сигнала "
        "сложатся в общую модель.\n"
        "4. **Геокластер для GroupKFold.** Текущая 5-fold CV не учитывает, что "
        "точки из одного ЖК встречаются в нескольких фолдах. `geo_leaf` как "
        "группа для `GroupKFold` даст более честную оценку генерализации.\n"
        "5. **Увеличить `max_leaf_nodes` до 20** — посмотреть, не растёт ли CV R². "
        "Сейчас train ≈ CV → дерево недообучено, есть запас. Но 10 листьев "
        "удобнее для отчётности.\n\n"
        "**Сохранённые артефакты для будущих экспериментов:**\n"
        "- [`tree_geo.joblib`](tree_geo.joblib) — обученное дерево.\n"
        "- [`geo_leaf_meta.json`](geo_leaf_meta.json) — параметры и медианы по листьям.\n"
        "- На тесте: `tree.apply(test[['lat','lon']])` → leaf_id, дальше OHE.\n"
    )

    if marker in text:
        text = text.split(marker)[0].rstrip()
        if text.endswith("---"):
            text = text[:-3].rstrip()
        text += "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    rpt.write_text(text, encoding="utf-8")


def main() -> None:
    df = load()
    print(f"loaded {len(df)} rows")

    tree, info = fit_tree(df)
    print(f"Tree: {info['n_leaves']} leaves, depth={info['depth']}")
    print(f"Train R² = {info['train_r2']}, CV R² = {info['cv_r2_mean']} ± {info['cv_r2_std']}")

    plot_map(df, tree, info)
    plot_tree_structure(tree)
    summary = leaf_summary(df, tree)
    print("\nLeaf summary:")
    print(summary)

    joblib.dump(tree, "tree_geo.joblib")
    meta = {
        "params": {
            "max_leaf_nodes": MAX_LEAF_NODES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
        },
        "info": info,
        "leaf_stats": {
            int(leaf): {
                "n": int(row["n"]),
                "median_ppsqm": float(row["median_ppsqm"]),
                "mean_ppsqm": float(row["mean_ppsqm"]),
                "std_ppsqm": float(row["std_ppsqm"]),
            }
            for leaf, row in summary.iterrows()
        },
    }
    Path("geo_leaf_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved tree_geo.joblib + geo_leaf_meta.json")

    update_report(info, summary)
    print("Report updated with §14.")


if __name__ == "__main__":
    main()
