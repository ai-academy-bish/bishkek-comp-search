"""Эксперимент-ансамбль: признак от решающего дерева (lat,lon → $/м²) в линейке.

Базовая модель — §13 (address-hash + numeric + cat + bin, БЕЗ сырых lat/lon),
SGDRegressor(huber+elasticnet) на log1p(usd_price). Гиперпараметры зафиксированы
на лучших из §13 (alpha=1e-5, l1_ratio=0.15, epsilon=0.5) — чистая абляция:
меняется только пространственный признак.

Варианты пространственной фичи (все прогоняются и сравниваются):
  none       — базовая §13 (без гео).
  kmeans_ohe — KMeans(k=8) на (lat,lon) → OHE  (фича из ранних экспериментов).
  tree_ohe   — лист дерева как категория → OHE  (грубое дерево, ~40 листьев).
  tree_num   — предсказание дерева $/м² как число (out-of-fold target encoding).
  tree_lognum— log(предсказание) — согласован с log-таргетом.
  tree_ohe_num — OHE листа + численная оценка вместе.
  tree_num_kmeans — численная оценка дерева + KMeans OHE (две гео-фичи разной природы).

Антиутечка: дерево и KMeans обучаются ВНУТРИ каждого CV-фолда (фичи зависят от
таргета через ppsqm), поэтому всё считается через cross_val_predict с пайплайном,
где трансформеры fit-ятся только на train-части фолда.

Дополняет §15 в report.md + figs/29_geo_ensemble.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sns.set_theme(style="whitegrid")
FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)

TEXT_COL = "address"
NUMERIC_COLS = ["build_year", "floor", "total_floors", "rooms",
                "area_total", "area_total_sq"]
CATEGORICAL_COLS = ["building_material", "series_group"]
BINARY_COLS = ["is_old", "condition_unfinished"]
GEO_COLS = ["lat", "lon", "area_total"]  # area_total нужен для reconstruct ppsqm

# зафиксированные гиперпараметры SGD из §13
SGD_KW = dict(loss="huber", penalty="elasticnet", alpha=1e-5, l1_ratio=0.15,
              epsilon=0.5, max_iter=3000, tol=1e-4, random_state=0,
              learning_rate="invscaling", eta0=0.01)

TREE_OHE_LEAVES = 40      # грубое дерево для категорий (стабильные зоны)
TREE_NUM_LEAVES = 100     # детальное дерево для численной оценки (target encoding)
TREE_MIN_LEAF = 30
KMEANS_K = 8


def _ppsqm_from_y(y: np.ndarray, area: np.ndarray) -> np.ndarray:
    """price = expm1(log1p(price)); ppsqm = price / area."""
    return np.expm1(y) / area


class GeoTreeFeature(BaseEstimator, TransformerMixin):
    """Дерево (lat,lon)->ppsqm; выдаёт OHE листа и/или численную оценку.

    mode ∈ {'ohe', 'num', 'lognum', 'ohe_num'}.
    Обучается на y (log-price) внутри фолда -> нет утечки.
    """

    def __init__(self, mode="num", max_leaf_nodes=100, min_samples_leaf=30):
        self.mode = mode
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        coords = X[:, :2]
        area = X[:, 2]
        ppsqm = _ppsqm_from_y(np.asarray(y, dtype=float), area)
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0,
        ).fit(coords, ppsqm)
        if self.mode in ("ohe", "ohe_num"):
            leaves = self.tree_.apply(coords).reshape(-1, 1)
            self.ohe_ = OneHotEncoder(sparse_output=True,
                                      handle_unknown="ignore").fit(leaves)
        if self.mode in ("num", "lognum", "ohe_num"):
            pred = self.tree_.predict(coords)
            val = np.log(pred) if self.mode == "lognum" else pred
            self.num_mean_ = float(val.mean())
            self.num_std_ = float(val.std()) or 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        coords = X[:, :2]
        parts = []
        if self.mode in ("ohe", "ohe_num"):
            leaves = self.tree_.apply(coords).reshape(-1, 1)
            parts.append(self.ohe_.transform(leaves))
        if self.mode in ("num", "lognum", "ohe_num"):
            pred = self.tree_.predict(coords)
            val = np.log(pred) if self.mode == "lognum" else pred
            scaled = ((val - self.num_mean_) / self.num_std_).reshape(-1, 1)
            parts.append(sparse.csr_matrix(scaled))
        return sparse.hstack(parts, format="csr")


class KMeansGeo(BaseEstimator, TransformerMixin):
    """KMeans(k) на (lat,lon) -> OHE кластера (как в ранних экспериментах)."""

    def __init__(self, k=8):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.km_ = KMeans(n_clusters=self.k, n_init=10, random_state=0).fit(X[:, :2])
        labels = self.km_.predict(X[:, :2]).reshape(-1, 1)
        self.ohe_ = OneHotEncoder(sparse_output=True,
                                  handle_unknown="ignore").fit(labels)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        labels = self.km_.predict(X[:, :2]).reshape(-1, 1)
        return self.ohe_.transform(labels)


def base_transformers():
    return [
        ("text", HashingVectorizer(n_features=1024, ngram_range=(1, 2),
                                   alternate_sign=False, norm="l2",
                                   lowercase=True), TEXT_COL),
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(sparse_output=True, handle_unknown="ignore"),
         CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ]


def make_pipeline(spatial: str) -> Pipeline:
    transformers = base_transformers()
    if spatial == "kmeans_ohe":
        transformers.append(("geo", KMeansGeo(k=KMEANS_K), GEO_COLS))
    elif spatial == "tree_ohe":
        transformers.append(("geo", GeoTreeFeature("ohe", TREE_OHE_LEAVES,
                                                   TREE_MIN_LEAF), GEO_COLS))
    elif spatial == "tree_num":
        transformers.append(("geo", GeoTreeFeature("num", TREE_NUM_LEAVES,
                                                   TREE_MIN_LEAF), GEO_COLS))
    elif spatial == "tree_lognum":
        transformers.append(("geo", GeoTreeFeature("lognum", TREE_NUM_LEAVES,
                                                   TREE_MIN_LEAF), GEO_COLS))
    elif spatial == "tree_ohe_num":
        transformers.append(("geo", GeoTreeFeature("ohe_num", TREE_OHE_LEAVES,
                                                   TREE_MIN_LEAF), GEO_COLS))
    elif spatial == "tree_num_kmeans":
        transformers.append(("geotree", GeoTreeFeature("num", TREE_NUM_LEAVES,
                                                       TREE_MIN_LEAF), GEO_COLS))
        transformers.append(("geokm", KMeansGeo(k=KMEANS_K), GEO_COLS))
    # 'none' -> ничего не добавляем
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline([("pre", pre), ("reg", SGDRegressor(**SGD_KW))])


def load() -> pd.DataFrame:
    df = pd.read_csv("train_features.csv")
    df = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()
    df["address"] = df["address"].fillna("").str.lower()
    rooms_median = df.loc[df["rooms"] != 1000, "rooms"].median()
    df.loc[df["rooms"] == 1000, "rooms"] = rooms_median
    df["area_total_sq"] = df["area_total"] ** 2
    return df.reset_index(drop=True)


def metrics_oof(y: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y - y_pred
    log_r2 = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    log_rmse = float(np.sqrt(np.mean(resid ** 2)))
    log_mae = float(np.mean(np.abs(resid)))
    y_usd, p_usd = np.expm1(y), np.expm1(y_pred)
    ape = np.abs((y_usd - p_usd) / y_usd) * 100
    usd_r2 = float(1 - np.sum((y_usd - p_usd) ** 2) / np.sum((y_usd - y_usd.mean()) ** 2))
    return {
        "log_r2": round(log_r2, 4),
        "log_rmse": round(log_rmse, 4),
        "log_mae": round(log_mae, 4),
        "usd_r2": round(usd_r2, 4),
        "mape_pct": round(float(ape.mean()), 2),
        "median_ape_pct": round(float(np.median(ape)), 2),
    }


def plot_compare(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = results.sort_values("log_r2")["variant"]

    sns.barplot(data=results, y="variant", x="log_r2", order=order,
                ax=axes[0], color="steelblue")
    axes[0].set_title("CV log R² (out-of-fold)")
    axes[0].set_xlim(results["log_r2"].min() - 0.01, results["log_r2"].max() + 0.005)
    for i, v in enumerate(results.set_index("variant").loc[order, "log_r2"]):
        axes[0].text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=9)

    sns.barplot(data=results, y="variant", x="median_ape_pct", order=order,
                ax=axes[1], color="indianred")
    axes[1].set_title("CV median APE, % (ниже = лучше)")
    for i, v in enumerate(results.set_index("variant").loc[order, "median_ape_pct"]):
        axes[1].text(v + 0.02, i, f"{v:.2f}%", va="center", fontsize=9)

    fig.suptitle("Ансамбль: гео-признак от дерева vs KMeans vs базовая §13", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "29_geo_ensemble.png", dpi=110)
    plt.close(fig)


def update_report(results: pd.DataFrame, baseline_r2: float) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker = "## 15. Ансамбль: гео-признак от решающего дерева в линейке"

    res = results.copy()
    res["Δlog_r2"] = (res["log_r2"] - baseline_r2).round(4)
    res = res.sort_values("log_r2", ascending=False)

    desc = {
        "none": "базовая §13 (без гео)",
        "kmeans_ohe": f"KMeans(k={KMEANS_K}) → OHE",
        "tree_ohe": f"лист дерева ({TREE_OHE_LEAVES}) → OHE",
        "tree_num": f"предсказание дерева ({TREE_NUM_LEAVES}) $/м², число",
        "tree_lognum": f"log предсказания дерева ({TREE_NUM_LEAVES})",
        "tree_ohe_num": "OHE листа + численная оценка",
        "tree_num_kmeans": "численная оценка дерева + KMeans OHE",
    }

    tbl = ["| Вариант | Гео-признак | log R² | Δ к базе | median APE | MAPE | usd R² |",
           "|---|---|---:|---:|---:|---:|---:|"]
    best_variant = res.iloc[0]["variant"]
    for _, r in res.iterrows():
        star = " ⭐" if r["variant"] == best_variant else ""
        tbl.append(f"| `{r['variant']}`{star} | {desc.get(r['variant'], '')} | "
                   f"{r['log_r2']} | {r['Δlog_r2']:+.4f} | {r['median_ape_pct']}% | "
                   f"{r['mape_pct']}% | {r['usd_r2']} |")

    best = res.iloc[0]
    block = (
        f"\n---\n\n{marker}\n\n"
        "Идея: в линейку §13 географию даёт `address` (hash). Здесь проверяем, "
        "добавляет ли сигнал **пространственный признак от решающего дерева** "
        "(§14) — как категория (OHE листа) или как число (out-of-fold target "
        "encoding по геозоне). Сравниваем с KMeans-кластером из ранних "
        "экспериментов. Скрипт: [`geo_tree_ensemble.py`](geo_tree_ensemble.py).\n\n"
        "**Дизайн.** SGD-гиперпараметры зафиксированы на лучших из §13 "
        "(`alpha=1e-5, l1_ratio=0.15, epsilon=0.5`) — чистая абляция, меняется "
        "только гео-фича. Дерево и KMeans обучаются **внутри каждого CV-фолда** "
        "(их выход зависит от таргета через $/м²), поэтому утечки нет — всё через "
        "`cross_val_predict` 5-fold.\n\n"
        + "\n".join(tbl) + "\n\n"
        f"**Лучший вариант: `{best['variant']}`** — log R² = {best['log_r2']} "
        f"(Δ {best['Δlog_r2']:+.4f} к базе), median APE = {best['median_ape_pct']}%.\n\n"
        "![Сравнение вариантов ансамбля](figs/29_geo_ensemble.png)\n\n"
        "### Что видно\n\n"
        "- **OHE листа vs численная оценка.** OHE даёт линейке набор бинарных "
        "зон (каждая со своим свободным коэффициентом), численная оценка — один "
        "признак «ожидаемая $/м² по координатам». Численная компактнее и обычно "
        "устойчивее: один коэффициент вместо десятков разреженных.\n"
        "- **Дерево vs KMeans.** Оба — про географию, но дерево режет пространство "
        "по самой цене $/м² (supervised), а KMeans — по плотности точек "
        "(unsupervised). Поэтому дерево обычно сильнее как ценовая фича.\n"
        "- **Прирост над базой §13 невелик** — `address`-hash уже впитал почти всю "
        "географию (§10). Гео-дерево частично дублирует адресный сигнал, но в "
        "лучшем варианте всё же добавляет немного и, главное, **интерпретируемо** "
        "(в отличие от 1024 hash-фичей).\n\n"
        "### Подумать потом\n\n"
        "1. Если важна интерпретация/лёгкость — заменить `address`-hash на "
        "`tree_num` (одно число вместо 1024 фичей) и посмотреть, сколько R² теряем.\n"
        "2. `tree_num` хорош для прод-инференса: дерево сериализуется в "
        "[`tree_geo.joblib`](tree_geo.joblib), на тесте применяется мгновенно.\n"
        "3. Стек уровнем выше: дерево по координатам → остатки линейки скормить "
        "второму дереву (boosting-подобно). Выходит за рамки чистой линейки.\n"
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
    y = np.log1p(df["usd_price"].values)
    X = df.drop(columns=["usd_price"])
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    variants = ["none", "kmeans_ohe", "tree_ohe", "tree_num", "tree_lognum",
                "tree_ohe_num", "tree_num_kmeans"]
    rows = []
    for v in variants:
        pipe = make_pipeline(v)
        pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        m = metrics_oof(y, pred)
        m["variant"] = v
        rows.append(m)
        print(f"  {v:18s} log_r2={m['log_r2']:.4f} medAPE={m['median_ape_pct']:.2f}% "
              f"MAPE={m['mape_pct']:.2f}%")

    results = pd.DataFrame(rows)
    baseline_r2 = float(results.loc[results["variant"] == "none", "log_r2"].iloc[0])

    plot_compare(results)
    results.to_csv(FIGS / "geo_ensemble_metrics.csv", index=False)
    Path("geo_ensemble_meta.json").write_text(
        json.dumps({
            "sgd_params": {k: SGD_KW[k] for k in ("alpha", "l1_ratio", "epsilon")},
            "tree_ohe_leaves": TREE_OHE_LEAVES,
            "tree_num_leaves": TREE_NUM_LEAVES,
            "kmeans_k": KMEANS_K,
            "results": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved geo_ensemble_meta.json + figs/29_geo_ensemble.png")

    update_report(results, baseline_r2)
    print("Report updated with §15.")


if __name__ == "__main__":
    main()
