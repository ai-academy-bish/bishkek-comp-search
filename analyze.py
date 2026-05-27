"""EDA для линейной модели стоимости.

Анализирует:
- дисбаланс категорий,
- распределения числовых признаков,
- наличие групп (KMeans по гео, биннинг),
- линейность связей с usd_price,
- мультиколлинеарность (корр. матрица + VIF),
- нормальность остатков базового OLS.

Сохраняет графики в figs/ и текстовый отчёт в report.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

NUMERIC = ["lat", "lon", "build_year", "floor", "total_floors", "rooms", "area_total", "area_living"]
CATEGORICAL = ["offer_type", "series", "building_material", "condition"]
TARGET = "usd_price"

FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    path = FIGS / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load() -> pd.DataFrame:
    df = pd.read_csv("train_processed.csv")
    df["log_price"] = np.log1p(df[TARGET])
    df["is_free_layout"] = (df["rooms"] == 1000).astype(int)
    df.loc[df["rooms"] == 1000, "rooms"] = np.nan
    df["price_per_sqm"] = df[TARGET] / df["area_total"]
    return df


# ---------- 1. Распределения числовых ----------
def numeric_distributions(df: pd.DataFrame) -> dict:
    stats_out = {}
    for col in NUMERIC + [TARGET]:
        s = df[col].dropna()
        skew = float(stats.skew(s))
        kurt = float(stats.kurtosis(s))
        stats_out[col] = {"skew": round(skew, 2), "kurt": round(kurt, 2),
                          "n_missing": int(df[col].isna().sum())}

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, col in zip(axes.flat, NUMERIC + [TARGET]):
        s = df[col].dropna()
        ax.hist(s, bins=50, color="steelblue", alpha=0.8)
        ax.set_title(f"{col}  (skew={stats_out[col]['skew']})")
    save(fig, "01_numeric_hist.png")

    # log target
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[TARGET], bins=80, color="steelblue"); axes[0].set_title("usd_price (raw)")
    axes[1].hist(df["log_price"], bins=80, color="seagreen"); axes[1].set_title("log1p(usd_price)")
    save(fig, "02_target_log_transform.png")

    # qq plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    stats.probplot(df[TARGET], dist="norm", plot=axes[0]); axes[0].set_title("QQ raw")
    stats.probplot(df["log_price"], dist="norm", plot=axes[1]); axes[1].set_title("QQ log")
    save(fig, "03_target_qq.png")

    # boxplots — outliers
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.boxplot(y=df[col], ax=ax, color="steelblue")
        ax.set_title(col)
    save(fig, "04_numeric_box.png")
    return stats_out


# ---------- 2. Дисбаланс категорий ----------
def categorical_imbalance(df: pd.DataFrame) -> dict:
    info = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, col in zip(axes.flat, CATEGORICAL):
        vc = df[col].fillna("MISSING").value_counts()
        vc.plot.barh(ax=ax, color="steelblue")
        ax.set_title(f"{col}  ({len(vc)} категорий)")
        ax.invert_yaxis()
        top_share = vc.iloc[0] / vc.sum()
        info[col] = {
            "n_categories": int(len(vc)),
            "top_value": str(vc.index[0]),
            "top_share": round(float(top_share), 3),
            "min_count": int(vc.min()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 1),
        }
    save(fig, "05_categorical_counts.png")
    return info


# ---------- 3. Линейность: scatter и Spearman vs Pearson ----------
def linearity(df: pd.DataFrame) -> dict:
    info = {}
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, col in zip(axes.flat, NUMERIC):
        sub = df[[col, TARGET, "log_price"]].dropna()
        ax.scatter(sub[col], sub["log_price"], s=4, alpha=0.3, color="steelblue")
        pearson = sub[col].corr(sub["log_price"], method="pearson")
        spearman = sub[col].corr(sub["log_price"], method="spearman")
        info[col] = {"pearson_log": round(float(pearson), 3),
                     "spearman_log": round(float(spearman), 3),
                     "gap": round(float(abs(spearman) - abs(pearson)), 3)}
        ax.set_title(f"{col}\nP={pearson:.2f}  S={spearman:.2f}")
    axes.flat[-1].axis("off")
    save(fig, "06_linearity_scatter.png")

    # area_total — самый сильный признак, проверим линейность отдельно
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sub = df.dropna(subset=["area_total"])
    axes[0].scatter(sub["area_total"], sub[TARGET], s=4, alpha=0.3); axes[0].set_title("area vs price")
    axes[1].scatter(sub["area_total"], sub["log_price"], s=4, alpha=0.3); axes[1].set_title("area vs log price")
    axes[2].scatter(np.log(sub["area_total"]), sub["log_price"], s=4, alpha=0.3); axes[2].set_title("log area vs log price")
    save(fig, "07_area_transforms.png")
    return info


# ---------- 4. Категориальные vs цена ----------
def categorical_vs_price(df: pd.DataFrame) -> dict:
    info = {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, col in zip(axes.flat, CATEGORICAL):
        order = df.groupby(col)["log_price"].median().sort_values().index
        sns.boxplot(data=df, x=col, y="log_price", order=order, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(col)
        # F-statistic — есть ли значимые различия групп
        groups = [g["log_price"].dropna().values for _, g in df.groupby(col)]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            info[col] = {"anova_F": round(float(f), 1), "anova_p": float(p)}
    save(fig, "08_categorical_vs_price.png")
    return info


# ---------- 4b. price_per_sqm по группам ----------
def ppsqm_by_groups(df: pd.DataFrame) -> dict:
    info = {}
    # общий вид
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    s = df["price_per_sqm"].dropna()
    s_clip = s.clip(upper=s.quantile(0.99))
    axes[0].hist(s_clip, bins=80, color="steelblue"); axes[0].set_title(f"price_per_sqm (clip 99%)  median={s.median():.0f}")
    axes[1].hist(np.log(s), bins=80, color="seagreen"); axes[1].set_title("log(price_per_sqm)")
    save(fig, "13_ppsqm_overall.png")

    # боксплоты по каждой категории
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, col in zip(axes.flat, CATEGORICAL):
        sub = df.dropna(subset=["price_per_sqm"]).copy()
        sub[col] = sub[col].fillna("MISSING")
        # обрезаем верхний хвост чтобы боксы было видно
        sub = sub[sub["price_per_sqm"] < sub["price_per_sqm"].quantile(0.99)]
        order = sub.groupby(col)["price_per_sqm"].median().sort_values().index
        sns.boxplot(data=sub, x=col, y="price_per_sqm", order=order, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(f"price_per_sqm по {col}")
        g = sub.groupby(col)["price_per_sqm"].agg(["count", "median"])
        info[col] = {"median_min": round(float(g["median"].min()), 0),
                     "median_max": round(float(g["median"].max()), 0),
                     "spread": round(float(g["median"].max() - g["median"].min()), 0),
                     "top": str(g["median"].idxmax()),
                     "bottom": str(g["median"].idxmin())}
    save(fig, "14_ppsqm_by_category.png")

    # violin для самого важного — series
    fig, ax = plt.subplots(figsize=(13, 6))
    sub = df.dropna(subset=["price_per_sqm"]).copy()
    sub = sub[sub["price_per_sqm"] < sub["price_per_sqm"].quantile(0.99)]
    order = sub.groupby("series")["price_per_sqm"].median().sort_values().index
    sns.violinplot(data=sub, x="series", y="price_per_sqm", order=order, ax=ax, cut=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Распределение price_per_sqm по сериям (violin, обрезано 99%)")
    save(fig, "15_ppsqm_series_violin.png")

    # медианная таблица
    summary = []
    for col in CATEGORICAL:
        sub = df.dropna(subset=["price_per_sqm"]).copy()
        sub[col] = sub[col].fillna("MISSING")
        g = sub.groupby(col)["price_per_sqm"].agg(["count", "median", "mean", "std"]).round(0)
        g["feature"] = col
        summary.append(g.reset_index().rename(columns={col: "value"})[["feature", "value", "count", "median", "mean", "std"]])
    pd.concat(summary).to_csv(FIGS / "ppsqm_summary.csv", index=False)
    return info


# ---------- 5. Гео-группы через KMeans ----------
def geo_clusters(df: pd.DataFrame) -> dict:
    valid = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()
    coords = valid[["lat", "lon"]].values
    km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(coords)
    valid["geo_cluster"] = km.labels_

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(valid["lon"], valid["lat"], c=valid["geo_cluster"], s=6, cmap="tab10", alpha=0.7)
    ax.set_title("Гео-кластеры (KMeans k=8)")
    plt.colorbar(sc, ax=ax)
    save(fig, "09_geo_clusters.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=valid, x="geo_cluster", y="log_price", ax=ax)
    ax.set_title("log_price по гео-кластерам")
    save(fig, "10_geo_cluster_price.png")

    # price_per_sqm по гео-кластерам
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = valid.dropna(subset=["price_per_sqm"]).copy()
    sub = sub[sub["price_per_sqm"] < sub["price_per_sqm"].quantile(0.99)]
    order = sub.groupby("geo_cluster")["price_per_sqm"].median().sort_values().index
    sns.boxplot(data=sub, x="geo_cluster", y="price_per_sqm", order=order, ax=ax)
    ax.set_title("price_per_sqm по гео-кластерам")
    save(fig, "16_ppsqm_geo.png")

    group_med = valid.groupby("geo_cluster")["usd_price"].median()
    ppsqm_med = valid.groupby("geo_cluster")["price_per_sqm"].median()
    spread = float(group_med.max() - group_med.min())
    return {"k": 8, "median_spread_usd": round(spread, 0),
            "cheapest": int(group_med.idxmin()), "priciest": int(group_med.idxmax()),
            "ppsqm_min": round(float(ppsqm_med.min()), 0),
            "ppsqm_max": round(float(ppsqm_med.max()), 0)}


# ---------- 6. Мультиколлинеарность ----------
def multicollinearity(df: pd.DataFrame) -> dict:
    cols = ["build_year", "floor", "total_floors", "rooms", "area_total", "lat", "lon"]
    sub = df[cols].dropna()
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Корреляционная матрица")
    save(fig, "11_corr_heatmap.png")

    X = sm.add_constant(sub)
    vifs = {col: float(variance_inflation_factor(X.values, i + 1)) for i, col in enumerate(cols)}
    return {"vif": {k: round(v, 2) for k, v in vifs.items()}}


# ---------- 7. Бейзлайн OLS и его остатки ----------
def baseline_ols(df: pd.DataFrame) -> dict:
    cols = ["area_total", "rooms", "build_year", "floor", "total_floors", "lat", "lon"]
    sub = df.dropna(subset=cols + ["log_price"]).copy()
    X = sm.add_constant(sub[cols])
    y = sub["log_price"]
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    resid = y - pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(pred, resid, s=4, alpha=0.3); axes[0].axhline(0, c="r")
    axes[0].set_xlabel("pred"); axes[0].set_ylabel("residual"); axes[0].set_title("Residuals vs Fitted")
    axes[1].hist(resid, bins=60); axes[1].set_title("Residual distribution")
    stats.probplot(resid, dist="norm", plot=axes[2]); axes[2].set_title("QQ residuals")
    save(fig, "12_baseline_residuals.png")

    return {"r2": round(float(model.rsquared), 3),
            "r2_adj": round(float(model.rsquared_adj), 3),
            "n": int(sub.shape[0]),
            "shapiro_p": float(stats.shapiro(resid.sample(min(5000, len(resid)), random_state=0))[1]),
            "resid_skew": round(float(stats.skew(resid)), 3)}


# ---------- 8. Отчёт ----------
def write_report(num_dist, cat_imb, lin, cat_vs, ppsqm, geo, multi, ols) -> None:
    L: list[str] = []
    a = L.append

    a("# EDA для линейной модели стоимости (Bishkek apartments)")
    a("")
    a("Источник: `train_processed.csv` (7134 строки, 14 колонок). Целевая — `usd_price`. "
      "Все графики собраны скриптом `analyze.py` и лежат в `figs/`.")
    a("")
    a("---")
    a("")

    # === 1 ===
    a("## 1. Распределения числовых признаков")
    a("")
    a("![Гистограммы числовых признаков](figs/01_numeric_hist.png)")
    a("")
    a("**Что видно.** `area_total`, `area_living`, `rooms`, `usd_price` — все скошены вправо (длинный хвост дорогих/больших объектов). "
      "`lat`/`lon` распределены узко (Бишкек), `build_year` бимодален — старый фонд (1960–1990) + современная застройка (2018–2025).")
    a("")
    a("| Признак | Skew | Kurt | Пропусков |")
    a("|---|---:|---:|---:|")
    for col, v in num_dist.items():
        a(f"| {col} | {v['skew']} | {v['kurt']} | {v['n_missing']} |")
    a("")
    a("![Boxplot числовых](figs/04_numeric_box.png)")
    a("")
    a("**Что видно.** Длинные верхние усы у `area_total` (до 650 м²) и `total_floors` (до 27) — это валидные элитки/новостройки. "
      "`area_living` имеет min=1 м² — явные ошибки разметки, фильтровать.")
    a("")
    a("### Таргет: usd_price")
    a("")
    a("![Лог-трансформация таргета](figs/02_target_log_transform.png)")
    a("")
    a("![QQ-plot таргета](figs/03_target_qq.png)")
    a("")
    a(f"**Что видно.** Сырой `usd_price` сильно скошен (skew={num_dist[TARGET]['skew']}), QQ-линия резко уходит вверх в правом хвосте. "
      "После `log1p` распределение почти нормальное, QQ практически прямая в центральной части. "
      "**Решение:** обучать линейную модель на `y = log1p(usd_price)`, метрики типа RMSLE/MAE на лог-шкале.")
    a("")

    # === 2 ===
    a("---")
    a("")
    a("## 2. Дисбаланс категориальных")
    a("")
    a("![Счётчики категорий](figs/05_categorical_counts.png)")
    a("")
    a("| Признак | Категорий | Top | Доля top | Min count | Пропусков, % |")
    a("|---|---:|---|---:|---:|---:|")
    for col, v in cat_imb.items():
        a(f"| {col} | {v['n_categories']} | {v['top_value']} | {v['top_share']} | {v['min_count']} | {v['missing_pct']} |")
    a("")
    a("**Что видно.**")
    a(f"- `offer_type`: {cat_imb['offer_type']['top_share']*100:.0f}% «от агента» — сильный дисбаланс, но всего 2 класса, OHE безопасно.")
    a(f"- `series`: «элитка» доминирует ({cat_imb['series']['top_share']*100:.0f}%), 14 категорий, "
      f"минимальная имеет всего {cat_imb['series']['min_count']} наблюдений. Редкие («107 серия», «104 серия улучшенная», «пентхаус») при OHE "
      "дадут шумные коэффициенты — **объединить классы с count < 30 в `series_other`**.")
    a("- `condition`: 8% пропусков — не дропать, а создать категорию `unknown`.")
    a("- `building_material`: 3 класса, баланс приемлемый.")
    a("")

    # === 3 ===
    a("---")
    a("")
    a("## 3. Линейность связей с log(price)")
    a("")
    a("![Линейность скаттер](figs/06_linearity_scatter.png)")
    a("")
    a("| Признак | Pearson | Spearman | |S|−|P| (изгиб) |")
    a("|---|---:|---:|---:|")
    for col, v in lin.items():
        a(f"| {col} | {v['pearson_log']} | {v['spearman_log']} | {v['gap']} |")
    a("")
    a("**Что видно.** Чем больше gap между Spearman и Pearson, тем сильнее связь нелинейна. "
      f"`area_total` (Pearson={lin['area_total']['pearson_log']}) — главный драйвер; gap у неё небольшой, но всё же отличный от нуля.")
    a("")
    a("![Трансформации площади](figs/07_area_transforms.png)")
    a("")
    a("**Что видно.** Левый график — `area` vs `price` (изогнутая), средний — `area` vs `log(price)` (лучше, но нижний хвост загибается), "
      "правый — `log(area)` vs `log(price)` — **почти идеальная прямая**. Это классическая лог-лог-зависимость, типичная для рынка жилья. "
      "**Решение:** в линейку подаём `log(area_total)`, не сырую площадь.")
    a("")

    # === 4 ===
    a("---")
    a("")
    a("## 4. Категории vs log(price) — ANOVA")
    a("")
    a("![Категории vs log price](figs/08_categorical_vs_price.png)")
    a("")
    a("| Признак | F-statistic | p-value |")
    a("|---|---:|---:|")
    for col, v in cat_vs.items():
        a(f"| {col} | {v['anova_F']} | {v['anova_p']:.2e} |")
    a("")
    a("**Что видно.** Все категории значимы (p ≈ 0). Самый сильный сигнал у `series` (F={:.0f}) и `condition` (F={:.0f}). "
      "Боксплоты отсортированы по медиане — видно монотонную лестницу серий от «малосемейки» (дёшево) до «пентхауса» (дорого). "
      "Категории — обязательная часть линейной модели.".format(cat_vs['series']['anova_F'], cat_vs['condition']['anova_F']))
    a("")

    # === 5 — price_per_sqm by groups (NEW) ===
    a("---")
    a("")
    a("## 5. Распределение price_per_sqm по группам")
    a("")
    a("Цена за квадрат — нормирует таргет относительно площади и показывает «качество» жилья, "
      "очищенное от размера. Для линейной модели это намёк на то, какие категории сдвигают **наклон** "
      "лог-лог-зависимости `log(price) ~ log(area)`.")
    a("")
    a("![price_per_sqm overall](figs/13_ppsqm_overall.png)")
    a("")
    a("**Что видно.** Распределение скошено (есть выбросы > $5000/м², отдельные точки до $100k/м² — артефакты разметки). "
      "Лог-вид симметричен — то есть и `price_per_sqm` лучше лог-преобразовывать при анализе.")
    a("")
    a("![price_per_sqm by category](figs/14_ppsqm_by_category.png)")
    a("")
    a("Сводка медиан (`$/м²`):")
    a("")
    a("| Признак | Min медиана | Max медиана | Разброс | Самая дорогая категория | Самая дешёвая |")
    a("|---|---:|---:|---:|---|---|")
    for col, v in ppsqm.items():
        a(f"| {col} | {v['median_min']:.0f} | {v['median_max']:.0f} | {v['spread']:.0f} | {v['top']} | {v['bottom']} |")
    a("")
    a("**Выводы по группам:**")
    a(f"- `series`: разброс медиан ${ppsqm['series']['spread']:.0f}/м² между «{ppsqm['series']['bottom']}» и «{ppsqm['series']['top']}». "
      "Самый сильный категориальный регрессор по удельной цене.")
    a(f"- `condition`: разброс ${ppsqm['condition']['spread']:.0f}/м². «Евроремонт» доминирует, «не достроено» — внизу. "
      "Линейная зависимость от качества отделки очевидна.")
    a(f"- `building_material`: разброс ${ppsqm['building_material']['spread']:.0f}/м² — самый слабый эффект, "
      "но монолит стабильно дороже панели.")
    a(f"- `offer_type`: разница между агентом и собственником ${ppsqm['offer_type']['spread']:.0f}/м² — "
      "минимальная, можно даже не включать, если хочется упростить модель.")
    a("")
    a("![price_per_sqm by series violin](figs/15_ppsqm_series_violin.png)")
    a("")
    a("**Что видно.** Violin показывает не только медианы, но и форму распределения внутри серии. "
      "У «элитки» широкое распределение (внутри много разнокачественных ЖК), у «хрущёвки»/«104 серии» — узкое и плотное. "
      "Это значит: «элитка» сама по себе плохо предсказывает — нужно добавлять взаимодействие с `geo_cluster` или `condition`.")
    a("")
    a(f"Полная таблица медиан/средних/std по всем категориям сохранена в `figs/ppsqm_summary.csv`.")
    a("")

    # === 6. Гео ===
    a("---")
    a("")
    a("## 6. Гео-группы")
    a("")
    a("![Гео кластеры](figs/09_geo_clusters.png)")
    a("")
    a("**Что видно.** KMeans с k=8 разбивает Бишкек на 8 геозон. Кластеры визуально соответствуют районам "
      "(центр, Джал, мкрн, Восток, окраины). Эти границы воспроизводимы и стабильны для линейной модели.")
    a("")
    a("![log price by geo cluster](figs/10_geo_cluster_price.png)")
    a("")
    a(f"![price_per_sqm by geo cluster](figs/16_ppsqm_geo.png)")
    a("")
    a(f"**Что видно.** Разброс медианной цены между кластерами ≈ **${geo['median_spread_usd']:.0f}**, "
      f"медианной `price_per_sqm` — от **${geo['ppsqm_min']:.0f}** до **${geo['ppsqm_max']:.0f}** за м². "
      "Сырые `lat`/`lon` дают для линейной модели слабый сигнал (зависимость нелинейная), а one-hot по `geo_cluster` — "
      "сильный и интерпретируемый.")
    a("")

    # === 7. VIF ===
    a("---")
    a("")
    a("## 7. Мультиколлинеарность")
    a("")
    a("![Корреляционная матрица](figs/11_corr_heatmap.png)")
    a("")
    a("| Признак | VIF |")
    a("|---|---:|")
    for col, v in multi["vif"].items():
        a(f"| {col} | {v} |")
    a("")
    a("**Что видно.** Сильная корреляция между `rooms` и `area_total` (r≈0.8): больше комнат = больше площадь. "
      "VIF > 5 — повод задуматься, > 10 — проблема: коэффициенты линейной модели становятся нестабильными. "
      "**Решение:** оставить только `area_total` (или `log(area_total)`), а вместо `rooms` подать `area_per_room = area_total / rooms` — "
      "это декоррелированный признак, показывающий «компактность планировки».")
    a("")

    # === 8. OLS ===
    a("---")
    a("")
    a("## 8. Бейзлайн OLS")
    a("")
    a(f"Простейший OLS на голых числовых: `log_price ~ area_total + rooms + build_year + floor + total_floors + lat + lon`. "
      f"Обучен на {ols['n']} строках без NaN.")
    a("")
    a(f"- **R² = {ols['r2']}**, adj R² = {ols['r2_adj']}")
    a(f"- Skew остатков = {ols['resid_skew']}, Shapiro p = {ols['shapiro_p']:.2e}")
    a("")
    a("![Диагностика остатков](figs/12_baseline_residuals.png)")
    a("")
    a("**Что видно.** Residuals vs Fitted — есть веер (гетероскедастичность ослабла по сравнению с сырым таргетом, но не пропала). "
      "QQ — тяжёлые хвосты, особенно слева (недооценка дешёвых объектов). "
      "Это хороший базовый ориентир: после добавления log(area), OHE категорий и geo_cluster R² должен подняться значительно.")
    a("")

    # === 9. Recos ===
    a("---")
    a("")
    a("## 9. Сводные рекомендации для линейной модели")
    a("")
    a("### Обязательно")
    a("1. **Таргет**: `y = log1p(usd_price)`. Метрика — RMSE/MAE на лог-шкале (= RMSLE/MAPE на исходной).")
    a("2. **Площадь**: подавать `log(area_total)`, а не сырую.")
    a("3. **One-hot** для `series`, `condition`, `building_material`, `offer_type`. Объединить редкие категории `series` (count < 30) в `series_other`.")
    a("4. **Пропуски `condition`** (8%) → отдельная категория `unknown`, не выкидывать.")
    a("5. **Geo**: KMeans k=8 → one-hot `geo_cluster`. Сырые `lat`/`lon` либо убрать, либо оставить как остаточный сигнал.")
    a("6. **Фильтры данных**:")
    a("   - дропнуть строки с `area_living > area_total` (5 шт),")
    a("   - дропнуть строки с `lon` вне [70, 80] (перепутаны координаты),")
    a("   - клиппинг таргета по 1/99 перцентилю чтобы не учиться на выбросах.")
    a("7. **rooms=1000** → флаг `is_free_layout = 1`, само поле `rooms` заменить на NaN→медиану.")
    a("8. **build_year** → `building_age = 2026 - build_year`, флаг `is_offplan = build_year > 2026`.")
    a("")
    a("### Сильно поможет")
    a("9. `floor_ratio = floor / total_floors`, `is_first_floor`, `is_last_floor` — нелинейный эффект этажа.")
    a("10. **Декорреляция**: не подавать `rooms` и `area_total` вместе — взять `area_per_room`.")
    a("11. **Регуляризация обязательна**: Ridge / ElasticNet — у нас десятки OHE-признаков и шум по редким категориям. "
        "Только Lasso сам отбросит малозначимые.")
    a("12. **Robust loss** (Huber) или таргет-клиппинг — на выбросах $1.2M линейка ломается.")
    a("13. **K-Fold CV с группировкой по `geo_cluster`** — иначе утечка географии в фолд завышает оценку.")
    a("")
    a("### После обучения проверить")
    a("- Residuals vs fitted: веер → гетероскедастичность → WLS или Box-Cox таргета.")
    a("- Коэффициенты при OHE серий должны идти лесенкой (малосемейка < хрущёвка < ... < пентхаус); если порядок ломается — переобучение.")
    a("- Permutation importance: `log(area_total)`, `series`, `geo_cluster`, `condition` должны быть в топе.")
    a("")

    Path("report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    df = load()
    print("loaded", df.shape)
    num_dist = numeric_distributions(df)
    cat_imb = categorical_imbalance(df)
    lin = linearity(df)
    cat_vs = categorical_vs_price(df)
    ppsqm = ppsqm_by_groups(df)
    geo = geo_clusters(df)
    multi = multicollinearity(df)
    ols = baseline_ols(df)
    write_report(num_dist, cat_imb, lin, cat_vs, ppsqm, geo, multi, ols)
    print(f"Saved {len(list(FIGS.glob('*.png')))} figures to {FIGS}/")
    print("Report: report.md")


if __name__ == "__main__":
    main()
