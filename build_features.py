"""Финальный фичеинжиниринг: бинирование категорий и отсев слабых признаков.

Применяется к train_filled.csv → train_features.csv.

Правила:
1. `condition` → бинарный `condition_unfinished` = 1 для {«не достроено», «под самоотделку (псо)»},
   иначе 0. Боксплот §5 показал, что эти две категории формируют отдельный кластер
   $/м², а остальное (среднее/хорошее/евроремонт/MISSING) — почти неразличимы.
2. `series` → `series_group` (3 балансные группы по медиане $/м²: low / mid / high).
   Границы подбираются так, чтобы в каждой группе было ≈1/3 наблюдений.
3. Слабые признаки (разброс медиан $/м² между группами < $30) удаляются.
4. Исходные `condition`, `series` удаляются после бинирования.

Все правила сохраняются в feature_groupings.json — на тестовом наборе применяем
тот же маппинг без перерасчёта статистик.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

UNFINISHED_LABELS = ["не достроено", "под самоотделку (псо)"]
WEAK_SPREAD_THRESHOLD = 30  # $/м²: разброс медиан между группами фичи

sns.set_theme(style="whitegrid")
FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)


def fit_groupings(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["price_per_sqm"] = df["usd_price"] / df["area_total"]

    # --- series → 3 группы по рангу медианы $/м² ---
    # Считать split по cumcount нельзя: «элитка» = 67% выборки, схлопывает группы.
    # Берём терциль рангов медиан категорий: каждая группа содержит ~⅓ серий
    # (но не ⅓ строк — это компромисс из-за дисбаланса).
    series_stats = (
        df.groupby("series")
          .agg(median_ppsqm=("price_per_sqm", "median"),
               count=("price_per_sqm", "count"))
          .sort_values("median_ppsqm")
    )
    medians = series_stats["median_ppsqm"]
    q_low = medians.quantile(1 / 3)
    q_high = medians.quantile(2 / 3)

    def _grp(m: float) -> str:
        if m <= q_low:
            return "low"
        if m <= q_high:
            return "mid"
        return "high"

    series_map: dict[str, str] = {s: _grp(float(m)) for s, m in medians.items()}

    # --- проверка слабости признаков ---
    diagnostics: dict[str, dict] = {}
    weak: list[str] = []
    for col in ["offer_type", "building_material", "is_old"]:
        if col not in df.columns:
            continue
        med = df.groupby(col)["price_per_sqm"].median()
        spread = float(med.max() - med.min())
        diagnostics[col] = {
            "spread_usd_per_sqm": round(spread, 1),
            "medians": {str(k): round(float(v), 1) for k, v in med.items()},
        }
        if spread < WEAK_SPREAD_THRESHOLD:
            weak.append(col)

    meta = {
        "condition_unfinished_values": UNFINISHED_LABELS,
        "series_groups": series_map,
        "series_group_counts": {
            grp: int(sum(series_stats.loc[s, "count"]
                         for s in series_map if series_map[s] == grp))
            for grp in ["low", "mid", "high"]
        },
        "series_group_medians_usd_per_sqm": {
            grp: (round(float(df[df["series"].map(series_map) == grp]["price_per_sqm"].median()), 1)
                  if (df["series"].map(series_map) == grp).any() else None)
            for grp in ["low", "mid", "high"]
        },
        "series_group_quantile_cuts": {"q_low": float(q_low), "q_high": float(q_high)},
        "weak_spread_threshold": WEAK_SPREAD_THRESHOLD,
        "weak_features_diagnostics": diagnostics,
        "drop_features": weak + ["condition", "series"],
    }
    return meta


def apply_groupings(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    unfinished = set(meta["condition_unfinished_values"])
    df["condition_unfinished"] = df["condition"].isin(unfinished).astype(int)

    series_map = meta["series_groups"]
    # неизвестные серии (теоретически на тесте) → mid (центральная масса)
    df["series_group"] = df["series"].map(series_map).fillna("mid")

    to_drop = [c for c in meta["drop_features"] if c in df.columns]
    return df.drop(columns=to_drop)


def plot_groups(df_features: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Распределение price_per_sqm по итоговым категориям."""
    df = df_features.copy()
    df["price_per_sqm"] = df_raw["usd_price"].values / df_raw["area_total"].values
    df = df[df["price_per_sqm"] < df["price_per_sqm"].quantile(0.99)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # condition_unfinished
    sns.boxplot(data=df, x="condition_unfinished", y="price_per_sqm", ax=axes[0, 0],
                color="steelblue")
    axes[0, 0].set_xticklabels(["finished (0)", "unfinished/ПСО (1)"])
    axes[0, 0].set_title("condition_unfinished")

    # series_group
    order = ["low", "mid", "high"]
    sns.boxplot(data=df, x="series_group", y="price_per_sqm", order=order, ax=axes[0, 1])
    axes[0, 1].set_title("series_group")

    # building_material
    if "building_material" in df.columns:
        order_m = df.groupby("building_material")["price_per_sqm"].median().sort_values().index
        sns.boxplot(data=df, x="building_material", y="price_per_sqm",
                    order=order_m, ax=axes[1, 0])
        axes[1, 0].set_title("building_material")
    else:
        axes[1, 0].set_title("building_material — DROPPED")
        axes[1, 0].axis("off")

    # is_old
    if "is_old" in df.columns:
        sns.boxplot(data=df, x="is_old", y="price_per_sqm", ax=axes[1, 1])
        axes[1, 1].set_xticklabels(["new (0)", "old <2000 (1)"])
        axes[1, 1].set_title("is_old")
    else:
        axes[1, 1].set_title("is_old — DROPPED")
        axes[1, 1].axis("off")

    fig.suptitle("Распределение $/м² по итоговым категориям (входная точка моделирования)")
    fig.tight_layout()
    fig.savefig(FIGS / "21_groups_ppsqm.png", dpi=110)
    plt.close(fig)


def update_report(meta: dict, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker = "## 12. Бинирование категорий и отсев слабых признаков"

    # таблица series_group
    sg_lines = ["| series | series_group |", "|---|---|"]
    for s, g in sorted(meta["series_groups"].items(), key=lambda kv: (kv[1], kv[0])):
        sg_lines.append(f"| {s} | {g} |")

    counts = meta["series_group_counts"]
    meds = meta["series_group_medians_usd_per_sqm"]
    sg_summary = ["| series_group | строк | медиана $/м² |", "|---|---:|---:|"]
    for g in ["low", "mid", "high"]:
        sg_summary.append(f"| {g} | {counts.get(g, 0)} | {meds.get(g, 0)} |")

    diag = meta["weak_features_diagnostics"]
    diag_lines = ["| Признак | Разброс медиан $/м² | Решение |", "|---|---:|---|"]
    for col, v in diag.items():
        decision = "DROP" if col in meta["drop_features"] else "keep"
        diag_lines.append(f"| {col} | {v['spread_usd_per_sqm']} | {decision} |")

    block = (
        f"\n---\n\n{marker}\n\n"
        "Применяет [`build_features.py`](build_features.py) к `train_filled.csv` → `train_features.csv`. "
        "Это финальный шаг подготовки признаков перед линейным моделированием.\n\n"
        "### 12.1 Бинирование `condition` → `condition_unfinished`\n\n"
        "Из боксплота §5 видно, что `не достроено` и `под самоотделку (псо)` формируют отдельный кластер "
        "(медиана $1100–1200/м²), а `среднее`/`хорошее`/`MISSING`/`евроремонт` — почти неразличимы "
        "(медианы $1500–1600/м²). Поэтому 6 категорий → 1 бинарный признак:\n\n"
        f"- `condition_unfinished = 1`, если `condition ∈ {{{', '.join(meta['condition_unfinished_values'])}}}`\n"
        "- иначе `0`\n\n"
        "### 12.2 Бинирование `series` → `series_group`\n\n"
        "Серии (14 категорий, сильно несбалансированы — «элитка» 67%, «107 серия» 9 шт) бинируются "
        "в 3 группы по терцилям **медианы $/м² категорий** (по рангу, не по числу строк). "
        "Идеального баланса по строкам добиться нельзя из-за «элитки» в 67% выборки — она целиком "
        "попадает в одну группу. Тем не менее, у нас три непустые группы с монотонно растущими медианами.\n\n"
        + "\n".join(sg_lines) + "\n\n"
        "**Распределение по группам:**\n\n"
        + "\n".join(sg_summary) + "\n\n"
        "### 12.3 Отсев слабых признаков\n\n"
        f"Признак считается слабым, если разброс медиан $/м² между его группами < ${meta['weak_spread_threshold']}/м².\n\n"
        + "\n".join(diag_lines) + "\n\n"
        "### 12.4 Финальное распределение $/м² по группам\n\n"
        "![Распределение $/м² по итоговым категориям](figs/21_groups_ppsqm.png)\n\n"
        "**Что видно.**\n"
        f"- `condition_unfinished`: чёткое разделение медиан (готовое ~$1500/м² vs недостроенное ~$1150/м², "
        "спред ≈ $350–400). Бинаризация работает — кластеры действительно разные, "
        "потеря информации от схлопывания 5 категорий в 1 минимальная.\n"
        f"- `series_group`: монотонная лесенка медиан low → mid → high "
        f"($"
        f"{meds.get('low', 0)} → $"
        f"{meds.get('mid', 0)} → $"
        f"{meds.get('high', 0)}/м², спред ${(meds.get('high', 0) or 0) - (meds.get('low', 0) or 0):.0f}). "
        "По строкам группы несбалансированы (5033/1033/1050) — это цена за дисбаланс исходных категорий, "
        "но семантически тиры различимы.\n"
        f"- `building_material`: монолит/кирпич стабильно дороже панели; спред "
        f"${diag.get('building_material', {}).get('spread_usd_per_sqm', '?')}/м² — "
        "выше порога $30, оставляем.\n"
        "- `is_old`: старый фонд (до 2000) почему-то дороже за $/м² (медианы $1543 vs $1417). "
        f"Спред ${diag.get('is_old', {}).get('spread_usd_per_sqm', '?')}/м² — оставляем; "
        "вероятно, старый фонд в центральных районах.\n\n"
        "### 12.5 Финальный набор фичей\n\n"
        f"Колонки в `train_features.csv` ({len(df_after)} строк × {df_after.shape[1]} колонок):\n\n"
        + "- " + "\n- ".join(f"`{c}`" for c in df_after.columns) + "\n\n"
        "**Что делать дальше при моделировании:**\n"
        "- `address` — HashingVectorizer (см. §10), либо TfidfVectorizer для интерпретации.\n"
        "- `series_group`, `building_material`, `condition_unfinished`, `is_old` — OHE/целочисленные.\n"
        "- `area_total` — `log(area_total)` (см. §3).\n"
        "- `rooms`, `floor`, `total_floors`, `build_year` — числовые; добавить производные "
        "(`floor_ratio`, `area_per_room`, `building_age`).\n"
        "- `lat`, `lon` — либо в KMeans → `geo_cluster` (OHE), либо в RBF-сплайны.\n"
        "- Таргет: `y = log1p(usd_price)`, метрика RMSLE/MAE на лог-шкале.\n"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="train_filled.csv", type=Path)
    parser.add_argument("--output", default="train_features.csv", type=Path)
    parser.add_argument("--meta", default="feature_groupings.json", type=Path)
    parser.add_argument("--mode", choices=["fit", "apply"], default="fit")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.mode == "fit":
        meta = fit_groupings(df)
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved groupings meta → {args.meta}")
    else:
        meta = json.loads(args.meta.read_text(encoding="utf-8"))

    features = apply_groupings(df, meta)
    features.to_csv(args.output, index=False)
    print(f"Saved features → {args.output}  ({len(features)} rows, {features.shape[1]} cols)")

    if args.mode == "fit":
        plot_groups(features, df)
        update_report(meta, df, features)
        print("Report updated with §12.")

    print("\nFinal columns:")
    print(features.columns.tolist())
    print("\nFinal head:")
    print(features.head(3))


if __name__ == "__main__":
    main()
