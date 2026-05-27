"""Заполнение пропусков в train_processed.csv → train_filled.csv.

Стратегия (согласована с EDA в report.md, §1, §11):
1. `area_living` — удалить колонку (80% пропусков, шумная фича).
2. `build_year`:
   - импьют медианой по `building_material` (монолитный/кирпичный/панельный),
     fallback — глобальная медиана;
   - добавить бинарный `is_old = build_year < 2000` (бимодальное распределение,
     порог из EDA).
3. `condition` — импьют модой (наиболее вероятное значение).
4. `floor`/`total_floors` — удалить строки с NaN (18 шт).
5. `series` — удалить строку с NaN (1 шт).

Все статистики сохраняются в `imputation_meta.json` чтобы тот же препроцессинг
можно было применить к тестовому набору без переобучения статистик.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DROP_COLUMNS = ["area_living"]
DROP_ROWS_IF_NA = ["floor", "total_floors", "series"]
IS_OLD_THRESHOLD = 2000


def fit_impute(df: pd.DataFrame) -> dict:
    """Считает статистики заполнения по тренировочному набору."""
    by_mat = (
        df.dropna(subset=["build_year"])
          .groupby("building_material")["build_year"]
          .median()
          .to_dict()
    )
    meta = {
        "drop_columns": DROP_COLUMNS,
        "drop_rows_if_na": DROP_ROWS_IF_NA,
        "build_year_by_material": {k: float(v) for k, v in by_mat.items()},
        "build_year_global_median": float(df["build_year"].median()),
        "is_old_threshold": IS_OLD_THRESHOLD,
        "condition_mode": str(df["condition"].mode().iloc[0]),
    }
    return meta


def apply_impute(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Применяет заполнение к произвольному набору по сохранённым статистикам."""
    df = df.drop(columns=[c for c in meta["drop_columns"] if c in df.columns])
    df = df.dropna(subset=meta["drop_rows_if_na"]).copy()

    by_mat = pd.Series(meta["build_year_by_material"])
    df["build_year"] = (
        df["build_year"]
        .fillna(df["building_material"].map(by_mat))
        .fillna(meta["build_year_global_median"])
    )
    df["is_old"] = (df["build_year"] < meta["is_old_threshold"]).astype(int)

    df["condition"] = df["condition"].fillna(meta["condition_mode"])
    return df.reset_index(drop=True)


def report(before: pd.DataFrame, after: pd.DataFrame, meta: dict) -> str:
    lines = ["## Сводка заполнения пропусков", ""]
    lines.append(f"Строк до: {len(before)}, после: {len(after)} (удалено {len(before) - len(after)}).")
    lines.append("")
    lines.append("**Пропуски до / после:**")
    lines.append("")
    lines.append("| Колонка | Было | Стало |")
    lines.append("|---|---:|---:|")
    cols = sorted(set(before.columns) | set(after.columns))
    for c in cols:
        b = int(before[c].isna().sum()) if c in before.columns else "—"
        a = int(after[c].isna().sum()) if c in after.columns else "DROPPED"
        if b == 0 and a == 0:
            continue
        lines.append(f"| {c} | {b} | {a} |")
    lines.append("")
    lines.append("**Применённые статистики (imputation_meta.json):**")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


def update_report(summary: str) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker_start = "## 11. Обработка пропусков"
    block = (
        "\n---\n\n"
        f"{marker_start}\n\n"
        "Применённые правила (см. `fill_missing.py`):\n\n"
        "1. **`area_living`** — колонка удалена (80% пропусков, шум).\n"
        "2. **`build_year`** — пропуски заполнены медианой по `building_material`. "
        "Дополнительно создан бинарный признак **`is_old = build_year < 2000`** "
        "(в EDA §1 видна бимодальность: старый фонд до 2000 vs новостройки 2015+).\n"
        "3. **`condition`** — пропуски заполнены модой (наиболее частая категория).\n"
        "4. **`floor` / `total_floors`** — 18 строк с NaN удалены.\n"
        "5. **`series`** — 1 строка с NaN удалена.\n\n"
        "Статистики сохранены в `imputation_meta.json` — те же значения "
        "применяются к тестовому набору через `apply_impute(...)`. "
        "Это гарантирует отсутствие data leakage: на тесте не пересчитываем "
        "медианы/моды, а используем тренировочные.\n\n"
        f"{summary}\n"
    )
    if marker_start in text:
        text = text.split(marker_start)[0].rstrip()
        if text.endswith("---"):
            text = text[:-3].rstrip()
        text += "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    rpt.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="train_processed.csv", type=Path)
    parser.add_argument("--output", default="train_filled.csv", type=Path)
    parser.add_argument("--meta", default="imputation_meta.json", type=Path)
    parser.add_argument("--mode", choices=["fit", "apply"], default="fit",
                        help="fit — учим статистики на train; apply — применяем готовые к новому файлу")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.mode == "fit":
        meta = fit_impute(df)
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved imputation stats → {args.meta}")
    else:
        meta = json.loads(args.meta.read_text(encoding="utf-8"))
        print(f"Loaded imputation stats ← {args.meta}")

    filled = apply_impute(df, meta)
    filled.to_csv(args.output, index=False)
    print(f"Saved filled dataset → {args.output}  ({len(filled)} rows, {filled.shape[1]} cols)")

    if args.mode == "fit":
        summary = report(df, filled, meta)
        update_report(summary)
        print("Report updated with §11.")

    print("\nMissing after fill:")
    print(filled.isna().sum().to_string())


if __name__ == "__main__":
    main()
