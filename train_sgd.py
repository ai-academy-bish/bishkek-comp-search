"""SGDRegressor (Huber + ElasticNet) с GridSearchCV для предсказания log(usd_price).

Пайплайн:
1. address → HashingVectorizer(n_features=1024, ngram=(1,2)).
2. numeric (lat, lon, build_year, floor, total_floors, rooms, area_total, area_total²)
   → StandardScaler.
3. categorical (building_material, series_group) → OneHotEncoder.
4. binary (is_old, condition_unfinished) → passthrough.
5. target: log1p(usd_price).
6. SGDRegressor(loss='huber', penalty='elasticnet') с GridSearch.

Метрики считаются на out-of-fold предсказаниях (честная оценка генерализации).
Сохраняет model_params.json, графики в figs/, дополняет report.md §13.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)

TEXT_COL = "address"
# lat/lon исключены: связь с ценой нелинейная, сырые координаты в линейке —
# плохой сигнал. Географию ловит address через HashingVectorizer (см. §10).
NUMERIC_COLS = ["build_year", "floor", "total_floors", "rooms",
                "area_total", "area_total_sq"]
CATEGORICAL_COLS = ["building_material", "series_group"]
BINARY_COLS = ["is_old", "condition_unfinished"]


def load() -> pd.DataFrame:
    df = pd.read_csv("train_features.csv")
    df = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()
    df["address"] = df["address"].fillna("").str.lower()
    rooms_median = df.loc[df["rooms"] != 1000, "rooms"].median()
    df.loc[df["rooms"] == 1000, "rooms"] = rooms_median
    df["area_total_sq"] = df["area_total"] ** 2
    return df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("text", HashingVectorizer(n_features=1024, ngram_range=(1, 2),
                                        alternate_sign=False, norm="l2",
                                        lowercase=True), TEXT_COL),
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(sparse_output=True, handle_unknown="ignore"),
             CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    sgd = SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        max_iter=3000,
        tol=1e-4,
        random_state=0,
        learning_rate="invscaling",
        eta0=0.01,
    )
    return Pipeline([("pre", pre), ("reg", sgd)])


def metrics_oof(y: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y - y_pred
    log_rmse = float(np.sqrt(np.mean(resid ** 2)))
    log_mae = float(np.mean(np.abs(resid)))
    log_r2 = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    y_usd = np.expm1(y)
    p_usd = np.expm1(y_pred)
    ape = np.abs((y_usd - p_usd) / y_usd) * 100
    usd_mape = float(ape.mean())
    usd_med_ape = float(np.median(ape))
    usd_rmse = float(np.sqrt(np.mean((y_usd - p_usd) ** 2)))
    usd_mae = float(np.mean(np.abs(y_usd - p_usd)))
    usd_r2 = float(1 - np.sum((y_usd - p_usd) ** 2) / np.sum((y_usd - y_usd.mean()) ** 2))

    return {
        "log_rmse": round(log_rmse, 4),
        "log_mae": round(log_mae, 4),
        "log_r2": round(log_r2, 4),
        "usd_rmse": round(usd_rmse, 0),
        "usd_mae": round(usd_mae, 0),
        "usd_r2": round(usd_r2, 4),
        "usd_mape_pct": round(usd_mape, 2),
        "usd_median_ape_pct": round(usd_med_ape, 2),
    }


def plot_diagnostics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].scatter(y_pred, y, s=4, alpha=0.3)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    axes[0, 0].plot(lims, lims, "r--", lw=1)
    axes[0, 0].set_xlabel("predicted log(usd_price)")
    axes[0, 0].set_ylabel("actual log(usd_price)")
    axes[0, 0].set_title("Predicted vs Actual (log)")

    axes[0, 1].scatter(y_pred, resid, s=4, alpha=0.3)
    axes[0, 1].axhline(0, c="r", lw=1)
    axes[0, 1].set_xlabel("predicted log")
    axes[0, 1].set_ylabel("residual")
    axes[0, 1].set_title("Residuals vs Fitted")

    axes[1, 0].hist(resid, bins=80, color="steelblue")
    axes[1, 0].set_title(f"Residual distribution (σ={resid.std():.3f}, skew={stats.skew(resid):.2f})")
    axes[1, 0].set_xlabel("residual (log)")

    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ residuals")

    fig.tight_layout()
    fig.savefig(FIGS / "22_sgd_diagnostics.png", dpi=110)
    plt.close(fig)

    # USD scale
    y_usd = np.expm1(y)
    p_usd = np.expm1(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(p_usd, y_usd, s=4, alpha=0.3)
    lims = [0, max(y_usd.max(), p_usd.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("predicted, $")
    axes[0].set_ylabel("actual, $")
    axes[0].set_title("Pred vs Actual (USD, linear scale)")

    axes[1].loglog(p_usd, y_usd, ".", ms=2, alpha=0.3)
    axes[1].plot(lims, lims, "r--", lw=1)
    axes[1].set_xlabel("predicted, $ (log)")
    axes[1].set_ylabel("actual, $ (log)")
    axes[1].set_title("Pred vs Actual (USD, log-log)")
    fig.tight_layout()
    fig.savefig(FIGS / "23_sgd_usd_scatter.png", dpi=110)
    plt.close(fig)

    return {
        "resid_skew": float(stats.skew(resid)),
        "resid_std": float(resid.std()),
        "resid_kurt": float(stats.kurtosis(resid)),
    }


def update_report(out: dict) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker = "## 13. Модель: SGDRegressor (Huber + ElasticNet)"

    m = out["cv_metrics_oof"]
    bp = out["best_params"]
    rd = out["residual_diagnostics"]

    block = (
        f"\n---\n\n{marker}\n\n"
        "Финальная линейная модель на отобранных признаках. Скрипт: "
        "[`train_sgd.py`](train_sgd.py).\n\n"
        "### 13.1 Пайплайн\n\n"
        "```\nColumnTransformer:\n"
        "  address  → HashingVectorizer(n_features=1024, ngram=(1,2), norm='l2')\n"
        "  numeric  → StandardScaler  (build_year, floor, total_floors,\n"
        "                              rooms, area_total, area_total²)\n"
        "  cat      → OneHotEncoder   (building_material, series_group)\n"
        "  binary   → passthrough     (is_old, condition_unfinished)\n"
        "  ↓\n"
        "SGDRegressor(loss='huber', penalty='elasticnet')\n"
        "  target = log1p(usd_price)\n```\n\n"
        "**Важно: `lat`/`lon` НЕ участвуют в модели.** Связь координат с ценой нелинейная "
        "(Бишкек — не радиальный город), и в линейке сырые `(lat, lon)` дают мусорный сигнал "
        "(см. §6: pearson ≈ −0.14 / −0.00). Географию полностью забирает `address` через "
        "HashingVectorizer (§10: R²=0.343 на голом адресе vs R²=0.126 на geo_cluster).\n\n"
        "**Зачем именно так:**\n"
        "- `log1p(usd_price)` — таргет сильно скошен (§2), на log распределение почти нормальное → MSE на log = RMSLE.\n"
        "- `area_total²` — единственный полиномиальный признак: проверка в §3 показала,"
        " что зависимость `price ~ area` слегка изогнута (gap Spearman−Pearson ≠ 0).\n"
        "- StandardScaler перед SGD обязателен — иначе градиенты по разным признакам несоразмерны.\n"
        "- Huber loss — устойчив к выбросам, которые мы видели в §8 (хвосты $1.2M).\n"
        "- ElasticNet — у нас 1024 hash-фичей + OHE + численные → нужно отбирать сигнал.\n\n"
        "### 13.2 GridSearch\n\n"
        f"Сетка: `alpha` × `l1_ratio` × `epsilon` = "
        f"{len(out['grid']['alpha'])}×{len(out['grid']['l1_ratio'])}×{len(out['grid']['epsilon'])} "
        f"= {len(out['grid']['alpha']) * len(out['grid']['l1_ratio']) * len(out['grid']['epsilon'])} "
        "конфигураций × 5 фолдов.\n\n"
        "**Лучшие гиперпараметры:**\n\n"
        f"- `alpha = {bp['alpha']}` (сила регуляризации)\n"
        f"- `l1_ratio = {bp['l1_ratio']}` (баланс L1/L2; 0 = чистый Ridge, 1 = чистый Lasso)\n"
        f"- `epsilon = {bp['epsilon']}` (Huber threshold в лог-шкале)\n\n"
        f"Лучший CV RMSE (log) во время grid search: **{out['best_cv_rmse_log']:.4f}**.\n\n"
        "### 13.3 Метрики (out-of-fold, 5-fold)\n\n"
        "| Метрика | Значение |\n"
        "|---|---:|\n"
        f"| R² (log) | {m['log_r2']} |\n"
        f"| RMSE (log) | {m['log_rmse']} |\n"
        f"| MAE (log) | {m['log_mae']} |\n"
        f"| R² (USD) | {m['usd_r2']} |\n"
        f"| RMSE (USD) | ${m['usd_rmse']:,.0f} |\n"
        f"| MAE (USD) | ${m['usd_mae']:,.0f} |\n"
        f"| MAPE | {m['usd_mape_pct']}% |\n"
        f"| median APE | {m['usd_median_ape_pct']}% |\n\n"
        "**Что важно:**\n"
        "- `log_r2` и `RMSE(log)` — на той шкале, на которой модель действительно учится. "
        "Это «честная» точность.\n"
        f"- `median APE = {m['usd_median_ape_pct']}%` показывает типичную ошибку: половина "
        "предсказаний попадает с ошибкой меньше этого процента. Удобно для бизнеса.\n"
        f"- `MAPE = {m['usd_mape_pct']}%` среднее — заметно выше медианы, потому что выбросы "
        "(квартиры за $500k–$1.2M) тянут метрику вверх. На лог-шкале их влияние сглажено.\n\n"
        "### 13.4 Диагностика остатков\n\n"
        "![SGD diagnostics](figs/22_sgd_diagnostics.png)\n\n"
        f"σ остатков = {rd['resid_std']:.3f} в лог-шкале (≈ ±{(np.exp(rd['resid_std'])-1)*100:.0f}% по цене), "
        f"skew = {rd['resid_skew']:.2f}, kurtosis = {rd['resid_kurt']:.2f}.\n\n"
        "**Что видно.**\n"
        "- *Predicted vs Actual (log)*: облако вокруг диагонали, у дорогих объектов "
        "(правый верх) разброс растёт — типичный признак того, что верхний хвост ($500k+) "
        "учится хуже.\n"
        "- *Residuals vs Fitted*: есть слабый веер (гетероскедастичность); "
        "Huber loss её частично гасит, но в верхней части ошибка систематически больше.\n"
        f"- *Распределение остатков*: близко к симметричному, skew={rd['resid_skew']:.2f}. "
        "Тяжёлые хвосты есть (kurtosis > 0) — это «странные» объявления, выбивающиеся из паттерна.\n"
        "- *QQ*: центральная часть на прямой, отклонения на хвостах. Линейная модель видит "
        "большинство объектов нормально, но 1–2% точек сильно вне распределения.\n\n"
        "![Pred vs Actual USD](figs/23_sgd_usd_scatter.png)\n\n"
        "На обычной шкале модель занижает топовые цены (выше $400k), потому что обратное "
        "преобразование `expm1` плюс Huber + ElasticNet → консервативно. Это плата за устойчивость.\n\n"
        "### 13.5 Что попробовать дальше\n\n"
        "1. **Группировать CV по `series_group` или гео-кластеру** — текущий KFold чуть оптимистичен "
        "(один ЖК встречается в нескольких фолдах).\n"
        "2. **Quantile regression** (`loss='epsilon_insensitive'` или отдельная модель на квантили) "
        "если задача — давать диапазон, а не точку.\n"
        "3. **Расширить hash до n_features=4096** — в §10 показано, что выше 1024 R² растёт ещё "
        "на 0.05 на голом адресе.\n"
        "4. **Добавить `area_per_room`, `floor_ratio`, `building_age`** — рекомендации из §9, "
        "не реализованы здесь специально, чтобы оставить место для следующей итерации.\n"
        "5. **Градиентный бустинг** (LightGBM/CatBoost) — будет верхней границей качества; "
        "линейка покажет, сколько мы теряем на простоте.\n"
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

    pipe = build_pipeline()

    grid = {
        "reg__alpha": [1e-5, 1e-4, 1e-3],
        "reg__l1_ratio": [0.15, 0.5, 0.85],
        "reg__epsilon": [0.05, 0.1, 0.5],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    print(f"GridSearch: {len(grid['reg__alpha']) * len(grid['reg__l1_ratio']) * len(grid['reg__epsilon'])} configs × 5 folds")
    gs = GridSearchCV(
        pipe, grid, cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=1,
    )
    gs.fit(X, y)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV RMSE (log): {-gs.best_score_:.4f}")

    print("Computing out-of-fold predictions for diagnostics...")
    y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, n_jobs=-1)

    metrics = metrics_oof(y, y_pred)
    print("\nOOF metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    rd = plot_diagnostics(y, y_pred)

    out = {
        "best_params": {k.replace("reg__", ""): v for k, v in gs.best_params_.items()},
        "best_cv_rmse_log": float(-gs.best_score_),
        "cv_metrics_oof": metrics,
        "residual_diagnostics": rd,
        "feature_config": {
            "text_col": TEXT_COL,
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "binary_cols": BINARY_COLS,
        },
        "preprocessing": {
            "address": "HashingVectorizer(n_features=1024, ngram=(1,2), norm='l2')",
            "numeric": "StandardScaler",
            "categorical": "OneHotEncoder",
            "target_transform": "log1p",
            "polynomial": "area_total_sq = area_total**2",
        },
        "grid": {k.replace("reg__", ""): v for k, v in grid.items()},
        "n_train": int(len(df)),
    }
    Path("model_params.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved model_params.json")

    update_report(out)
    print("Report updated with §13.")


if __name__ == "__main__":
    main()
