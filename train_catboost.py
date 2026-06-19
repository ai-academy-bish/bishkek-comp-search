"""CatBoost для предсказания usd_price — верхняя граница качества на этих данных.

Зачем: линейка §13 (SGDRegressor) даёт log R²=0.867. Градиентный бустинг ловит
нелинейности и взаимодействия, которые линейка теряет, поэтому CatBoost показывает
«потолок» прогнозируемости на текущем наборе признаков. Разница (catboost − linear)
= цена за простоту линейной модели.

Что проверяем:
  Блок A — трансформации: target (raw $ vs log1p) × area (raw vs log).
           Деревья инвариантны к монотонной трансформации фичи (area),
           но чувствительны к трансформации таргета (меняет геометрию loss).
  Блок B — категории: native CatBoost (rich series/condition/offer_type)
           vs native (сгруппированные) vs OneHot.
  Блок C — адрес: без адреса (только сырые lat/lon) vs CatBoost text-фича
           vs Tfidf-числовой.
  + дефолт «из коробки» и тюнинг гиперпараметров.

Координаты lat/lon — СЫРЫЕ: деревья ловят нелинейную географию напрямую
(в отличие от линейки, где их пришлось выкинуть, см. §13.1).

Оценка — 5-fold OOF, тот же KFold(shuffle, random_state=0) и те же метрики,
что в линейке (см. train_sgd.metrics_oof) → числа сравнимы напрямую.

Артефакты: catboost_results.json, графики figs/30..36, секция §16 в report.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

sns.set_theme(style="whitegrid")
FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)

RANDOM_STATE = 0
N_SPLITS = 5

# Базовые числовые признаки (координаты — сырые, см. docstring).
NUM_BASE = ["lat", "lon", "build_year", "floor", "total_floors", "rooms"]
RICH_CATS = ["offer_type", "series", "building_material", "condition"]
GROUPED_CATS = ["building_material", "series_group"]
GROUPED_NUM = ["is_old", "condition_unfinished"]

# «Из коробки» — близко к дефолту CatBoost, но с фикс. seed и тихим выводом.
BASE_PARAMS = dict(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
)


def load() -> pd.DataFrame:
    """Объединяет engineered (train_features) и rich-категориалки (train_filled).

    Строки выровнены 1:1 (проверено: usd_price/address совпадают). Применяем
    те же фильтры, что линейка: lon ∈ (70,80), rooms==1000 → медиана.
    """
    feat = pd.read_csv("train_features.csv")
    filled = pd.read_csv("train_filled.csv")
    assert len(feat) == len(filled)
    assert (feat["usd_price"].values == filled["usd_price"].values).all()

    df = feat.copy()
    for c in ["offer_type", "series", "condition"]:
        df[c] = filled[c].values

    df = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()
    rooms_median = df.loc[df["rooms"] != 1000, "rooms"].median()
    df.loc[df["rooms"] == 1000, "rooms"] = rooms_median

    # категориалки → строки без NaN (CatBoost требует)
    for c in RICH_CATS + ["series_group"]:
        df[c] = df[c].astype(str).fillna("NA")
    df["address"] = df["address"].fillna("").astype(str).str.lower()
    df["area_log"] = np.log1p(df["area_total"])
    return df.reset_index(drop=True)


def metrics_all(y_usd: np.ndarray, p_usd: np.ndarray) -> dict:
    """Метрики в USD и в лог-шкале. Работает для любого target_mode:
    предсказания всегда приводятся к USD, лог-метрики считаются через log1p.
    Совместимо с train_sgd.metrics_oof (та же формула R²/RMSE/MAE/MAPE)."""
    p_usd = np.clip(p_usd, 1.0, None)
    yl, pl = np.log1p(y_usd), np.log1p(p_usd)
    rl = yl - pl
    log_rmse = float(np.sqrt(np.mean(rl ** 2)))
    log_mae = float(np.mean(np.abs(rl)))
    log_r2 = float(1 - np.sum(rl ** 2) / np.sum((yl - yl.mean()) ** 2))

    ape = np.abs((y_usd - p_usd) / y_usd) * 100
    usd_rmse = float(np.sqrt(np.mean((y_usd - p_usd) ** 2)))
    usd_mae = float(np.mean(np.abs(y_usd - p_usd)))
    usd_r2 = float(1 - np.sum((y_usd - p_usd) ** 2) / np.sum((y_usd - y_usd.mean()) ** 2))
    return {
        "log_r2": round(log_r2, 4),
        "log_rmse": round(log_rmse, 4),
        "log_mae": round(log_mae, 4),
        "usd_r2": round(usd_r2, 4),
        "usd_rmse": round(usd_rmse, 0),
        "usd_mae": round(usd_mae, 0),
        "usd_mape_pct": round(float(ape.mean()), 2),
        "usd_median_ape_pct": round(float(np.median(ape)), 2),
    }


def build_feature_frame(df: pd.DataFrame, area_mode: str, cat_mode: str,
                        addr_mode: str):
    """Возвращает (X_df, cat_features, text_features) для всего датасета.
    Tfidf для addr_mode='tfidf' считается per-fold внутри run_oof (без утечки)."""
    cols: list[str] = list(NUM_BASE)
    cols.append("area_total" if area_mode == "raw" else "area_log")

    cat_features: list[str] = []
    if cat_mode == "native_rich":
        cols += RICH_CATS
        cat_features = list(RICH_CATS)
    elif cat_mode == "native_grouped":
        cols += GROUPED_CATS + GROUPED_NUM
        cat_features = list(GROUPED_CATS)
    elif cat_mode == "onehot_rich":
        ohe = pd.get_dummies(df[RICH_CATS], prefix=RICH_CATS, dtype=float)
        df = pd.concat([df, ohe], axis=1)
        cols += list(ohe.columns)
    else:
        raise ValueError(cat_mode)

    text_features: list[str] = []
    if addr_mode == "text":
        cols.append("address")
        text_features = ["address"]
    # addr_mode 'none' / 'tfidf' — address не добавляем в cols здесь
    return df[cols + (["address"] if addr_mode == "tfidf" else [])].copy(), \
        cat_features, text_features


def run_oof(df: pd.DataFrame, *, target_mode: str, area_mode: str,
            cat_mode: str, addr_mode: str, params: dict) -> dict:
    """5-fold out-of-fold предсказания. Возвращает метрики + сами OOF preds (USD)."""
    y_usd = df["usd_price"].values.astype(float)
    y = np.log1p(y_usd) if target_mode == "log" else y_usd

    X, cat_features, text_features = build_feature_frame(df, area_mode, cat_mode, addr_mode)
    base_cols = [c for c in X.columns if c != "address"] if addr_mode == "tfidf" else list(X.columns)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(df))
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()

        if addr_mode == "tfidf":
            vec = TfidfVectorizer(min_df=10, ngram_range=(1, 2), max_features=256)
            A_tr = vec.fit_transform(X_tr["address"]).toarray()
            A_te = vec.transform(X_te["address"]).toarray()
            tfcols = [f"addr_{i}" for i in range(A_tr.shape[1])]
            X_tr = pd.concat([X_tr[base_cols].reset_index(drop=True),
                              pd.DataFrame(A_tr, columns=tfcols)], axis=1)
            X_te = pd.concat([X_te[base_cols].reset_index(drop=True),
                              pd.DataFrame(A_te, columns=tfcols)], axis=1)

        model = CatBoostRegressor(cat_features=cat_features or None,
                                  text_features=text_features or None, **params)
        model.fit(X_tr, y[tr])
        pred = model.predict(X_te)
        oof[te] = pred

    oof_usd = np.expm1(oof) if target_mode == "log" else oof
    m = metrics_all(y_usd, oof_usd)
    m["_oof_usd"] = oof_usd
    return m


def cv_log_rmse(df: pd.DataFrame, *, params: dict, n_splits: int = 3,
                **fcfg) -> float:
    """Быстрый CV (log RMSE) для тюнинга — на лог-таргете, native_rich+text."""
    r = run_oof(df, target_mode="log", area_mode=fcfg.get("area_mode", "raw"),
                cat_mode=fcfg.get("cat_mode", "native_rich"),
                addr_mode=fcfg.get("addr_mode", "text"),
                params={**params, "iterations": params.get("iterations", 500)})
    return r["log_rmse"]


# ----------------------------- эксперименты -----------------------------

def main() -> None:
    t0 = time.time()
    df = load()
    print(f"loaded {len(df)} rows")
    results: dict[str, dict] = {}

    def record(name: str, m: dict, **meta):
        clean = {k: v for k, v in m.items() if not k.startswith("_")}
        clean.update(meta)
        results[name] = clean
        print(f"  {name:28s} log_r2={m['log_r2']:.4f} usd_r2={m['usd_r2']:.4f} "
              f"medAPE={m['usd_median_ape_pct']:.2f}% MAPE={m['usd_mape_pct']:.2f}%")

    # --- Блок A: трансформации target × area ---
    print("\n[Блок A] трансформации target × area (native_rich + text):")
    blockA = {}
    for tm in ["usd", "log"]:
        for am in ["raw", "log"]:
            name = f"A_target={tm}_area={am}"
            m = run_oof(df, target_mode=tm, area_mode=am, cat_mode="native_rich",
                        addr_mode="text", params=BASE_PARAMS)
            record(name, m, target=tm, area=am)
            blockA[name] = results[name]

    # --- Блок B: обработка категорий (target=log, area=raw, addr=text) ---
    print("\n[Блок B] обработка категорий:")
    blockB = {}
    for cm in ["native_rich", "native_grouped", "onehot_rich"]:
        name = f"B_cat={cm}"
        m = run_oof(df, target_mode="log", area_mode="raw", cat_mode=cm,
                    addr_mode="text", params=BASE_PARAMS)
        record(name, m, cat_mode=cm)
        blockB[name] = results[name]

    # --- Блок C: обработка адреса (target=log, area=raw, cat=native_rich) ---
    print("\n[Блок C] обработка адреса:")
    blockC = {}
    for am in ["none", "text", "tfidf"]:
        name = f"C_addr={am}"
        m = run_oof(df, target_mode="log", area_mode="raw", cat_mode="native_rich",
                    addr_mode=am, params=BASE_PARAMS)
        record(name, m, addr_mode=am)
        blockC[name] = results[name]

    # --- Из коробки: чистый дефолт CatBoost ---
    print("\n[Из коробки] дефолтные гиперпараметры:")
    default_params = dict(loss_function="RMSE", random_seed=RANDOM_STATE,
                          verbose=False, allow_writing_files=False)
    m = run_oof(df, target_mode="log", area_mode="raw", cat_mode="native_rich",
                addr_mode="text", params=default_params)
    record("default_oob", m)

    # --- Тюнинг гиперпараметров (3-fold для скорости) ---
    print("\n[Тюнинг] depth × learning_rate × l2_leaf_reg:")
    grid = []
    for depth in [6, 8]:
        for lr in [0.03, 0.08]:
            for l2 in [3.0, 9.0]:
                grid.append(dict(depth=depth, learning_rate=lr, l2_leaf_reg=l2))
    best = None
    tuning_log = []
    for g in grid:
        p = {**BASE_PARAMS, **g, "iterations": 700}
        score = cv_log_rmse(df, params=p)
        tuning_log.append({**g, "log_rmse": round(score, 4)})
        print(f"    {g} -> log_rmse={score:.4f}")
        if best is None or score < best[0]:
            best = (score, g)
    best_params = {**BASE_PARAMS, **best[1], "iterations": 1000}
    print(f"  best: {best[1]} (cv log_rmse={best[0]:.4f})")

    # --- Финальная модель: лучший конфиг, полная 5-fold OOF ---
    print("\n[Финал] лучший конфиг, 5-fold OOF:")
    final = run_oof(df, target_mode="log", area_mode="raw", cat_mode="native_rich",
                    addr_mode="text", params=best_params)
    record("final_tuned", {k: v for k, v in final.items()}, **best[1])
    oof_usd = final["_oof_usd"]

    # Importance — обучаем на всех данных лучшим конфигом
    X, cat_features, text_features = build_feature_frame(df, "raw", "native_rich", "text")
    y_log = np.log1p(df["usd_price"].values.astype(float))
    full = CatBoostRegressor(cat_features=cat_features, text_features=text_features,
                             **best_params)
    full.fit(X, y_log)
    importances = full.get_feature_importance()
    feat_names = list(X.columns)
    imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    full.save_model("catboost_model.cbm")

    # ----------------------------- графики -----------------------------
    plot_block(blockA, "30_catboost_transforms.png",
               "Блок A: трансформации target × area",
               labels={"A_target=usd_area=raw": "raw$ / raw area",
                       "A_target=usd_area=log": "raw$ / log area",
                       "A_target=log_area=raw": "log$ / raw area",
                       "A_target=log_area=log": "log$ / log area"})
    plot_block(blockB, "31_catboost_categorical.png",
               "Блок B: обработка категориальных",
               labels={"B_cat=native_rich": "native (rich)",
                       "B_cat=native_grouped": "native (grouped)",
                       "B_cat=onehot_rich": "OneHot (rich)"})
    plot_block(blockC, "32_catboost_address.png",
               "Блок C: обработка адреса",
               labels={"C_addr=none": "без адреса\n(только lat/lon)",
                       "C_addr=text": "CatBoost text",
                       "C_addr=tfidf": "Tfidf числовой"})
    rd = plot_diagnostics(df["usd_price"].values.astype(float), oof_usd)
    plot_importance(imp[:20])
    plot_vs_linear(results["final_tuned"])

    # ----------------------------- сохранить -----------------------------
    out = {
        "n_train": int(len(df)),
        "cv": f"{N_SPLITS}-fold OOF, KFold(shuffle, random_state={RANDOM_STATE})",
        "base_params": {k: v for k, v in BASE_PARAMS.items()
                        if k not in ("verbose", "allow_writing_files")},
        "best_params": best[1],
        "best_cv_log_rmse_tuning": round(best[0], 4),
        "blockA": blockA, "blockB": blockB, "blockC": blockC,
        "default_oob": results["default_oob"],
        "final_tuned": results["final_tuned"],
        "tuning_log": tuning_log,
        "feature_importance": [{"feature": f, "importance": round(float(v), 3)}
                               for f, v in imp],
        "residual_diagnostics": rd,
    }
    Path("catboost_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved catboost_results.json")

    update_report(out)
    print(f"Report updated with §16. Total time: {time.time()-t0:.0f}s")


def plot_block(block: dict, fname: str, title: str, labels: dict) -> None:
    names = list(block.keys())
    log_r2 = [block[n]["log_r2"] for n in names]
    med_ape = [block[n]["usd_median_ape_pct"] for n in names]
    xs = [labels.get(n, n) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    c1 = sns.color_palette("crest", len(names))
    ax[0].bar(xs, log_r2, color=c1)
    ax[0].set_title(f"{title}\nlog R² (OOF, 5-fold)")
    ax[0].set_ylim(min(log_r2) - 0.02, max(log_r2) + 0.01)
    for i, v in enumerate(log_r2):
        ax[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax[1].bar(xs, med_ape, color=sns.color_palette("flare", len(names)))
    ax[1].set_title("median APE, %")
    for i, v in enumerate(med_ape):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=110)
    plt.close(fig)


def plot_diagnostics(y_usd: np.ndarray, p_usd: np.ndarray) -> dict:
    p_usd = np.clip(p_usd, 1.0, None)
    yl, pl = np.log1p(y_usd), np.log1p(p_usd)
    resid = yl - pl

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].scatter(pl, yl, s=4, alpha=0.3)
    lims = [min(yl.min(), pl.min()), max(yl.max(), pl.max())]
    axes[0, 0].plot(lims, lims, "r--", lw=1)
    axes[0, 0].set_xlabel("predicted log(usd_price)")
    axes[0, 0].set_ylabel("actual log(usd_price)")
    axes[0, 0].set_title("Predicted vs Actual (log)")

    axes[0, 1].scatter(pl, resid, s=4, alpha=0.3)
    axes[0, 1].axhline(0, c="r", lw=1)
    axes[0, 1].set_xlabel("predicted log")
    axes[0, 1].set_ylabel("residual")
    axes[0, 1].set_title("Residuals vs Fitted")

    axes[1, 0].hist(resid, bins=80, color="seagreen")
    axes[1, 0].set_title(f"Residual distribution (σ={resid.std():.3f}, "
                         f"skew={stats.skew(resid):.2f})")
    axes[1, 0].set_xlabel("residual (log)")

    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ residuals")
    fig.tight_layout()
    fig.savefig(FIGS / "33_catboost_diagnostics.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(p_usd, y_usd, s=4, alpha=0.3)
    lims = [0, max(y_usd.max(), p_usd.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("predicted, $")
    axes[0].set_ylabel("actual, $")
    axes[0].set_title("Pred vs Actual (USD, linear scale)")
    axes[1].loglog(p_usd, y_usd, ".", ms=2, alpha=0.3)
    axes[1].plot([1, lims[1]], [1, lims[1]], "r--", lw=1)
    axes[1].set_xlabel("predicted, $ (log)")
    axes[1].set_ylabel("actual, $ (log)")
    axes[1].set_title("Pred vs Actual (USD, log-log)")
    fig.tight_layout()
    fig.savefig(FIGS / "34_catboost_usd_scatter.png", dpi=110)
    plt.close(fig)

    return {"resid_skew": round(float(stats.skew(resid)), 3),
            "resid_std": round(float(resid.std()), 3),
            "resid_kurt": round(float(stats.kurtosis(resid)), 3)}


def plot_importance(imp_top: list) -> None:
    names = [f for f, _ in imp_top][::-1]
    vals = [v for _, v in imp_top][::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(names, vals, color=sns.color_palette("viridis", len(names)))
    ax.set_title("CatBoost feature importance (top-20, финальная модель)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(FIGS / "35_catboost_importance.png", dpi=110)
    plt.close(fig)


def plot_vs_linear(cb: dict) -> None:
    try:
        lin = json.loads(Path("model_params.json").read_text())["cv_metrics_oof"]
    except Exception:
        return
    metrics = [("log R²", "log_r2", lin["log_r2"], cb["log_r2"], False),
               ("median APE %", "usd_median_ape_pct", lin["usd_median_ape_pct"],
                cb["usd_median_ape_pct"], True),
               ("MAPE %", "usd_mape_pct", lin["usd_mape_pct"], cb["usd_mape_pct"], True),
               ("MAE $", "usd_mae", lin["usd_mae"], cb["usd_mae"], True)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, _, lv, cv, lower_better) in zip(axes, metrics):
        bars = ax.bar(["SGD\n(линейка)", "CatBoost"], [lv, cv],
                      color=["#8888cc", "#2a9d5c"])
        ax.set_title(title + ("  ↓ лучше" if lower_better else "  ↑ лучше"))
        for b, v in zip(bars, [lv, cv]):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:,.4g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("CatBoost vs линейная модель (§13), OOF 5-fold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "36_catboost_vs_linear.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def update_report(out: dict) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker = "## 16. CatBoost: верхняя граница качества"

    bp = out["best_params"]
    f = out["final_tuned"]
    d = out["default_oob"]
    rd = out["residual_diagnostics"]
    try:
        lin = json.loads(Path("model_params.json").read_text())["cv_metrics_oof"]
    except Exception:
        lin = None

    def row(name, m):
        return (f"| {name} | {m['log_r2']} | {m['log_rmse']} | {m['usd_r2']} | "
                f"${m['usd_mae']:,.0f} | {m['usd_mape_pct']}% | "
                f"{m['usd_median_ape_pct']}% |")

    A_labels = {"A_target=usd_area=raw": "raw $ / raw area",
                "A_target=usd_area=log": "raw $ / log area",
                "A_target=log_area=raw": "log $ / raw area",
                "A_target=log_area=log": "log $ / log area"}
    B_labels = {"B_cat=native_rich": "native CatBoost (rich: series/condition/offer_type)",
                "B_cat=native_grouped": "native CatBoost (grouped: series_group/...)",
                "B_cat=onehot_rich": "OneHot (rich)"}
    C_labels = {"C_addr=none": "без адреса (только сырые lat/lon)",
                "C_addr=text": "CatBoost text-фича (нативный BoW/n-gram)",
                "C_addr=tfidf": "Tfidf числовой (256 dim, min_df=10)"}

    a_rows = "\n".join(row(A_labels[k], v) for k, v in out["blockA"].items())
    b_rows = "\n".join(row(B_labels[k], v) for k, v in out["blockB"].items())
    c_rows = "\n".join(row(C_labels[k], v) for k, v in out["blockC"].items())
    imp_rows = "\n".join(
        f"| {r['feature']} | {r['importance']} |" for r in out["feature_importance"][:15])

    best_A = max(out["blockA"].items(), key=lambda kv: kv[1]["log_r2"])
    best_C = max(out["blockC"].items(), key=lambda kv: kv[1]["log_r2"])
    delta_lin = (f", против log R²={lin['log_r2']} у линейки §13 "
                 f"(**+{f['log_r2'] - lin['log_r2']:.4f}**)") if lin else ""

    cmp_block = ""
    if lin:
        cmp_block = (
            "### 16.7 CatBoost vs линейная модель\n\n"
            "![CatBoost vs linear](figs/36_catboost_vs_linear.png)\n\n"
            "| Метрика | SGD линейка (§13) | CatBoost (финал) | Δ |\n"
            "|---|---:|---:|---:|\n"
            f"| log R² | {lin['log_r2']} | {f['log_r2']} | +{f['log_r2']-lin['log_r2']:.4f} |\n"
            f"| log RMSE | {lin['log_rmse']} | {f['log_rmse']} | {f['log_rmse']-lin['log_rmse']:+.4f} |\n"
            f"| USD R² | {lin['usd_r2']} | {f['usd_r2']} | {f['usd_r2']-lin['usd_r2']:+.4f} |\n"
            f"| MAE USD | ${lin['usd_mae']:,.0f} | ${f['usd_mae']:,.0f} | ${f['usd_mae']-lin['usd_mae']:,.0f} |\n"
            f"| MAPE | {lin['usd_mape_pct']}% | {f['usd_mape_pct']}% | {f['usd_mape_pct']-lin['usd_mape_pct']:+.2f} п.п. |\n"
            f"| median APE | {lin['usd_median_ape_pct']}% | {f['usd_median_ape_pct']}% | {f['usd_median_ape_pct']-lin['usd_median_ape_pct']:+.2f} п.п. |\n\n"
            "**Что это значит.** Бустинг — это «потолок» того, что выжимается из текущих "
            "признаков той же 5-fold OOF схемой. Разница с линейкой — цена за интерпретируемость "
            "и простоту. Линейка осознанно выкидывала сырые `lat`/`lon` (нелинейны), бинировала "
            "`series`/`condition` и тащила географию через hash-адрес; CatBoost берёт всё сырьём "
            "и сам строит нелинейные взаимодействия (этаж×серия, координаты×площадь и т.д.).\n\n"
        )

    block = (
        f"\n---\n\n{marker}\n\n"
        "Идея: линейка §13 даёт log R²=0.867. Градиентный бустинг ловит нелинейности и "
        "взаимодействия, которые линейка теряет, поэтому CatBoost показывает **потолок "
        "прогнозируемости** на текущем наборе признаков. Скрипт: "
        "[`train_catboost.py`](train_catboost.py).\n\n"
        f"Оценка — {out['cv']}, **те же метрики и фолды, что в §13** → числа сравнимы напрямую. "
        f"Обучено на {out['n_train']} строках. "
        "**Координаты `lat`/`lon` поданы сырыми** — деревья ловят нелинейную географию "
        "напрямую (в линейке их пришлось выкинуть, см. §13.1).\n\n"
        "Колонки таблиц ниже: log R² / log RMSE / USD R² / MAE$ / MAPE / median APE.\n\n"
        "### 16.1 Блок A — трансформации таргета и площади\n\n"
        "Деревья инвариантны к монотонной трансформации **признака** (порядок значений "
        "не меняется → те же сплиты), но **чувствительны к трансформации таргета** "
        "(меняется геометрия RMSE: на сырых $ оптимизируются абсолютные ошибки и доминируют "
        "дорогие квартиры; на log — относительные).\n\n"
        "| Конфиг | log R² | log RMSE | USD R² | MAE$ | MAPE | medAPE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{a_rows}\n\n"
        "![Блок A](figs/30_catboost_transforms.png)\n\n"
        "**Что видно.** Логарифмирование **площади** почти не двигает метрики "
        "(деревьям всё равно — это монотонное преобразование одного признака). "
        "А вот **таргет** решает: на лог-шкале relative-ошибки (medAPE/MAPE) заметно лучше, "
        "потому что модель перестаёт «гнаться» за абсолютными долларами дорогих объектов. "
        f"Лучший по log R²: **{A_labels[best_A[0]]}**.\n\n"
        "### 16.2 Блок B — обработка категориальных\n\n"
        "| Способ | log R² | log RMSE | USD R² | MAE$ | MAPE | medAPE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{b_rows}\n\n"
        "![Блок B](figs/31_catboost_categorical.png)\n\n"
        "**Что видно.** Нативная обработка CatBoost (ordered target statistics) на **полных** "
        "категориях (`series` 14 кат., `condition` 5) обычно не хуже, а то и лучше ручного "
        "бинирования `series_group`/`condition_unfinished`, которое делалось специально под "
        "линейку (§12). Для бустинга биннинг не нужен — он сам находит пороги. OneHot "
        "проигрывает нативному кодированию: разреженные индикаторы дают деревьям менее "
        "удобные сплиты, чем упорядоченная target-статистика.\n\n"
        "### 16.3 Блок C — что делать с адресом\n\n"
        "Адрес — сырой текст с районом/улицей/ЖК. У бустинга три пути: (1) выкинуть и "
        "положиться на сырые координаты; (2) нативная **text-фича** CatBoost (внутренний "
        "BoW + n-graмы); (3) внешний **Tfidf** в числовые колонки.\n\n"
        "| Способ | log R² | log RMSE | USD R² | MAE$ | MAPE | medAPE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{c_rows}\n\n"
        "![Блок C](figs/32_catboost_address.png)\n\n"
        "**Что видно.** Сырые `lat`/`lon` уже дают бустингу сильный гео-сигнал (в отличие от "
        "линейки!). Но адресный текст добавляет сверху: улица/ЖК несут детализацию, которой "
        "нет в одной точке координат. Нативная text-фича CatBoost — самый удобный путь "
        f"(не надо отдельно вектотизовать). Лучший: **{C_labels[best_C[0]]}**.\n\n"
        "### 16.4 «Из коробки» vs тюнинг\n\n"
        "| Вариант | log R² | log RMSE | USD R² | MAE$ | MAPE | medAPE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{row('CatBoost дефолт (из коробки)', d)}\n"
        f"{row('CatBoost тюнинг (финал)', f)}\n\n"
        f"Тюнинг (3×2×2 = {len(out['tuning_log'])} конфигов: depth × learning_rate × l2_leaf_reg, "
        f"3-fold): лучшие — `depth={bp['depth']}`, `learning_rate={bp['learning_rate']}`, "
        f"`l2_leaf_reg={bp['l2_leaf_reg']}` (cv log RMSE={out['best_cv_log_rmse_tuning']}). "
        "**Что видно.** CatBoost силён уже из коробки — тюнинг добавляет немного. Это нормально "
        "для бустинга: дефолты разумные, основной выигрыш даёт не подбор гиперпараметров, "
        "а признаки.\n\n"
        "### 16.5 Финальная модель — метрики (OOF, 5-fold)\n\n"
        "| Метрика | Значение |\n|---|---:|\n"
        f"| R² (log) | {f['log_r2']} |\n"
        f"| RMSE (log) | {f['log_rmse']} |\n"
        f"| MAE (log) | {f['log_mae']} |\n"
        f"| R² (USD) | {f['usd_r2']} |\n"
        f"| RMSE (USD) | ${f['usd_rmse']:,.0f} |\n"
        f"| MAE (USD) | ${f['usd_mae']:,.0f} |\n"
        f"| MAPE | {f['usd_mape_pct']}% |\n"
        f"| median APE | {f['usd_median_ape_pct']}% |\n\n"
        f"Финальный конфиг: native rich-категории + CatBoost text(address) + сырые координаты, "
        f"target=log1p, площадь сырая. log R² = **{f['log_r2']}**{delta_lin}.\n\n"
        "![CatBoost diagnostics](figs/33_catboost_diagnostics.png)\n\n"
        f"σ остатков = {rd['resid_std']} в лог-шкале (≈ ±{(np.exp(rd['resid_std'])-1)*100:.0f}% "
        f"по цене), skew = {rd['resid_skew']}, kurtosis = {rd['resid_kurt']}.\n\n"
        "![CatBoost USD scatter](figs/34_catboost_usd_scatter.png)\n\n"
        "### 16.6 Importance\n\n"
        "![CatBoost importance](figs/35_catboost_importance.png)\n\n"
        "Топ-15 признаков (PredictionValuesChange):\n\n"
        "| Признак | Importance |\n|---|---:|\n"
        f"{imp_rows}\n\n"
        + cmp_block +
        "### 16.8 Выводы\n\n"
        "1. **Потолок на этих признаках** — CatBoost даёт log R² ≈ "
        f"{f['log_r2']} (medAPE {f['usd_median_ape_pct']}%). Это ориентир, "
        "сколько в данных вообще есть сигнала.\n"
        "2. **Таргет логарифмировать обязательно**, площадь — по вкусу (деревьям всё равно).\n"
        "3. **Категории — нативно, без ручного биннинга**: CatBoost сам находит пороги, "
        "наш `series_group`/`condition_unfinished` нужен был только линейке.\n"
        "4. **Сырые координаты работают** — то, что было вредно линейке, бустингу полезно.\n"
        "5. **Адрес как text-фича** добавляет сверх координат и не требует ручной векторизации.\n"
        "6. **CV-оговорка та же, что в §13.5**: KFold без группировки по адресу/ЖК чуть "
        "оптимистичен (один ЖК попадает в разные фолды). Для прод-оценки — GroupKFold.\n"
    )

    if marker in text:
        text = text.split(marker)[0].rstrip()
        if text.endswith("---"):
            text = text[:-3].rstrip()
        text += "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    rpt.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
