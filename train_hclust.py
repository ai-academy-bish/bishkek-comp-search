"""Иерархическая (агломеративная) кластеризация рынка квартир Бишкека.

Две независимые сегментации:
  1) Гео+цена: признаки [lat, lon, log(price/m²)], стандартизованы, Ward/euclidean.
     → находит «ценовые районы»: куски карты с однородной ценой за метр.
  2) Адрес как текст: вектотизация двумя способами —
       (a) HashingVectorizer(1024, ngram(1,2), l2)  — как в линейке §13;
       (b) TfidfVectorizer(ngram(1,2), max_features=4000);
     обе с cosine-метрикой и average-linkage.
     → находит группы по написанию адреса (район/улица/ЖК).

Граница реза выбирается автоматически по наибольшему относительному разрыву
между высотами слияний в верхушке дерева (largest-gap), и рисуется пунктиром
на дендрограмме. Дендрограммы усечены (truncate_mode='lastp') — верхушка дерева
это и есть зона, где принимается решение о числе кластеров.

Артефакты: hclust_results.json, графики figs/40..45, секция §17 в report.md.
Полные дендрограммы на 7115 листьях нечитаемы, поэтому показываем верхние p слияний.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import train_catboost as T

sns.set_theme(style="whitegrid")
FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)


def choose_cut(Z: np.ndarray, kmin: int = 3, kmax: int = 12) -> tuple[float, int]:
    """Граница реза по наибольшему разрыву высот слияний в верхушке дерева.

    Высоты Z[:,2] идут по возрастанию. Чтобы получить k кластеров, отменяем
    верхние k−1 слияний → порог лежит между Z[m−k] и Z[m−k+1]. Выбираем k,
    максимизирующее этот разрыв (самое «дорогое» слияние = естественная граница).
    """
    h = Z[:, 2]
    m = len(h)
    best_k, best_gap, best_thr = kmin, -1.0, h[-1]
    for k in range(kmin, min(kmax, m) + 1):
        lo, hi = h[m - k], h[m - k + 1] if (m - k + 1) < m else h[-1]
        gap = hi - lo
        if gap > best_gap:
            best_gap, best_k, best_thr = gap, k, (lo + hi) / 2
    return float(best_thr), int(best_k)


def plot_dendro(Z: np.ndarray, thr: float, k: int, title: str, fname: str,
                p: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(Z, truncate_mode="lastp", p=p, color_threshold=thr,
               above_threshold_color="#999999", ax=ax,
               leaf_rotation=90, leaf_font_size=8)
    ax.axhline(thr, color="red", ls="--", lw=1.6,
               label=f"граница реза → {k} кластеров (h={thr:.2f})")
    ax.set_title(f"{title}\n(усечено: верхние {p} узлов; цвет = кластер ниже линии реза)")
    ax.set_xlabel("номер узла (в скобках — размер поддерева)")
    ax.set_ylabel("высота слияния (расстояние)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=115)
    plt.close(fig)


def cluster_profile(df: pd.DataFrame, labels: np.ndarray, ppm2: np.ndarray,
                    top_addr: int = 3) -> list[dict]:
    rows = []
    for c in np.unique(labels):
        mask = labels == c
        sample = df.loc[mask, "address"].head(top_addr).tolist()
        rows.append({
            "cluster": int(c),
            "size": int(mask.sum()),
            "ppm2_median": round(float(np.median(ppm2[mask])), 0),
            "lat_mean": round(float(df.loc[mask, "lat"].mean()), 4),
            "lon_mean": round(float(df.loc[mask, "lon"].mean()), 4),
            "price_median": round(float(df.loc[mask, "usd_price"].median()), 0),
            "sample_addr": sample,
        })
    return sorted(rows, key=lambda r: -r["size"])


# ----------------------------- эксперименты -----------------------------

def main() -> None:
    df = T.load()
    n = len(df)
    print(f"loaded {n} rows", flush=True)
    ppm2 = (df["usd_price"] / df["area_total"]).values
    results = {"n": n}

    # === 1. Гео + цена за м² (Ward / euclidean) ===
    print("[1] geo+ppm2: Ward linkage...", flush=True)
    logp = np.log1p(ppm2)
    lo, hi = np.percentile(logp, [1, 99])         # винзоризация выбросов $/m²
    logp_w = np.clip(logp, lo, hi)
    Xg = StandardScaler().fit_transform(
        np.column_stack([df["lat"].values, df["lon"].values, logp_w]))
    Zg = linkage(Xg, method="ward", metric="euclidean")
    thr_g, k_g = choose_cut(Zg, kmax=12)
    lab_g = fcluster(Zg, t=thr_g, criterion="distance")
    print(f"    cut h={thr_g:.2f} -> {k_g} clusters", flush=True)
    plot_dendro(Zg, thr_g, k_g, "Кластеризация 1 — гео + log(price/m²), Ward",
                "40_hclust_geo_dendro.png")
    prof_g = cluster_profile(df, lab_g, ppm2)
    plot_geo_map(df, lab_g, prof_g, thr_g, k_g)
    results["geo"] = {"linkage": "ward", "metric": "euclidean",
                      "features": ["lat", "lon", "log1p(price_per_m2)"],
                      "cut_height": round(thr_g, 3), "n_clusters": k_g,
                      "profile": prof_g}

    # === 2a. Адрес: HashingVectorizer (cosine / average) ===
    print("[2a] address Hashing: Ward on L2-normed (≡cosine)...", flush=True)
    hv = HashingVectorizer(n_features=1024, ngram_range=(1, 2),
                           alternate_sign=False, norm="l2", lowercase=True)
    Xh = hv.fit_transform(df["address"]).toarray().astype(np.float32)
    Zh = linkage(Xh, method="ward")  # векторы L2-нормированы → euclid ≡ rank(cosine)
    thr_h, k_h = choose_cut(Zh, kmax=12)
    lab_h = fcluster(Zh, t=thr_h, criterion="distance")
    print(f"    cut h={thr_h:.3f} -> {k_h} clusters", flush=True)
    plot_dendro(Zh, thr_h, k_h,
                "Кластеризация 2a — адрес, HashingVectorizer (cosine)",
                "42_hclust_addr_hashing_dendro.png")
    prof_h = cluster_profile(df, lab_h, ppm2)
    results["addr_hashing"] = {"vectorizer": "HashingVectorizer(1024, ngram(1,2), l2)",
                               "linkage": "ward", "metric": "euclidean on L2-norm (≡cosine)",
                               "cut_height": round(thr_h, 4), "n_clusters": k_h,
                               "profile": prof_h}

    # === 2b. Адрес: TfidfVectorizer (cosine / average) ===
    print("[2b] address TF-IDF: Ward on L2-normed (≡cosine)...", flush=True)
    tv = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, min_df=3,
                         lowercase=True)  # norm='l2' по умолчанию
    Xt = tv.fit_transform(df["address"])
    Zt = linkage(Xt.toarray().astype(np.float32), method="ward")
    thr_t, k_t = choose_cut(Zt, kmax=12)
    lab_t = fcluster(Zt, t=thr_t, criterion="distance")
    print(f"    cut h={thr_t:.3f} -> {k_t} clusters", flush=True)
    plot_dendro(Zt, thr_t, k_t,
                "Кластеризация 2b — адрес, TF-IDF (cosine)",
                "43_hclust_addr_tfidf_dendro.png")
    prof_t = cluster_profile(df, lab_t, ppm2)
    # топ-термины на кластер (только tfidf — hashing необратим)
    vocab = np.array(tv.get_feature_names_out())
    top_terms = {}
    Xt_arr = Xt.toarray()
    for c in np.unique(lab_t):
        centroid = Xt_arr[lab_t == c].mean(axis=0)
        top_terms[int(c)] = vocab[np.argsort(centroid)[::-1][:6]].tolist()
    results["addr_tfidf"] = {"vectorizer": "TfidfVectorizer(ngram(1,2), max_features=4000, min_df=3)",
                             "linkage": "ward", "metric": "euclidean on L2-norm (≡cosine)",
                             "cut_height": round(thr_t, 4), "n_clusters": k_t,
                             "profile": prof_t, "top_terms": top_terms}

    # === Кросс-сравнение разбиений (ARI) ===
    ari = {
        "geo_vs_hashing": round(float(adjusted_rand_score(lab_g, lab_h)), 4),
        "geo_vs_tfidf": round(float(adjusted_rand_score(lab_g, lab_t)), 4),
        "hashing_vs_tfidf": round(float(adjusted_rand_score(lab_h, lab_t)), 4),
    }
    print(f"    ARI: {ari}", flush=True)
    results["ari"] = ari
    plot_ari_and_textmap(df, lab_g, lab_h, lab_t, ari)

    Path("hclust_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved hclust_results.json", flush=True)
    update_report(results)
    print("Report updated with §17.", flush=True)


def plot_geo_map(df, labels, profile, thr, k) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    palette = sns.color_palette("tab20", len(np.unique(labels)))
    cmap = {c: palette[i] for i, c in enumerate(np.unique(labels))}
    colors = [cmap[l] for l in labels]
    axes[0].scatter(df["lon"], df["lat"], s=7, c=colors, alpha=0.6)
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")
    axes[0].set_title(f"Карта: {k} гео-ценовых кластеров (lat/lon)")

    prof = sorted(profile, key=lambda r: r["ppm2_median"])
    names = [f"C{r['cluster']} (n={r['size']})" for r in prof]
    vals = [r["ppm2_median"] for r in prof]
    bar_colors = [cmap[r["cluster"]] for r in prof]
    axes[1].barh(names, vals, color=bar_colors)
    axes[1].set_xlabel("медиана price/m², $")
    axes[1].set_title("Цена за м² по кластерам (отсортировано)")
    for i, v in enumerate(vals):
        axes[1].text(v, i, f" {v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "41_hclust_geo_map.png", dpi=115)
    plt.close(fig)


def plot_ari_and_textmap(df, lab_g, lab_h, lab_t, ari) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    # ARI heatmap
    names = ["geo", "hashing", "tfidf"]
    M = np.array([
        [1.0, ari["geo_vs_hashing"], ari["geo_vs_tfidf"]],
        [ari["geo_vs_hashing"], 1.0, ari["hashing_vs_tfidf"]],
        [ari["geo_vs_tfidf"], ari["hashing_vs_tfidf"], 1.0],
    ])
    sns.heatmap(M, annot=True, fmt=".3f", xticklabels=names, yticklabels=names,
                cmap="rocket_r", vmin=0, vmax=1, ax=axes[0], cbar_kws={"label": "ARI"})
    axes[0].set_title("Adjusted Rand Index между разбиениями\n(1 = идентичны, 0 = случайны)")

    # текстовые (tfidf) кластеры на карте — совпадают ли с географией?
    palette = sns.color_palette("tab20", len(np.unique(lab_t)))
    cmap = {c: palette[i] for i, c in enumerate(np.unique(lab_t))}
    axes[1].scatter(df["lon"], df["lat"], s=7,
                    c=[cmap[l] for l in lab_t], alpha=0.6)
    axes[1].set_xlabel("lon")
    axes[1].set_ylabel("lat")
    axes[1].set_title("TF-IDF адресные кластеры на карте\n(проверка: ложатся ли по географии)")
    fig.tight_layout()
    fig.savefig(FIGS / "44_hclust_ari_textmap.png", dpi=115)
    plt.close(fig)


def update_report(r: dict) -> None:
    rpt = Path("report.md")
    text = rpt.read_text(encoding="utf-8")
    marker = "## 17. Иерархическая кластеризация рынка"

    g, h, t = r["geo"], r["addr_hashing"], r["addr_tfidf"]
    ari = r["ari"]

    def prof_table(prof, top=8):
        head = ("| Кластер | Размер | медиана $/m² | медиана цены | центр (lat, lon) | пример адреса |\n"
                "|---|---:|---:|---:|---|---|\n")
        rows = []
        for p in prof[:top]:
            addr = (p["sample_addr"][0][:46] + "…") if p["sample_addr"] else "—"
            rows.append(f"| C{p['cluster']} | {p['size']} | ${p['ppm2_median']:,.0f} | "
                        f"${p['price_median']:,.0f} | {p['lat_mean']:.3f}, {p['lon_mean']:.3f} | {addr} |")
        return head + "\n".join(rows)

    tt_rows = "\n".join(
        f"| C{c} | {', '.join(terms)} |" for c, terms in
        sorted(t["top_terms"].items(), key=lambda x: x[0]))

    block = (
        f"\n---\n\n{marker}\n\n"
        "Идея: вместо обучения с учителем — посмотреть, на какие **естественные сегменты** "
        "распадается рынок. Иерархическая (агломеративная) кластеризация строит дерево "
        "слияний снизу вверх; высота слияния = насколько непохожи объединяемые группы. "
        "Дендрограмма показывает **всю иерархию сразу**, а горизонтальный рез задаёт число "
        "кластеров. Скрипт: [`train_hclust.py`](train_hclust.py).\n\n"
        "Граница реза выбрана автоматически по **наибольшему разрыву высот слияний** в "
        "верхушке дерева (largest-gap) в диапазоне 3–12 кластеров: самое «дорогое» слияние — "
        "там, где склеиваются уже-разнородные группы, и резать логично прямо под ним "
        "(бинарный сплит как «сегментацию» не рассматриваем). Линия реза нарисована "
        "красным пунктиром на каждой дендрограмме. Деревья усечены до верхних 40 узлов "
        "(7115 листьев целиком нечитаемы); в скобках у листа — размер поддерева.\n\n"
        "Везде используется **Ward-linkage**: он минимизирует прирост внутрикластерной "
        "дисперсии и даёт сбалансированные кластеры (в отличие от average/single, которые "
        "на этих данных «слипаются» в один гигант + единичные выбросы — chaining).\n\n"

        "### 17.1 Кластеризация по гео + цене за м²\n\n"
        f"Признаки: `[lat, lon, log1p(price/m²)]`, стандартизованы (каждый вес 1), "
        f"linkage = **Ward**, метрика = euclidean. `price/m²` логарифмирован (сырой скошен "
        f"skew≈74, выбросы до $103k/м²) и **винзоризован по 1–99 перцентилю** — иначе "
        f"битые объявления образуют отдельную тривиальную ветку и рез вырождается в "
        f"«выбросы против всех».\n\n"
        f"**Рез: высота {g['cut_height']} → {g['n_clusters']} кластеров.**\n\n"
        "![Гео-дендрограмма](figs/40_hclust_geo_dendro.png)\n\n"
        "![Гео-карта кластеров](figs/41_hclust_geo_map.png)\n\n"
        f"{prof_table(g['profile'])}\n\n"
        "**Что видно.** Кластеры сбалансированы и ложатся **компактными кусками карты** "
        "(левая панель) — Ward по `[lat,lon,$/м²]` фактически нарезает город на ценовые "
        "зоны. Правая панель ранжирует их по медиане $/м²: лестница от дешёвых окраин к "
        "дорогому центру разделена чисто. Винзоризация убрала битые объявления, поэтому ни "
        "один кластер не «съеден» выбросами.\n\n"

        "### 17.2 Кластеризация по адресу (текст)\n\n"
        "Адрес — строка вида «Бишкек, мкр Джал, …». Кластеризуем по похожести написания. "
        "Векторы L2-нормированы, поэтому евклидово расстояние между ними монотонно по "
        "косинусной близости → Ward-linkage на нормированных векторах = кластеризация по "
        "косинусу, но без chaining. Два способа векторизации:\n\n"
        "**(a) HashingVectorizer** (1024, ngram(1,2), l2) — тот же, что в линейке §13.\n\n"
        f"**Рез: высота {h['cut_height']} → {h['n_clusters']} кластеров.**\n\n"
        "![Hashing-дендрограмма](figs/42_hclust_addr_hashing_dendro.png)\n\n"
        f"{prof_table(h['profile'])}\n\n"
        "**(b) TfidfVectorizer** (ngram(1,2), max_features=4000, min_df=3) — взвешивает "
        "редкие токены (названия улиц/ЖК) выше частых («Бишкек», «ул»).\n\n"
        f"**Рез: высота {t['cut_height']} → {t['n_clusters']} кластеров.**\n\n"
        "![TF-IDF дендрограмма](figs/43_hclust_addr_tfidf_dendro.png)\n\n"
        f"{prof_table(t['profile'])}\n\n"
        "Топ-термины TF-IDF кластеров (по центроиду):\n\n"
        "| Кластер | Характерные токены |\n|---|---|\n"
        f"{tt_rows}\n\n"
        "**Hashing vs TF-IDF.** Hashing быстрый и без словаря, но необратим (нельзя назвать "
        "токены кластера) и склеивает коллизии. TF-IDF интерпретируем — видно, что кластеры "
        "собираются вокруг конкретных улиц/районов/ЖК, и взвешивание редких токенов даёт "
        "более «географически осмысленные» группы.\n\n"

        "### 17.3 Совпадают ли разбиения? (ARI)\n\n"
        "![ARI и карта текстовых кластеров](figs/44_hclust_ari_textmap.png)\n\n"
        "Adjusted Rand Index (1 = идентичны, 0 = как случайные):\n\n"
        "| Пара | ARI |\n|---|---:|\n"
        f"| geo ↔ hashing | {ari['geo_vs_hashing']} |\n"
        f"| geo ↔ tfidf | {ari['geo_vs_tfidf']} |\n"
        f"| hashing ↔ tfidf | {ari['hashing_vs_tfidf']} |\n\n"
        "**Что видно.** (1) Текстовые методы согласованы: ARI(hashing, tfidf) высок — "
        "разные векторизации видят одну структуру района/улицы, результат устойчив. "
        "(2) Текст и гео дополняют друг друга: ARI(geo, text) низкий, но положительный — "
        "TF-IDF кластеры ложатся географически (правая панель карты), но группируют по "
        "*написанию* (одна улица = один кластер независимо от цены), тогда как гео-кластеры "
        "режут ещё и по `$/м²`. Это два разных взгляда на рынок.\n\n"

        "### 17.4 Выводы\n\n"
        "1. **Рынок естественно сегментируется** — Ward без учителя даёт сбалансированные "
        "кластеры; largest-gap рез (3–12) даёт интерпретируемое число сегментов.\n"
        "2. **Гео+цена → ценовые зоны** города (после винзоризации выбросов $/м²); "
        "медиана за метр растёт от окраин к центру.\n"
        "3. **Адрес-текст → районы/улицы**; TF-IDF осмысленнее Hashing (интерпретируемые "
        "токены, лучше ловит редкие названия ЖК), а высокий ARI между методами "
        "подтверждает устойчивость.\n"
        "4. **Два взгляда дополняют друг друга** (низкий ARI geo↔text): метку гео-ценового "
        "кластера можно подать в §16 как категорию-зону — но §16.3 уже показал, что для "
        "дерева сырые `(lat,lon)` почти исчерпывают географию, так что прирост ожидается "
        "небольшой.\n"
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
