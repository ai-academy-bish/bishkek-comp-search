"""Process raw Bishkek apartments train.csv into a clean numeric-ready dataset."""
import argparse
import re
from pathlib import Path

import pandas as pd

KEEP_COLUMNS = [
    "main",
    "address",
    "lat",
    "lon",
    "Тип предложения",
    "Серия",
    "Дом",
    "Этаж",
    "Площадь",
    "Состояние",
    "usd_price",
]

RENAME = {
    "address": "address",
    "lat": "lat",
    "lon": "lon",
    "Тип предложения": "offer_type",
    "Серия": "series",
    "Состояние": "condition",
    "usd_price": "usd_price",
}

UNDEFINED_ROOMS = 1000


def extract_rooms(main: str) -> int:
    if not isinstance(main, str):
        return UNDEFINED_ROOMS
    m = re.match(r"\s*(\d+)\s*-комн", main)
    if m:
        return int(m.group(1))
    if "6 и более" in main:
        return 6
    return UNDEFINED_ROOMS


def extract_floor(value: str) -> tuple[float, float]:
    if not isinstance(value, str):
        return (float("nan"), float("nan"))
    m = re.match(r"\s*(\d+)\s*этаж\s*из\s*(\d+)", value)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (float("nan"), float("nan"))


def extract_area(value: str) -> tuple[float, float]:
    if not isinstance(value, str):
        return (float("nan"), float("nan"))
    total = re.match(r"\s*([\d.]+)\s*м2", value)
    living = re.search(r"жилая:\s*([\d.]+)\s*м2", value)
    total_v = float(total.group(1)) if total else float("nan")
    living_v = float(living.group(1)) if living else float("nan")
    return (total_v, living_v)


def extract_building(value: str) -> tuple[str | float, float]:
    if not isinstance(value, str):
        return (float("nan"), float("nan"))
    parts = [p.strip() for p in value.split(",")]
    material = parts[0] if parts else float("nan")
    year = float("nan")
    for p in parts[1:]:
        m = re.search(r"(\d{4})\s*г", p)
        if m:
            year = float(m.group(1))
            break
    return (material, year)


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df[KEEP_COLUMNS].copy()

    df["rooms"] = df["main"].apply(extract_rooms)

    floors = df["Этаж"].apply(extract_floor)
    df["floor"] = [f[0] for f in floors]
    df["total_floors"] = [f[1] for f in floors]

    areas = df["Площадь"].apply(extract_area)
    df["area_total"] = [a[0] for a in areas]
    df["area_living"] = [a[1] for a in areas]

    buildings = df["Дом"].apply(extract_building)
    df["building_material"] = [b[0] for b in buildings]
    df["build_year"] = [b[1] for b in buildings]

    df = df.drop(columns=["main", "Этаж", "Площадь", "Дом"]).rename(columns=RENAME)

    ordered = [
        "address",
        "lat",
        "lon",
        "offer_type",
        "series",
        "building_material",
        "build_year",
        "floor",
        "total_floors",
        "rooms",
        "area_total",
        "area_living",
        "condition",
        "usd_price",
    ]
    return df[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="train.csv", type=Path)
    parser.add_argument("--output", default="train_processed.csv", type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    clean = process(raw)
    clean.to_csv(args.output, index=False)
    print(f"Saved {len(clean)} rows, {clean.shape[1]} cols -> {args.output}")
    print(clean.dtypes)
    print(clean.head())


if __name__ == "__main__":
    main()
