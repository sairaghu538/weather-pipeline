from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .storage import read_json, write_parquet

def json_to_hourly_df(raw_json_path: Path) -> pd.DataFrame:
    payload = read_json(raw_json_path)

    meta = payload.get("_meta", {})
    city = meta.get("city", "unknown")

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError(f"No hourly.time found in {raw_json_path.name}")

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for k, v in hourly.items():
        if k == "time":
            continue
        df[k] = v

    df["city"] = city
    df["ingested_file"] = raw_json_path.name
    return df

def build_curated(raw_files: List[Path], curated_dir: Path) -> Path:
    dfs = [json_to_hourly_df(p) for p in raw_files]
    out_df = pd.concat(dfs, ignore_index=True)

    out_df["date"] = out_df["time"].dt.date.astype(str)
    out_df = out_df.sort_values(["city", "time"]).reset_index(drop=True)

    out_path = curated_dir / "weather_hourly.parquet"
    write_parquet(out_df, out_path)
    return out_path

def build_daily_aggregates(curated_hourly_path: Path, curated_dir: Path) -> Path:
    df = pd.read_parquet(curated_hourly_path)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date.astype(str)

    agg = (
        df.groupby(["city", "date"], as_index=False)
        .agg(
            temp_avg=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            precip_sum=("precipitation", "sum"),
            wind_avg=("windspeed_10m", "mean"),
        )
    )

    out_path = curated_dir / "weather_daily.parquet"
    agg.to_parquet(out_path, index=False)
    return out_path
