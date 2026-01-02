from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

@dataclass
class DQResult:
    ok: bool
    message: str

def run_dq(curated_hourly_path: Path) -> list[DQResult]:
    df = pd.read_parquet(curated_hourly_path)

    results: list[DQResult] = []

    # Basic freshness: must have at least some rows
    results.append(DQResult(ok=len(df) > 0, message=f"row_count={len(df)}"))

    # Null checks
    for col in ["city", "time", "temperature_2m"]:
        nulls = df[col].isna().sum()
        results.append(DQResult(ok=nulls == 0, message=f"nulls_{col}={nulls}"))

    # Duplicate check on city+time
    dups = df.duplicated(subset=["city", "time"]).sum()
    results.append(DQResult(ok=dups == 0, message=f"dups_city_time={dups}"))

    return results
