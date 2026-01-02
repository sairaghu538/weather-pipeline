from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import requests

from .config import OPEN_METEO_URL, HOURLY_VARS, TIMEZONE, City
from .storage import write_json

def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def fetch_open_meteo(city: City) -> Dict[str, Any]:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def ingest_cities(cities: List[City], raw_dir: Path) -> list[Path]:
    stamp = _now_utc_stamp()
    saved: list[Path] = []

    for c in cities:
        payload = fetch_open_meteo(c)
        payload["_meta"] = {
            "city": c.name,
            "lat": c.lat,
            "lon": c.lon,
            "fetched_at_utc": stamp,
            "source": "open-meteo",
        }
        safe_name = c.name.replace(",", "").replace(" ", "_")
        out = raw_dir / f"{safe_name}_{stamp}.json"
        write_json(out, payload)
        saved.append(out)

    return saved
