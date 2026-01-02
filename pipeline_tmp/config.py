from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
CURATED_DIR = BASE_DIR / "data" / "curated"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CURATED_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float

CITIES = [
    City("Newark, DE", 39.6837, -75.7497),
    City("Wilmington, DE", 39.7391, -75.5398),
    City("Philadelphia, PA", 39.9526, -75.1652),
    City("New York, NY", 40.7128, -74.0060),
    City("San Jose, CA", 37.3382, -121.8863),
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloudcover",
    "windspeed_10m",
]

TIMEZONE = "UTC"
