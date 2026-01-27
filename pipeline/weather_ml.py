from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

try:
    import joblib
except Exception:
    joblib = None


# -----------------------------
# Paths (project root)
# -----------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
CACHE_DIR = Path(".cache_noaa")

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# -----------------------------
# NOAA GHCN-D public URLs
# -----------------------------
NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
STATIONS_URL = f"{NOAA_BASE}/ghcnd-stations.txt"
DLY_URL_TEMPLATE = f"{NOAA_BASE}/all/{{station_id}}.dly"


# -----------------------------
# Geocoding (free, no key)
# -----------------------------
_geolocator = Nominatim(user_agent="weather-noaa-ml-app")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)


def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


@dataclass
class ForecastResult:
    city: str
    days_used: int
    station_id: str
    station_name: str
    predicted_avg_temp_c: float
    predicted_avg_temp_f: float
    test_mae_c: Optional[float]
    rows_total: int
    rows_train: int
    rows_test: int
    parquet_path: str

    # NEW: for alignment with Open-Meteo reference day
    noaa_last_date: Optional[str] = None
    target_date: Optional[str] = None
    
    # NEW: 7-Day Forecast
    dates_7d: Optional[List[str]] = None
    predicted_temp_c_7d: Optional[List[float]] = None
    predicted_temp_f_7d: Optional[List[float]] = None


# -----------------------------
# Helpers: download + cache
# -----------------------------
def _http_get_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def get_city_latlon(city: str) -> tuple[float, float, str]:
    loc = _geocode(city)
    if loc is None:
        raise ValueError(f"Could not geocode city: {city}")
    return float(loc.latitude), float(loc.longitude), str(loc.address)


def _load_stations_df() -> pd.DataFrame:
    """
    Loads NOAA station metadata from ghcnd-stations.txt.
    Caches it to .cache_noaa/ghcnd-stations.txt and .cache_noaa/stations.parquet.
    Format reference: fixed-width columns from NOAA GHCN-D docs.
    """
    stations_txt = CACHE_DIR / "ghcnd-stations.txt"
    stations_pq = CACHE_DIR / "stations.parquet"

    if stations_pq.exists():
        return pd.read_parquet(stations_pq)

    if not stations_txt.exists():
        text = _http_get_text(STATIONS_URL)
        stations_txt.write_text(text, encoding="utf-8")

    # Fixed-width parsing
    # ID            1-11
    # LATITUDE     13-20
    # LONGITUDE    22-30
    # ELEVATION    32-37
    # STATE        39-40
    # NAME         42-71
    # GSN FLAG     73-75
    # HCN/CRN FLAG 77-79
    # WMO ID       81-85
    rows: List[Dict[str, object]] = []
    with stations_txt.open("r", encoding="utf-8") as f:
        for line in f:
            sid = line[0:11].strip()
            lat = line[12:20].strip()
            lon = line[21:30].strip()
            elev = line[31:37].strip()
            state = line[38:40].strip()
            name = line[41:71].strip()

            if not sid:
                continue

            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                continue

            elev_f = None
            try:
                elev_f = float(elev)
            except Exception:
                elev_f = None

            rows.append(
                {
                    "station_id": sid,
                    "lat": lat_f,
                    "lon": lon_f,
                    "elev": elev_f,
                    "state": state,
                    "name": name,
                }
            )

    df = pd.DataFrame(rows)

    # Keep US stations only (station id starts with "US")
    df = df[df["station_id"].str.startswith("US")].copy()

    # Cache parquet
    df.to_parquet(stations_pq, index=False)
    return df


def _haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Vectorized haversine distance in KM.
    """
    import numpy as np

    r = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2.astype(float))
    lon2r = np.radians(lon2.astype(float))

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def pick_station_for_city(city: str, top_k: int = 25) -> Tuple[str, str]:
    """
    Pick nearest GHCN-D US station to the city's lat/lon.
    """
    lat, lon, _addr = get_city_latlon(city)
    stations = _load_stations_df()

    stations = stations.copy()
    stations["dist_km"] = _haversine_km(lat, lon, stations["lat"], stations["lon"])
    stations = stations.sort_values("dist_km").head(top_k)

    # return best candidate; we will still validate data availability later
    row = stations.iloc[0]
    return str(row["station_id"]), str(row["name"])


def _download_station_dly(station_id: str) -> str:
    """
    Download station .dly file and cache in .cache_noaa/dly/<station_id>.dly
    """
    dly_dir = CACHE_DIR / "dly"
    dly_dir.mkdir(exist_ok=True)

    dly_path = dly_dir / f"{station_id}.dly"
    if dly_path.exists() and dly_path.stat().st_size > 0:
        return dly_path.read_text(encoding="utf-8", errors="ignore")

    url = DLY_URL_TEMPLATE.format(station_id=station_id)
    text = _http_get_text(url, timeout=60)
    dly_path.write_text(text, encoding="utf-8")
    return text


def _parse_dly(text: str, start: date, end: date) -> pd.DataFrame:
    """
    Parse NOAA .dly fixed-width data for TMIN, TMAX, PRCP into daily rows.

    Units:
      - TMIN/TMAX are tenths of °C
      - PRCP is tenths of mm
    Missing value is -9999
    """
    records: Dict[date, Dict[str, float]] = {}

    def add_value(d: date, element: str, value: int):
        if value == -9999:
            return
        if element in ("TMIN", "TMAX"):
            v = value / 10.0
        elif element == "PRCP":
            v = value / 10.0
        else:
            return

        if d not in records:
            records[d] = {}
        records[d][element] = v

    for line in text.splitlines():
        if len(line) < 269:
            continue

        year = int(line[11:15])
        month = int(line[15:17])
        element = line[17:21]

        if element not in ("TMIN", "TMAX", "PRCP"):
            continue

        # 31 days blocks; each is 8 chars: value(5) + mflag + qflag + sflag
        for day in range(1, 32):
            pos = 21 + (day - 1) * 8
            val_str = line[pos : pos + 5].strip()
            if not val_str:
                continue
            try:
                val = int(val_str)
            except Exception:
                continue

            try:
                d = date(year, month, day)
            except ValueError:
                continue

            if d < start or d > end:
                continue

            add_value(d, element, val)

    if not records:
        return pd.DataFrame(columns=["date", "tmin_c", "tmax_c", "prcp_mm"])

    rows = []
    for d, vals in records.items():
        rows.append(
            {
                "date": d,
                "tmin_c": vals.get("TMIN", None),
                "tmax_c": vals.get("TMAX", None),
                "prcp_mm": vals.get("PRCP", None),
            }
        )

    df = pd.DataFrame(rows).sort_values("date")
    return df


def fetch_daily_history(city: str, days: int) -> Tuple[pd.DataFrame, str, str]:
    """
    Fetch daily history from NOAA GHCN-D:
      - pick nearest station
      - download station .dly
      - parse range
      - build avg_temp_c = (tmin + tmax)/2 when available
    """
    end = datetime.now(timezone.utc).date()
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date()

    # Try multiple nearby stations if the first has no usable range data
    lat, lon, _addr = get_city_latlon(city)
    stations = _load_stations_df().copy()
    stations["dist_km"] = _haversine_km(lat, lon, stations["lat"], stations["lon"])
    stations = stations.sort_values("dist_km").head(50)

    tried: List[str] = []

    for _, srow in stations.iterrows():
        station_id = str(srow["station_id"])
        station_name = str(srow["name"])
        tried.append(f"{station_id}:{station_name}")

        try:
            dly_text = _download_station_dly(station_id)
            df_raw = _parse_dly(dly_text, start=start, end=end)
            if df_raw is None or df_raw.empty:
                continue

            # Compute avg temp when both present
            df_raw["avg_temp_c"] = None
            mask = df_raw["tmin_c"].notna() & df_raw["tmax_c"].notna()
            df_raw.loc[mask, "avg_temp_c"] = (df_raw.loc[mask, "tmin_c"] + df_raw.loc[mask, "tmax_c"]) / 2.0

            # Rename to match schema
            df = pd.DataFrame(
                {
                    "date": df_raw["date"],
                    "avg_temp_c": df_raw["avg_temp_c"],
                    "min_temp_c": df_raw["tmin_c"],
                    "max_temp_c": df_raw["tmax_c"],
                    "precip_mm": df_raw["prcp_mm"],
                }
            )

            # Wind is not in GHCN-D daily in a consistent way, so set as NA
            df["wind_mps"] = pd.NA

            # Drop rows where we have no temp at all
            df = df.dropna(subset=["min_temp_c", "max_temp_c"], how="all")
            df = df.sort_values("date")

            if df.empty:
                continue

            return df, station_id, station_name
        except Exception:
            continue

    raise ValueError(
        f"Could not fetch usable NOAA daily data for city='{city}' in last {days} days. "
        f"Tried: {tried[:10]}{'...' if len(tried) > 10 else ''}"
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag features and next-day target.
    """
    df = df.copy()

    # Ensure avg_temp_c exists: if missing, derive from min/max when possible
    if "avg_temp_c" not in df.columns or df["avg_temp_c"].isna().all():
        mask = df["min_temp_c"].notna() & df["max_temp_c"].notna()
        df.loc[mask, "avg_temp_c"] = (df.loc[mask, "min_temp_c"] + df.loc[mask, "max_temp_c"]) / 2.0

    df["avg_temp_c_lag1"] = df["avg_temp_c"].shift(1)
    df["avg_temp_c_lag2"] = df["avg_temp_c"].shift(2)
    df["avg_temp_c_lag3"] = df["avg_temp_c"].shift(3)

    df["target_avg_temp_c"] = df["avg_temp_c"].shift(-1)

    needed = [
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "precip_mm",
        "avg_temp_c_lag1",
        "avg_temp_c_lag2",
        "avg_temp_c_lag3",
        "target_avg_temp_c",
    ]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    return df


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    cut = max(1, int(len(df) * train_ratio))
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    return train_df, test_df


def train_and_evaluate(df_feat: pd.DataFrame) -> Tuple[LinearRegression, Optional[float], int, int]:
    """
    Train Linear Regression and compute MAE on the last chunk (time-aware split).
    """
    feature_cols = [
        "avg_temp_c_lag1",
        "avg_temp_c_lag2",
        "avg_temp_c_lag3",
        "min_temp_c",
        "max_temp_c",
        "precip_mm",
        # wind_mps excluded because NOAA GHCN-D daily does not provide it consistently
    ]
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    train_df, test_df = time_series_split(df_feat, train_ratio=0.8)

    X_train = train_df[feature_cols]
    y_train = train_df["target_avg_temp_c"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    mae = None
    if len(test_df) >= 2:
        X_test = test_df[feature_cols]
        y_test = test_df["target_avg_temp_c"]
        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))

    return model, mae, len(train_df), len(test_df)


def predict_next_day_avg_temp(model: LinearRegression, df_feat: pd.DataFrame) -> float:
    """
    Predict next day avg temp using the most recent feature row.
    """
    feature_cols = [
        "avg_temp_c_lag1",
        "avg_temp_c_lag2",
        "avg_temp_c_lag3",
        "min_temp_c",
        "max_temp_c",
        "precip_mm",
    ]
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    latest = df_feat.sort_values("date").iloc[-1]
    X_latest = pd.DataFrame([{c: float(latest[c]) for c in feature_cols}])
    pred_c = float(model.predict(X_latest)[0])
    return pred_c


def predict_next_7_days(model: LinearRegression, df_feat: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Predict next 7 days using recursive regression.
    Predict t+1, assume it's true, use it to predict t+2, etc.
    """
    feature_cols = [
        "avg_temp_c_lag1",
        "avg_temp_c_lag2",
        "avg_temp_c_lag3",
        "min_temp_c",
        "max_temp_c",
        "precip_mm",
    ]
    # Filter only available columns
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    # Get latest known data
    last_row = df_feat.sort_values("date").iloc[-1].copy()
    last_date = pd.to_datetime(last_row["date"])
    
    predictions_c = []
    future_dates = []

    # Recursive loop for 7 days
    current_feats = {c: float(last_row[c]) for c in feature_cols}
    
    # We need to maintain the lag state manually
    # lags: [lag1, lag2, lag3] -> [pred, lag1, lag2]
    lag1 = current_feats.get("avg_temp_c_lag1", current_feats.get("min_temp_c", 0)) # fallback
    lag2 = current_feats.get("avg_temp_c_lag2", lag1)
    lag3 = current_feats.get("avg_temp_c_lag3", lag2)

    for i in range(1, 8):
        # 1. Build input vector
        # Note: In a real recursive model, we'd predict min/max too or use a vector model.
        # Here we simplify: assume min/max/precip stay roughly similar (persistence) or 
        # use season avg. For short term (7d), persistence of exogenous vars + lag updates is a common baseline.
        
        # specific assumption: lags update, others stay constant (persistence forecast for exogenous)
        input_row = pd.DataFrame([current_feats])
        
        # Update lags in input_row based on previous step
        if "avg_temp_c_lag1" in feature_cols: input_row["avg_temp_c_lag1"] = lag1
        if "avg_temp_c_lag2" in feature_cols: input_row["avg_temp_c_lag2"] = lag2
        if "avg_temp_c_lag3" in feature_cols: input_row["avg_temp_c_lag3"] = lag3

        # Predict Next Temp
        pred_c = float(model.predict(input_row)[0])
        predictions_c.append(pred_c)
        
        next_dt = (last_date + pd.Timedelta(days=i)).date()
        future_dates.append(str(next_dt))

        # Update state for next iteration
        lag3 = lag2
        lag2 = lag1
        lag1 = pred_c # The prediction becomes lag1 for the next day

    return future_dates, predictions_c


def run_city_forecast(city: str, days: int = 365, save_model: bool = True) -> ForecastResult:
    """
    Orchestrates grabbing data and running the forecast.
    Attempts multiple windows (days, 365, 180, 90) to ensure we find enough data.
    """
    # Robust History Strategy: Try 365 -> 180 -> 90 -> 'days'
    windows_to_try = sorted(list(set([365, 180, 90, days])), reverse=True)
    
    raw_df = pd.DataFrame()
    station_id = ""
    station_name = ""
    start_days_used = days

    for w in windows_to_try:
        try:
            # print(f"Trying window: {w} days...")
            raw_df, station_id, station_name = fetch_daily_history(city, w)
            df_check = build_features(raw_df)
            if len(df_check) >= 14:
                start_days_used = w
                break
        except Exception:
            continue
            
    if raw_df.empty:
         raise ValueError(
            f"Could not fetch enough history (need 14+ rows) for city '{city}' even after trying windows {windows_to_try}."
        )

    # Calculate Last Date / Target Date
    noaa_last_date: Optional[str] = None
    target_date: Optional[str] = None
    if not raw_df.empty and "date" in raw_df.columns:
        try:
            noaa_last_dt = pd.to_datetime(raw_df["date"]).max().date()
            target_dt = (pd.to_datetime(noaa_last_dt) + pd.Timedelta(days=1)).date()
            noaa_last_date = str(noaa_last_dt)
            target_date = str(target_dt)
        except Exception:
            noaa_last_date = None
            target_date = None

    df_feat = build_features(raw_df)

    if len(df_feat) < 10:
        raise ValueError(
            f"Not enough usable rows after feature building: {len(df_feat)}. History fetch likely failed."
        )

    parquet_path = DATA_DIR / f"{city.lower().replace(' ', '_')}_{start_days_used}d_daily.parquet"
    raw_df.to_parquet(parquet_path, index=False)

    model, mae_c, rows_train, rows_test = train_and_evaluate(df_feat)
    
    # 1-Day Prediction (Legacy)
    pred_c_1d = predict_next_day_avg_temp(model, df_feat)
    pred_f_1d = c_to_f(pred_c_1d)

    # 7-Day Prediction (New)
    dates_7d, preds_c_7d = predict_next_7_days(model, df_feat)
    preds_f_7d = [c_to_f(c) for c in preds_c_7d]

    if save_model and joblib is not None:
        model_path = MODEL_DIR / f"{city.lower().replace(' ', '_')}_{start_days_used}d_linreg.joblib"
        joblib.dump(model, model_path)

    return ForecastResult(
        city=city,
        days_used=start_days_used,
        station_id=station_id,
        station_name=station_name,
        predicted_avg_temp_c=pred_c_1d,
        predicted_avg_temp_f=pred_f_1d,
        test_mae_c=mae_c,
        rows_total=len(df_feat),
        rows_train=rows_train,
        rows_test=rows_test,
        parquet_path=str(parquet_path),
        noaa_last_date=noaa_last_date,
        target_date=target_date,
        dates_7d=dates_7d,
        predicted_temp_c_7d=preds_c_7d,
        predicted_temp_f_7d=preds_f_7d
    )


if __name__ == "__main__":
    # Quick manual test
    try:
        result = run_city_forecast("San Jose, CA", days=30)
    except ValueError:
        result = run_city_forecast("San Jose, CA", days=90)
    print(result)
