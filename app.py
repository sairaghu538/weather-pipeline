from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st

from pipeline.config import RAW_DIR, CURATED_DIR, City
from pipeline.ingest import ingest_cities
from pipeline.transform import build_curated, build_daily_aggregates
from pipeline.dq import run_dq


# ----------------------------
# Helpers
# ----------------------------
def city_filter_ui(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    default_city: str | None = None,
    label: str = "Filter results by city",
) -> pd.DataFrame:
    if "city" not in df.columns:
        st.info("No 'city' column found in data.")
        return df

    cities = sorted(df["city"].dropna().unique().tolist())
    if not cities:
        st.info("No city data available.")
        return df

    default_idx = 0
    if default_city and default_city in cities:
        default_idx = cities.index(default_city)

    selected_city = st.selectbox(
        label,
        options=cities,
        index=default_idx,
        key=f"{key_prefix}_city_select",
    )

    st.info(f"Showing data for: {selected_city}")
    return df[df["city"] == selected_city].copy()


@st.cache_data(ttl=3600)
def geocode_us_cities(query: str, limit: int = 10) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": limit, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    results = data.get("results") or []
    us_results = [x for x in results if x.get("country_code") == "US"]

    cleaned: list[dict[str, Any]] = []
    for x in us_results:
        cleaned.append(
            {
                "name": x.get("name"),
                "state": x.get("admin1") or "",
                "lat": x.get("latitude"),
                "lon": x.get("longitude"),
            }
        )
    return cleaned


def format_city_option(x: dict[str, Any]) -> str:
    name = x.get("name") or ""
    state = x.get("state") or ""
    return f"{name}, {state}".strip().rstrip(",")


def safe_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    # converts to datetime without crashing UI
    if col not in df.columns:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    return pd.to_datetime(df[col], errors="coerce")


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Weather Data Pipeline", layout="wide")

st.title("Weather Data Engineering Project")
st.caption("Open-Meteo ingestion → curated parquet → quality checks → dashboard")

hourly_path = CURATED_DIR / "weather_hourly.parquet"
daily_path = CURATED_DIR / "weather_daily.parquet"

col1, col2 = st.columns(2)

# ----------------------------
# Left: Search + Run
# ----------------------------
with col1:
    st.subheader("Pipeline")

    st.markdown("### Search a US city")
    city_query = st.text_input("Type a city name (example: Springfield)", value="")

    matches: list[dict[str, Any]] = []
    if city_query.strip():
        try:
            matches = geocode_us_cities(city_query, limit=15)
        except Exception as e:
            st.error(f"Geocoding failed: {e}")
            matches = []

    selected_city: dict[str, Any] | None = None
    if city_query.strip():
        if matches:
            selected_label = st.selectbox(
                "Select the exact city",
                options=[format_city_option(m) for m in matches],
                key="city_pick",
            )
            selected_city = next(m for m in matches if format_city_option(m) == selected_label)

            st.caption("Selected city details (for verification)")
            st.write(
                {
                    "city": selected_city["name"],
                    "state": selected_city["state"],
                    "lat": selected_city["lat"],
                    "lon": selected_city["lon"],
                }
            )
        else:
            st.info("No US matches found. Try a different spelling.")

    st.divider()

    run_for_selected = st.checkbox(
        "Run pipeline for selected city",
        value=True,
        disabled=(selected_city is None),
        key="run_for_selected",
    )

    clear_old = st.checkbox(
        "Clear old curated parquet before run (recommended)",
        value=True,
        key="clear_old",
    )

    if st.button("Run pipeline now", key="run_btn"):
        if selected_city is None:
            st.error("Please search and select a city first.")
        else:
            # Step 2: Build a single-city list for ingest
            city_obj = City(
                name=f"{selected_city['name']}, {selected_city['state']}".rstrip(", "),
                lat=float(selected_city["lat"]),
                lon=float(selected_city["lon"]),
            )
            cities_to_run = [city_obj]

            st.info(f"Running pipeline only for: {city_obj.name} ({city_obj.lat}, {city_obj.lon})")

            if clear_old:
                try:
                    if hourly_path.exists():
                        hourly_path.unlink()
                    if daily_path.exists():
                        daily_path.unlink()
                except Exception as e:
                    st.warning(f"Could not delete old curated files: {e}")

            with st.spinner("Ingesting weather data..."):
                raw_files = ingest_cities(cities_to_run, RAW_DIR)

            with st.spinner("Transforming to curated parquet..."):
                curated_hourly = build_curated(raw_files, CURATED_DIR)
                curated_daily = build_daily_aggregates(curated_hourly, CURATED_DIR)

            with st.spinner("Running data quality checks..."):
                dq_results = run_dq(curated_hourly)

            st.success("Pipeline completed")
            st.write("Curated files:")
            st.code(f"{curated_hourly}\n{curated_daily}")

            st.write("DQ Results")
            for r in dq_results:
                (st.success if r.ok else st.error)(r.message)

# ----------------------------
# Right: Status
# ----------------------------
with col2:
    st.subheader("Status")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.write(f"Now: {now}")

    if hourly_path.exists():
        dfh_status = pd.read_parquet(hourly_path)
        st.write("Hourly parquet rows:", len(dfh_status))
        ts = safe_datetime_series(dfh_status, "time")
        if len(ts) > 0:
            st.write("Latest timestamp:", str(ts.max()))
    else:
        st.info("No curated data yet. Run the pipeline.")

st.divider()

# ----------------------------
# Tabs: Daily + Hourly
# ----------------------------
tab1, tab2 = st.tabs(["Daily forecast", "Hourly forecast"])

with tab1:
    if daily_path.exists():
        dfd = pd.read_parquet(daily_path)

        # Filter UI (in case you later store multiple cities)
        dfd = city_filter_ui(dfd, key_prefix="daily", label="Select city for daily view")

        # ---- Step 3 UI: metrics + charts ----
        # Ensure date sorted
        if "date" in dfd.columns:
            dfd = dfd.sort_values("date")

        # Pick the next available day (last row after sort)
        last_row = dfd.tail(1)
        if len(last_row) == 1:
            r = last_row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            if "temp_max" in dfd.columns:
                m1.metric("Max temp (latest day)", f"{r.get('temp_max', '')}")
            if "temp_min" in dfd.columns:
                m2.metric("Min temp (latest day)", f"{r.get('temp_min', '')}")
            if "temp_avg" in dfd.columns:
                m3.metric("Avg temp (latest day)", f"{r.get('temp_avg', '')}")
            if "precip_sum" in dfd.columns:
                m4.metric("Precip (latest day)", f"{r.get('precip_sum', '')}")

        # Charts
        st.markdown("#### Temperature trend (daily)")
        temp_cols = [c for c in ["temp_min", "temp_avg", "temp_max"] if c in dfd.columns]
        if "date" in dfd.columns and temp_cols:
            chart_df = dfd[["date"] + temp_cols].set_index("date")
            st.line_chart(chart_df)

        st.markdown("#### Precipitation (daily)")
        if "date" in dfd.columns and "precip_sum" in dfd.columns:
            precip_df = dfd[["date", "precip_sum"]].set_index("date")
            st.bar_chart(precip_df)

        st.markdown("#### Daily table")
        st.dataframe(dfd.sort_values("date", ascending=False), use_container_width=True)
    else:
        st.info("Daily aggregates not created yet. Run the pipeline.")

with tab2:
    if hourly_path.exists():
        dfh = pd.read_parquet(hourly_path)

        # Filter UI (in case you later store multiple cities)
        dfh = city_filter_ui(dfh, key_prefix="hourly", label="Select city for hourly view")

        # ---- Step 3 UI: charts ----
        dfh["time_dt"] = safe_datetime_series(dfh, "time")
        dfh = dfh.sort_values("time_dt")

        st.markdown("#### Temperature (hourly)")
        if "temperature_2m" in dfh.columns and "time_dt" in dfh.columns:
            temp_h = dfh[["time_dt", "temperature_2m"]].set_index("time_dt")
            st.line_chart(temp_h)

        st.markdown("#### Wind (hourly)")
        if "windspeed_10m" in dfh.columns and "time_dt" in dfh.columns:
            wind_h = dfh[["time_dt", "windspeed_10m"]].set_index("time_dt")
            st.line_chart(wind_h)

        st.markdown("#### Precipitation (hourly)")
        if "precipitation" in dfh.columns and "time_dt" in dfh.columns:
            pr_h = dfh[["time_dt", "precipitation"]].set_index("time_dt")
            st.bar_chart(pr_h)

        st.markdown("#### Hourly table (latest 200 rows)")
        # Keep it readable: latest first
        st.dataframe(
            dfh.sort_values("time_dt", ascending=False).head(200).drop(columns=["time_dt"], errors="ignore"),
            use_container_width=True,
        )
    else:
        st.info("Hourly data not created yet. Run the pipeline.")
