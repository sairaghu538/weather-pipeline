from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st

from pipeline.config import CITIES, RAW_DIR, CURATED_DIR
from pipeline.ingest import ingest_cities
from pipeline.transform import build_curated, build_daily_aggregates
from pipeline.dq import run_dq


def city_filter_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a city selector and returns filtered dataframe
    """
    cities = sorted(df["city"].unique().tolist())
    selected_city = st.selectbox(
        "Filter results by city",
        options=cities,
        index=0,
    )
    st.info(f"Showing data for: **{selected_city}**")
    return df[df["city"] == selected_city]


st.set_page_config(page_title="Weather Data Pipeline", layout="wide")

st.title("Weather Data Engineering Project")
st.caption("Open-Meteo ingestion → curated parquet → quality checks → dashboard")

hourly_path = CURATED_DIR / "weather_hourly.parquet"
daily_path = CURATED_DIR / "weather_daily.parquet"


@st.cache_data(ttl=3600)
def geocode_us_cities(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Uses Open-Meteo geocoding (free, no key).
    Returns US-only matches with name/admin1/lat/lon.
    """
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

    cleaned = []
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


col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline")

    # --- Step 1 UI: search + select city ---
    st.markdown("### Search a US city")
    city_query = st.text_input("Type a city name (example: Springfield)", value="")

    matches: list[dict[str, Any]] = []
    if city_query.strip():
        try:
            matches = geocode_us_cities(city_query, limit=15)
        except Exception as e:
            st.error(f"Geocoding failed: {e}")
            matches = []

    selected_city = None
    if city_query.strip():
        if matches:
            selected_label = st.selectbox(
                "Select the exact city",
                options=[format_city_option(m) for m in matches],
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

    # st.markdown("### Default cities (current config)")
    # st.write([c.name for c in CITIES])

    # --- Run pipeline ---
    run_for_selected = st.checkbox(
        "Run pipeline for selected city (Step 2 will wire this into ingest)",
        value=True,
        disabled=(selected_city is None),
    )

    if st.button("Run pipeline now"):
        # Step 1 behavior:
        # - If selected city exists, we SHOW it and still run current pipeline (Step 2 changes ingest).
        # - This keeps app working while we build the next step safely.
        if run_for_selected and selected_city is not None:
            st.info(
                f"Selected city: {selected_city['name']}, {selected_city['state']} "
                f"({selected_city['lat']}, {selected_city['lon']})"
            )
            st.info("Step 1 validated. Step 2 will make ingest run only for this city.")

        with st.spinner("Ingesting weather data..."):
            raw_files = ingest_cities(CITIES, RAW_DIR)

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

with col2:
    st.subheader("Status")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.write(f"Now: {now}")

    if hourly_path.exists():
        dfh = pd.read_parquet(hourly_path)
        st.write("Hourly parquet rows:", len(dfh))
        st.write("Latest timestamp:", str(pd.to_datetime(dfh["time"]).max()))
    else:
        st.info("No curated data yet. Click 'Run pipeline now'.")

st.divider()

tab1, tab2 = st.tabs(["Daily aggregates", "Hourly sample"])

with tab1:
    if daily_path.exists():
        dfd = pd.read_parquet(daily_path)
        dfd = city_filter_ui(dfd)

        st.dataframe(
            dfd.sort_values("date", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("Daily aggregates not created yet.")


with tab2:
    if hourly_path.exists():
        dfh = pd.read_parquet(hourly_path)
        dfh = city_filter_ui(dfh)

        st.dataframe(
            dfh.sort_values("time", ascending=False).head(200),
            use_container_width=True,
        )
    else:
        st.info("Hourly data not created yet.")
