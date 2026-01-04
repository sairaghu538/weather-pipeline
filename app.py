from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st

from pipeline.config import RAW_DIR, CURATED_DIR, City
from pipeline.ingest import ingest_cities
from pipeline.transform import build_curated, build_daily_aggregates


# ----------------------------
# Helpers
# ----------------------------
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
    if col not in df.columns:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    return pd.to_datetime(df[col], errors="coerce")


def pick_city_from_df(
    df: pd.DataFrame,
    *,
    key: str,
    label: str,
    preferred_city: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    If df contains multiple cities, show a selectbox and return filtered df.
    If only one city exists, skip selectbox and return df as-is.
    """
    if "city" not in df.columns:
        st.info("No 'city' column found in data.")
        return df, None

    cities = sorted(df["city"].dropna().unique().tolist())
    if not cities:
        st.info("No city data available.")
        return df, None

    if len(cities) == 1:
        chosen = cities[0]
        st.info(f"Showing data for: {chosen}")
        return df[df["city"] == chosen].copy(), chosen

    default_idx = 0
    if preferred_city and preferred_city in cities:
        default_idx = cities.index(preferred_city)

    chosen = st.selectbox(label, options=cities, index=default_idx, key=key)
    st.info(f"Showing data for: {chosen}")
    return df[df["city"] == chosen].copy(), chosen


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
# Left: Search + Run (Step 1 + Step 2)
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
    selected_city_label: str | None = None

    if city_query.strip():
        if matches:
            selected_city_label = st.selectbox(
                "Select the exact city",
                options=[format_city_option(m) for m in matches],
                key="city_pick",
            )
            selected_city = next(m for m in matches if format_city_option(m) == selected_city_label)

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

    clear_old = st.checkbox(
        "Clear old curated parquet before run (recommended)",
        value=True,
        key="clear_old",
    )

    if st.button("Run pipeline now", key="run_btn"):
        if selected_city is None:
            st.error("Please search and select a city first.")
        else:
            # Step 2: run only for selected city
            city_obj = City(
                name=f"{selected_city['name']}, {selected_city['state']}".rstrip(", "),
                lat=float(selected_city["lat"]),
                lon=float(selected_city["lon"]),
            )
            cities_to_run = [city_obj]

            st.success(f"Running pipeline only for: {city_obj.name}")

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

            st.success("Pipeline completed")

            with st.expander("Pipeline output files"):
                st.code(f"{curated_hourly}\n{curated_daily}")

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
# Tabs: Daily + Hourly (Step 3 already done)
# ----------------------------
tab1, tab2 = st.tabs(["Daily forecast", "Hourly forecast"])

with tab1:
    if daily_path.exists():
        dfd_all = pd.read_parquet(daily_path).copy()

        # Prefer the just-searched city (if present in parquet)
        preferred = None
        if "city" in dfd_all.columns:
            if "city_pick" in st.session_state:
                preferred = st.session_state.get("city_pick")

        dfd, chosen_city = pick_city_from_df(
            dfd_all,
            key="daily_city_selectbox",
            label="Select city for daily view",
            preferred_city=preferred,
        )

        # ensure date sorted
        if "date" in dfd.columns:
            dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
            dfd = dfd.sort_values("date")

        # metrics from latest row
        last_row = dfd.tail(1)
        if len(last_row) == 1:
            r = last_row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            if "temp_max" in dfd.columns:
                m1.metric("Max temp (latest day)", f"{float(r.get('temp_max')):.1f}")
            if "temp_min" in dfd.columns:
                m2.metric("Min temp (latest day)", f"{float(r.get('temp_min')):.1f}")
            if "temp_avg" in dfd.columns:
                m3.metric("Avg temp (latest day)", f"{float(r.get('temp_avg')):.1f}")
            if "precip_sum" in dfd.columns:
                m4.metric("Precip (latest day)", f"{float(r.get('precip_sum')):.1f}")

        st.markdown("#### Temperature trend (daily)")
        temp_cols = [c for c in ["temp_min", "temp_avg", "temp_max"] if c in dfd.columns]
        if "date" in dfd.columns and temp_cols:
            chart_df = dfd[["date"] + temp_cols].set_index("date")
            st.line_chart(chart_df)

        st.markdown("#### Precipitation (daily)")
        if "date" in dfd.columns and "precip_sum" in dfd.columns:
            precip_df = dfd[["date", "precip_sum"]].set_index("date")
            st.bar_chart(precip_df)

        with st.expander("Daily table"):
            st.dataframe(dfd.sort_values("date", ascending=False), use_container_width=True)
    else:
        st.info("Daily aggregates not created yet. Run the pipeline.")

with tab2:
    if hourly_path.exists():
        dfh_all = pd.read_parquet(hourly_path).copy()

        preferred = None
        if "city" in dfh_all.columns:
            if "city_pick" in st.session_state:
                preferred = st.session_state.get("city_pick")

        dfh, chosen_city = pick_city_from_df(
            dfh_all,
            key="hourly_city_selectbox",
            label="Select city for hourly view",
            preferred_city=preferred,
        )

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

        with st.expander("Hourly table (latest 200 rows)"):
            st.dataframe(
                dfh.sort_values("time_dt", ascending=False)
                .head(200)
                .drop(columns=["time_dt"], errors="ignore"),
                use_container_width=True,
            )
    else:
        st.info("Hourly data not created yet. Run the pipeline.")
