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
def c_to_f(x: Any) -> float | None:
    try:
        return (float(x) * 9 / 5) + 32
    except Exception:
        return None


def safe_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def format_city_option(x: dict[str, Any]) -> str:
    name = x.get("name") or ""
    state = x.get("state") or ""
    return f"{name}, {state}".strip().rstrip(",")


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


def pick_city_ui(
    df: pd.DataFrame,
    *,
    key: str,
    label: str,
    default_city: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    if df is None or df.empty or "city" not in df.columns:
        st.info("No city data available yet.")
        return df, None

    cities = sorted(df["city"].dropna().unique().tolist())
    if not cities:
        st.info("No city data available yet.")
        return df, None

    idx = 0
    if default_city and default_city in cities:
        idx = cities.index(default_city)

    selected = st.selectbox(label, options=cities, index=idx, key=key)
    return df[df["city"] == selected].copy(), selected


def metric_temp(label: str, value_c: Any, unit_mode: str) -> tuple[str, str]:
    if value_c is None or (isinstance(value_c, float) and pd.isna(value_c)):
        return label, "NA"

    try:
        v = float(value_c)
    except Exception:
        return label, str(value_c)

    if unit_mode == "Fahrenheit (°F)":
        vf = c_to_f(v)
        return label, f"{vf:.1f} °F"
    if unit_mode == "Both (°C + °F)":
        vf = c_to_f(v)
        return label, f"{v:.1f} °C | {vf:.1f} °F" if vf is not None else f"{v:.1f} °C"
    return label, f"{v:.1f} °C"


def display_temp_series(df: pd.DataFrame, col: str, unit_mode: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if unit_mode == "Fahrenheit (°F)":
        return (s * 9 / 5) + 32
    return s


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Weather Data Pipeline", layout="wide")

st.title("Weather Data Engineering Project")
st.caption("Open-Meteo forecast ingestion → curated parquet → dashboard")

hourly_path = CURATED_DIR / "weather_hourly.parquet"
daily_path = CURATED_DIR / "weather_daily.parquet"

if "last_city_name" not in st.session_state:
    st.session_state.last_city_name = None

# Sidebar controls
st.sidebar.header("Display")
unit_mode = st.sidebar.radio(
    "Temperature units",
    ["Celsius (°C)", "Fahrenheit (°F)", "Both (°C + °F)"],
    index=2,
    key="unit_mode",
)
show_tables = st.sidebar.checkbox("Show raw tables", value=True, key="show_tables")

# ----------------------------
# Top: Pipeline + Status
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline")

    st.markdown("### Search a US city")
    city_query = st.text_input("Type a city name (example: Springfield)", value="", key="city_query")

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

            st.caption("Selected city details")
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
        "Clear old curated parquet before run",
        value=True,
        key="clear_old",
        help="Keeps the dashboard focused on the selected city.",
    )

    if st.button("Run pipeline now", key="run_btn"):
        if selected_city is None:
            st.error("Please search and select a city first.")
        else:
            city_name = f"{selected_city['name']}, {selected_city['state']}".strip().rstrip(",")
            city_obj = City(
                name=city_name,
                lat=float(selected_city["lat"]),
                lon=float(selected_city["lon"]),
            )

            st.session_state.last_city_name = city_obj.name
            st.info(f"Running pipeline for: {city_obj.name}")

            if clear_old:
                try:
                    if hourly_path.exists():
                        hourly_path.unlink()
                    if daily_path.exists():
                        daily_path.unlink()
                except Exception as e:
                    st.warning(f"Could not delete old curated files: {e}")

            with st.spinner("Ingesting forecast data..."):
                raw_files = ingest_cities([city_obj], RAW_DIR)

            with st.spinner("Building curated parquet..."):
                curated_hourly = build_curated(raw_files, CURATED_DIR)
                curated_daily = build_daily_aggregates(curated_hourly, CURATED_DIR)

            # Keep DQ in the pipeline (optional), but do not show in UI
            try:
                with st.spinner("Running checks..."):
                    _ = run_dq(curated_hourly)
            except Exception:
                pass

            st.success("Pipeline completed")
            st.caption("Curated output")
            st.code(f"{curated_hourly}\n{curated_daily}")

with col2:
    st.subheader("Status")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.write(f"Now: {now}")

    if hourly_path.exists():
        dfh_status = pd.read_parquet(hourly_path)
        st.write("Hourly rows:", len(dfh_status))
        if "time" in dfh_status.columns:
            ts = safe_dt(dfh_status["time"])
            if len(ts) > 0:
                st.write("Latest timestamp:", str(ts.max()))
    else:
        st.info("No curated data yet. Run the pipeline.")

st.divider()

# ----------------------------
# Read data once
# ----------------------------
dfh_all: pd.DataFrame | None = None
dfd_all: pd.DataFrame | None = None

if hourly_path.exists():
    dfh_all = pd.read_parquet(hourly_path)

if daily_path.exists():
    dfd_all = pd.read_parquet(daily_path)

# ----------------------------
# Tabs
# ----------------------------
tab_daily, tab_hourly = st.tabs(["Daily forecast", "Hourly forecast"])

with tab_daily:
    if dfd_all is None or dfd_all.empty:
        st.info("Daily forecast not available yet. Run the pipeline.")
    else:
        dfd_city, selected_city_name = pick_city_ui(
            dfd_all,
            key="daily_city_select",
            label="Select city",
            default_city=st.session_state.last_city_name,
        )

        if selected_city_name:
            st.info(f"Showing data for: {selected_city_name}")

        # Sort by date
        if "date" in dfd_city.columns:
            dfd_city["date_dt"] = safe_dt(dfd_city["date"])
            dfd_city = dfd_city.sort_values("date_dt")

        # Latest day metrics
        last_row = dfd_city.tail(1)
        if len(last_row) == 1:
            r = last_row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)

            if "temp_max" in dfd_city.columns:
                lbl, val = metric_temp("Max temp (latest day)", r.get("temp_max"), unit_mode)
                m1.metric(lbl, val)

            if "temp_min" in dfd_city.columns:
                lbl, val = metric_temp("Min temp (latest day)", r.get("temp_min"), unit_mode)
                m2.metric(lbl, val)

            if "temp_avg" in dfd_city.columns:
                lbl, val = metric_temp("Avg temp (latest day)", r.get("temp_avg"), unit_mode)
                m3.metric(lbl, val)

            if "precip_sum" in dfd_city.columns:
                try:
                    p = float(r.get("precip_sum", 0.0))
                    m4.metric("Precip (latest day)", f"{p:.1f} mm")
                except Exception:
                    m4.metric("Precip (latest day)", str(r.get("precip_sum", "NA")))

        st.markdown("### 7-day outlook")

        # 7 day tiles
        tiles = dfd_city.copy()
        if "date_dt" in tiles.columns:
            tiles = tiles.sort_values("date_dt").tail(7)
        else:
            tiles = tiles.tail(7)

        cols = st.columns(7)
        for i, (_, row) in enumerate(tiles.reset_index(drop=True).iterrows()):
            with cols[i]:
                day_label = "NA"
                if "date_dt" in tiles.columns and pd.notna(row.get("date_dt")):
                    day_label = row["date_dt"].strftime("%a")

                st.markdown(f"**{day_label}**")

                if "temp_max" in tiles.columns:
                    _, v = metric_temp("High", row.get("temp_max"), unit_mode)
                    st.write(f"High: {v}")

                if "temp_min" in tiles.columns:
                    _, v = metric_temp("Low", row.get("temp_min"), unit_mode)
                    st.write(f"Low: {v}")

                if "precip_sum" in tiles.columns:
                    try:
                        st.write(f"Precip: {float(row.get('precip_sum', 0.0)):.1f} mm")
                    except Exception:
                        st.write(f"Precip: {row.get('precip_sum', 'NA')}")

        st.markdown("### Temperature trend (daily)")
        temp_cols = [c for c in ["temp_min", "temp_avg", "temp_max"] if c in dfd_city.columns]
        if "date_dt" in dfd_city.columns and temp_cols:
            chart_df = pd.DataFrame(index=dfd_city["date_dt"])
            for c in temp_cols:
                chart_df[c] = display_temp_series(dfd_city, c, unit_mode)
            st.line_chart(chart_df)

        st.markdown("### Precipitation (daily)")
        if "date_dt" in dfd_city.columns and "precip_sum" in dfd_city.columns:
            precip_df = dfd_city[["date_dt", "precip_sum"]].set_index("date_dt")
            st.bar_chart(precip_df)

        if show_tables:
            st.markdown("### Daily table")
            drop_cols = ["date_dt"]
            st.dataframe(
                dfd_city.sort_values("date_dt", ascending=False).drop(columns=drop_cols, errors="ignore"),
                use_container_width=True,
            )

with tab_hourly:
    if dfh_all is None or dfh_all.empty:
        st.info("Hourly forecast not available yet. Run the pipeline.")
    else:
        dfh_city, selected_city_name = pick_city_ui(
            dfh_all,
            key="hourly_city_select",
            label="Select city",
            default_city=st.session_state.last_city_name,
        )

        if selected_city_name:
            st.info(f"Showing data for: {selected_city_name}")

        # Prepare time
        if "time" in dfh_city.columns:
            dfh_city["time_dt"] = safe_dt(dfh_city["time"])
            dfh_city = dfh_city.sort_values("time_dt")

        # "Now" style card from latest row
        latest = dfh_city.tail(1)
        if len(latest) == 1:
            r = latest.iloc[0]
            c1, c2, c3, c4 = st.columns(4)

            if "temperature_2m" in dfh_city.columns:
                lbl, val = metric_temp("Current temp", r.get("temperature_2m"), unit_mode)
                c1.metric(lbl, val)

            if "relative_humidity_2m" in dfh_city.columns:
                try:
                    c2.metric("Humidity", f"{float(r.get('relative_humidity_2m')):.0f}%")
                except Exception:
                    c2.metric("Humidity", str(r.get("relative_humidity_2m", "NA")))

            if "windspeed_10m" in dfh_city.columns:
                try:
                    c3.metric("Wind", f"{float(r.get('windspeed_10m')):.1f} km/h")
                except Exception:
                    c3.metric("Wind", str(r.get("windspeed_10m", "NA")))

            if "precipitation" in dfh_city.columns:
                try:
                    c4.metric("Precip", f"{float(r.get('precipitation')):.2f} mm")
                except Exception:
                    c4.metric("Precip", str(r.get("precipitation", "NA")))

        st.markdown("### Hourly view")
        view = st.radio(
            "View",
            ["Temperature", "Precipitation", "Wind"],
            horizontal=True,
            key="hourly_view",
        )

        if "time_dt" not in dfh_city.columns:
            st.info("No time column available to plot hourly charts.")
        else:
            plot_df = dfh_city.set_index("time_dt")

            if view == "Temperature":
                if "temperature_2m" in plot_df.columns:
                    tmp = display_temp_series(dfh_city, "temperature_2m", unit_mode)
                    temp_plot = pd.DataFrame({"temperature_2m": tmp.values}, index=plot_df.index)
                    st.line_chart(temp_plot)
                else:
                    st.info("temperature_2m not available in hourly data.")

            elif view == "Precipitation":
                if "precipitation" in plot_df.columns:
                    st.bar_chart(plot_df[["precipitation"]])
                else:
                    st.info("precipitation not available in hourly data.")

            else:
                if "windspeed_10m" in plot_df.columns:
                    st.line_chart(plot_df[["windspeed_10m"]])
                else:
                    st.info("windspeed_10m not available in hourly data.")

        if show_tables:
            st.markdown("### Hourly table (latest 200 rows)")
            st.dataframe(
                dfh_city.sort_values("time_dt", ascending=False)
                .head(200)
                .drop(columns=["time_dt"], errors="ignore"),
                use_container_width=True,
            )
