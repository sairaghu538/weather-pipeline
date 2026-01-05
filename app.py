from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st
import altair as alt

from pipeline.config import RAW_DIR, CURATED_DIR, City
from pipeline.ingest import ingest_cities
from pipeline.transform import build_curated, build_daily_aggregates

# NEW: NOAA ML forecast
from pipeline.weather_ml import run_city_forecast


# ----------------------------
# Temp helpers (unit toggle)
# ----------------------------
def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


def format_temp(celsius: float | None, unit: str) -> str:
    if celsius is None or pd.isna(celsius):
        return "—"
    c = float(celsius)
    f = c_to_f(c)

    if unit == "°C":
        return f"{c:.1f} °C"
    if unit == "°F":
        return f"{f:.1f} °F"
    return f"{c:.1f} °C / {f:.1f} °F"


def add_temp_cols(df: pd.DataFrame, cols: list[str], unit: str) -> pd.DataFrame:
    """
    Adds converted columns for charting:
    - if unit == °C -> keep original cols
    - if unit == °F -> add *_u columns in F
    - if unit == °C + °F -> keep original and add *_f columns in F
    """
    out = df.copy()
    if unit == "°C":
        for c in cols:
            if c in out.columns:
                out[f"{c}_u"] = out[c]
        return out

    if unit == "°F":
        for c in cols:
            if c in out.columns:
                out[f"{c}_u"] = out[c].apply(lambda x: c_to_f(float(x)) if pd.notna(x) else None)
        return out

    # °C + °F
    for c in cols:
        if c in out.columns:
            out[f"{c}_u"] = out[c]  # still plot Celsius by default
            out[f"{c}_f"] = out[c].apply(lambda x: c_to_f(float(x)) if pd.notna(x) else None)
    return out


# ----------------------------
# Generic helpers
# ----------------------------
def safe_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    return pd.to_datetime(df[col], errors="coerce")


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


def city_filter_ui(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    default_city: str | None = None,
    label: str = "Select city",
) -> tuple[pd.DataFrame, str | None]:
    if "city" not in df.columns:
        st.info("No 'city' column found in data.")
        return df, None

    cities = sorted(df["city"].dropna().unique().tolist())
    if not cities:
        st.info("No city data available.")
        return df, None

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
    return df[df["city"] == selected_city].copy(), selected_city


def normalize_city_for_noaa(selected_city_label: str) -> str:
    """
    NOAA path uses geopy Nominatim. Give it an unambiguous string.
    If UI gives "San Jose, California", NOAA geocoder usually accepts it,
    but adding USA helps.
    """
    s = (selected_city_label or "").strip()
    if not s:
        return s
    if "USA" in s.upper() or "UNITED STATES" in s.upper():
        return s
    return f"{s}, USA"


# ----------------------------
# NOAA ML cache wrapper
# ----------------------------
@st.cache_data(ttl=24 * 60 * 60)
def cached_noaa_forecast(city_query: str, days: int) -> dict[str, Any]:
    """
    Returns a serializable dict (Streamlit cache friendly).
    """
    res = run_city_forecast(city_query, days=days, save_model=False)
    return {
        "city": res.city,
        "days_used": res.days_used,
        "station_id": res.station_id,
        "station_name": res.station_name,
        "predicted_avg_temp_c": float(res.predicted_avg_temp_c),
        "predicted_avg_temp_f": float(res.predicted_avg_temp_f),
        "test_mae_c": None if res.test_mae_c is None else float(res.test_mae_c),
        "rows_total": int(res.rows_total),
        "rows_train": int(res.rows_train),
        "rows_test": int(res.rows_test),
        "parquet_path": str(res.parquet_path),
    }


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Weather Data Pipeline", layout="wide")

# Sidebar: unit toggle (keep UI clean)
st.sidebar.markdown("### Temperature unit")
temp_unit = st.sidebar.radio(
    "Display temperature as",
    options=["°C", "°F", "°C + °F"],
    index=2,  # default = both
)

# NEW: sidebar ML settings
st.sidebar.markdown("### ML forecast (NOAA)")
ml_days = st.sidebar.selectbox("Training window", options=[30, 90], index=0)
ml_auto_run = st.sidebar.checkbox("Auto-run ML when city changes", value=True)

st.title("Weather Data Engineering Project")
st.caption("Open-Meteo ingestion → curated parquet → dashboard")

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

    clear_old = st.checkbox(
        "Clear old curated parquet before run (recommended)",
        value=True,
        key="clear_old",
    )

    if st.button("Run pipeline now", key="run_btn"):
        if selected_city is None:
            st.error("Please search and select a city first.")
        else:
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

            st.success("Pipeline completed")
            st.write("Curated files:")
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
# Tabs: Daily + Hourly
# ----------------------------
tab1, tab2 = st.tabs(["Daily forecast", "Hourly forecast"])

with tab1:
    if daily_path.exists():
        dfd = pd.read_parquet(daily_path)
        dfd, selected_name = city_filter_ui(dfd, key_prefix="daily", label="Select city for daily view")

        if "date" in dfd.columns:
            dfd = dfd.sort_values("date")

        # Metrics (with units)
        last_row = dfd.tail(1)
        if len(last_row) == 1:
            r = last_row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max temp (latest day)", format_temp(r.get("temp_max"), temp_unit))
            m2.metric("Min temp (latest day)", format_temp(r.get("temp_min"), temp_unit))
            m3.metric("Avg temp (latest day)", format_temp(r.get("temp_avg"), temp_unit))
            m4.metric("Precip (latest day)", f"{float(r.get('precip_sum', 0.0)):.1f}")

        # NEW: ML forecast card (NOAA)
        st.markdown("### ML forecast (next day avg temp, NOAA)")
        if selected_name:
            noaa_city_query = normalize_city_for_noaa(selected_name)

            run_ml = False
            if ml_auto_run:
                run_ml = True
            else:
                run_ml = st.button("Run ML forecast now", key="ml_run_btn")

            if run_ml:
                try:
                    with st.spinner("Training model and predicting using NOAA daily data..."):
                        ml_out = cached_noaa_forecast(noaa_city_query, ml_days)

                    c_pred = ml_out["predicted_avg_temp_c"]
                    f_pred = ml_out["predicted_avg_temp_f"]

                    ml1, ml2, ml3, ml4 = st.columns(4)
                    ml1.metric("Predicted avg temp", format_temp(c_pred, temp_unit))
                    ml2.metric("MAE (°C)", "—" if ml_out["test_mae_c"] is None else f"{ml_out['test_mae_c']:.2f}")
                    ml3.metric("Rows used", f"{ml_out['rows_total']} (train {ml_out['rows_train']}, test {ml_out['rows_test']})")
                    ml4.metric("NOAA station", ml_out["station_name"])

                    with st.expander("Details"):
                        st.write(
                            {
                                "city_query": noaa_city_query,
                                "days_used": ml_out["days_used"],
                                "station_id": ml_out["station_id"],
                                "station_name": ml_out["station_name"],
                                "predicted_avg_temp_c": c_pred,
                                "predicted_avg_temp_f": f_pred,
                                "parquet_path": ml_out["parquet_path"],
                            }
                        )
                except Exception as e:
                    st.error(f"ML forecast failed: {e}")
                    st.info("Tip: try 90 days in the sidebar, or use a more specific city like 'San Jose, CA'.")
        else:
            st.info("Pick a city to run the ML forecast.")

        # Prepare temp columns (for charts)
        temp_cols = [c for c in ["temp_min", "temp_avg", "temp_max"] if c in dfd.columns]
        dfd = add_temp_cols(dfd, temp_cols, temp_unit)

        # ---- 1) Temperature trend: line + area fill ----
        st.markdown("#### Temperature trend (daily)")
        if "date" in dfd.columns and temp_cols:
            plot_cols = {c: f"{c}_u" for c in temp_cols if f"{c}_u" in dfd.columns}

            base = alt.Chart(dfd).encode(
                x=alt.X("date:T", title="Date")
            )

            area_field = plot_cols.get("temp_avg") or plot_cols.get("temp_max")
            y_title = "Temperature (°C)" if temp_unit == "°C" else "Temperature (°F)" if temp_unit == "°F" else "Temperature (°C)"

            area = base.mark_area(opacity=0.25).encode(
                y=alt.Y(f"{area_field}:Q", title=y_title)
            )

            lines = []
            for label, col in plot_cols.items():
                lines.append(
                    base.mark_line().encode(
                        y=alt.Y(f"{col}:Q"),
                        tooltip=["date:T", alt.Tooltip(f"{col}:Q", title=label)],
                    )
                )

            chart = area
            for ln in lines:
                chart = chart + ln

            st.altair_chart(chart.interactive(), use_container_width=True)
        else:
            st.info("Temperature fields not found in daily data.")

        # ---- 2) Precipitation: line chart ----
        st.markdown("#### Precipitation (daily)")
        if "date" in dfd.columns and "precip_sum" in dfd.columns:
            precip_line = (
                alt.Chart(dfd)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("precip_sum:Q", title="Precipitation"),
                    tooltip=["date:T", alt.Tooltip("precip_sum:Q", title="precip_sum")],
                )
            )
            st.altair_chart(precip_line.interactive(), use_container_width=True)
        else:
            st.info("Precipitation field not found in daily data.")

        st.markdown("#### Daily table")
        st.dataframe(dfd.sort_values("date", ascending=False), use_container_width=True)

    else:
        st.info("Daily aggregates not created yet. Run the pipeline.")

with tab2:
    if hourly_path.exists():
        dfh = pd.read_parquet(hourly_path)
        dfh, selected_name = city_filter_ui(dfh, key_prefix="hourly", label="Select city for hourly view")

        dfh["time_dt"] = safe_datetime_series(dfh, "time")
        dfh = dfh.sort_values("time_dt")

        # Convert hourly temperature if needed
        if "temperature_2m" in dfh.columns:
            if temp_unit == "°F":
                dfh["temperature_u"] = dfh["temperature_2m"].apply(lambda x: c_to_f(float(x)) if pd.notna(x) else None)
                y_title = "Temperature (°F)"
            elif temp_unit == "°C":
                dfh["temperature_u"] = dfh["temperature_2m"]
                y_title = "Temperature (°C)"
            else:
                dfh["temperature_u"] = dfh["temperature_2m"]
                y_title = "Temperature (°C)"
        else:
            y_title = "Temperature"

        st.markdown("#### Temperature (hourly)")
        if "time_dt" in dfh.columns and "temperature_u" in dfh.columns:
            temp_chart = (
                alt.Chart(dfh)
                .mark_area(opacity=0.25)
                .encode(
                    x=alt.X("time_dt:T", title="Time"),
                    y=alt.Y("temperature_u:Q", title=y_title),
                    tooltip=["time_dt:T", alt.Tooltip("temperature_u:Q", title="temp")],
                )
                + alt.Chart(dfh)
                .mark_line()
                .encode(
                    x="time_dt:T",
                    y="temperature_u:Q",
                )
            )
            st.altair_chart(temp_chart.interactive(), use_container_width=True)
        else:
            st.info("Hourly temperature fields not found.")

        st.markdown("#### Precipitation (hourly) - line")
        if "time_dt" in dfh.columns and "precipitation" in dfh.columns:
            pr_line = (
                alt.Chart(dfh)
                .mark_line()
                .encode(
                    x=alt.X("time_dt:T", title="Time"),
                    y=alt.Y("precipitation:Q", title="Precipitation"),
                    tooltip=["time_dt:T", alt.Tooltip("precipitation:Q", title="precip")],
                )
            )
            st.altair_chart(pr_line.interactive(), use_container_width=True)
        else:
            st.info("Hourly precipitation field not found.")

        st.markdown("#### Wind (hourly)")
        if "time_dt" in dfh.columns and "windspeed_10m" in dfh.columns:
            wind_line = (
                alt.Chart(dfh)
                .mark_line()
                .encode(
                    x=alt.X("time_dt:T", title="Time"),
                    y=alt.Y("windspeed_10m:Q", title="Wind speed"),
                    tooltip=["time_dt:T", alt.Tooltip("windspeed_10m:Q", title="wind")],
                )
            )
            st.altair_chart(wind_line.interactive(), use_container_width=True)

        st.markdown("#### Hourly table (latest 200 rows)")
        st.dataframe(
            dfh.sort_values("time_dt", ascending=False).head(200).drop(columns=["time_dt"], errors="ignore"),
            use_container_width=True,
        )

    else:
        st.info("Hourly data not created yet. Run the pipeline.")
