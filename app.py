from __future__ import annotations

from datetime import datetime, timezone, date
from typing import Any

import pandas as pd
import requests
import streamlit as st
import altair as alt

from pipeline.config import RAW_DIR, CURATED_DIR, City
from pipeline.ingest import ingest_cities
from pipeline.transform import build_curated, build_daily_aggregates

# NOAA ML forecast
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
            out[f"{c}_u"] = out[c]  # plot Celsius by default
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


@st.cache_data(ttl=3600)
def open_meteo_daily_avg_for_date(lat: float, lon: float, target_ymd: str) -> dict[str, Any]:
    """
    Return Open-Meteo DAILY mean temperature for an exact date (UTC).
    Uses start_date/end_date to fetch precisely the requested day.
    Output:
      {"date": "YYYY-MM-DD" | None, "tmean_c": float | None, "tmin_c": float|None, "tmax_c": float|None}
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "timezone": "UTC",
        "start_date": target_ymd,
        "end_date": target_ymd,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    tmean = daily.get("temperature_2m_mean") or []
    tmin = daily.get("temperature_2m_min") or []
    tmax = daily.get("temperature_2m_max") or []

    if not dates:
        return {"date": None, "tmean_c": None, "tmin_c": None, "tmax_c": None}

    def _safe_float(arr: list[Any], idx: int = 0) -> float | None:
        if len(arr) <= idx or arr[idx] is None:
            return None
        try:
            return float(arr[idx])
        except Exception:
            return None

    return {
        "date": dates[0],
        "tmean_c": _safe_float(tmean, 0),
        "tmin_c": _safe_float(tmin, 0),
        "tmax_c": _safe_float(tmax, 0),
    }


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

    return df[df["city"] == selected_city].copy(), selected_city


def normalize_city_for_noaa(selected_city_label: str) -> str:
    """
    NOAA path uses geopy Nominatim. Give it an unambiguous string.
    """
    s = (selected_city_label or "").strip()
    if not s:
        return s
    if "USA" in s.upper() or "UNITED STATES" in s.upper():
        return s
    return f"{s}, USA"


def _parse_ymd(x: Any) -> date | None:
    if x is None or pd.isna(x):
        return None
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


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
        # NEW fields from weather_ml.py
        "noaa_last_date": res.noaa_last_date,
        "target_date": res.target_date,
    }


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Weather Data Pipeline", layout="wide")

# Sidebar
st.sidebar.markdown("### Temperature unit")
temp_unit = st.sidebar.radio(
    "Display temperature as",
    options=["°C", "°F", "°C + °F"],
    index=2,
)

st.sidebar.markdown("### ML forecast (NOAA)")
ml_days = st.sidebar.selectbox("Training window", options=[30, 90, 365], index=0)
ml_auto_run = st.sidebar.checkbox("Auto-run ML when city changes", value=True)

st.title("Weather Data Engineering Project")
st.caption("Open-Meteo ingestion → curated parquet → dashboard")

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

tab1, tab2 = st.tabs(["Daily forecast", "Hourly forecast"])

# ============================
# Daily tab
# ============================
with tab1:
    if not daily_path.exists():
        st.info("Daily aggregates not created yet. Run the pipeline.")
    else:
        dfd = pd.read_parquet(daily_path)
        dfd, selected_name = city_filter_ui(dfd, key_prefix="daily", label="Select city for daily view")

        if "date" in dfd.columns:
            dfd = dfd.sort_values("date")

        # ----------------------------
        # Header (real header section)
        # ----------------------------
        city_title = selected_name or "—"
        last_updated_dt = None
        if "date" in dfd.columns and len(dfd) > 0:
            last_updated_dt = _parse_ymd(dfd["date"].max())

        st.markdown(f"## {city_title}")
        if last_updated_dt:
            st.caption(f"Last updated (Open-Meteo daily): {last_updated_dt}  |  Open-Meteo observations + NOAA ML next day forecast")
        else:
            st.caption("Open-Meteo observations + NOAA ML next day forecast")

        # Latest Open-Meteo day metrics
        last_row = dfd.tail(1)
        latest_openmeteo_date = None
        latest_openmeteo_avg_c = None
        latest_openmeteo_min_c = None
        latest_openmeteo_max_c = None
        latest_openmeteo_precip = None

        if len(last_row) == 1:
            r = last_row.iloc[0]
            latest_openmeteo_date = r.get("date")
            latest_openmeteo_avg_c = r.get("temp_avg")
            latest_openmeteo_min_c = r.get("temp_min")
            latest_openmeteo_max_c = r.get("temp_max")
            latest_openmeteo_precip = r.get("precip_sum")

        # ----------------------------
        # ML forecast + Open-Meteo reference (same target date)
        # ----------------------------
        st.markdown("### Forecast summary")

        hero_placeholder = st.empty()

        ml_out = None
        om_ref = {"date": None, "tmean_c": None, "tmin_c": None, "tmax_c": None}
        delta_c = None
        delta_f = None

        if selected_name:
            noaa_city_query = normalize_city_for_noaa(selected_name)

            # Geocode for Open-Meteo reference
            city_lat = None
            city_lon = None
            try:
                city_matches = geocode_us_cities(selected_name.split(",")[0], limit=15)
                chosen = None
                for m in city_matches:
                    if format_city_option(m).strip().lower() == selected_name.strip().lower():
                        chosen = m
                        break
                if chosen is None and city_matches:
                    chosen = city_matches[0]
                if chosen is not None:
                    city_lat = float(chosen["lat"])
                    city_lon = float(chosen["lon"])
            except Exception:
                city_lat = None
                city_lon = None

            run_ml = ml_auto_run or st.button("Run ML forecast now", key="ml_run_btn")

            if run_ml:
                try:
                    with st.spinner("Training model and predicting using NOAA daily data..."):
                        ml_out = cached_noaa_forecast(noaa_city_query, ml_days)

                    ml_pred_c = ml_out["predicted_avg_temp_c"]
                    mae_c = ml_out["test_mae_c"]
                    rows_total = ml_out["rows_total"]
                    rows_train = ml_out["rows_train"]
                    rows_test = ml_out["rows_test"]
                    station_name = ml_out["station_name"]
                    target_date = ml_out.get("target_date")
                    noaa_last_date = ml_out.get("noaa_last_date")

                    # Open-Meteo reference for the SAME target day (key fix)
                    if city_lat is not None and city_lon is not None and target_date:
                        try:
                            om_ref = open_meteo_daily_avg_for_date(city_lat, city_lon, target_date)
                        except Exception:
                            om_ref = {"date": None, "tmean_c": None, "tmin_c": None, "tmax_c": None}

                    om_tmean_c = om_ref.get("tmean_c")

                    if om_tmean_c is not None and ml_pred_c is not None:
                        try:
                            delta_c = float(ml_pred_c) - float(om_tmean_c)
                            delta_f = c_to_f(float(ml_pred_c)) - c_to_f(float(om_tmean_c))
                        except Exception:
                            delta_c = None
                            delta_f = None

                    # Hero metric
                    hero_big = format_temp(ml_pred_c, temp_unit)
                    hero_small = "—"
                    if om_tmean_c is not None and delta_c is not None:
                        hero_small = f"Open-Meteo {target_date} avg {format_temp(om_tmean_c, temp_unit)}  |  Delta {format_temp(delta_c, '°C') if temp_unit=='°C' else (f'{delta_f:+.1f} °F' if temp_unit=='°F' else f'{delta_c:+.1f} °C / {delta_f:+.1f} °F')}"
                    elif target_date:
                        hero_small = f"Target day: {target_date} (Open-Meteo reference unavailable)"

                    hero_placeholder.markdown(
                        f"""
                        <div style="padding:16px;border-radius:12px;border:1px solid rgba(0,0,0,0.08);">
                          <div style="font-size:12px;opacity:0.7;">NOAA ML predicted next-day average</div>
                          <div style="font-size:44px;line-height:1.1;font-weight:700;">{hero_big}</div>
                          <div style="font-size:13px;opacity:0.75;margin-top:6px;">{hero_small}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Two cards: Open-Meteo vs NOAA ML
                    st.markdown("### Details")

                    c1, c2 = st.columns(2)

                    # Card 1: Open-Meteo (latest available day in curated daily)
                    with c1:
                        st.markdown(
                            """
                            <div style="padding:14px;border-radius:12px;border:1px solid rgba(0,0,0,0.08);">
                              <div style="font-size:14px;font-weight:650;">Open-Meteo (latest available day)</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Max", format_temp(latest_openmeteo_max_c, temp_unit))
                        m2.metric("Min", format_temp(latest_openmeteo_min_c, temp_unit))
                        m3.metric("Avg", format_temp(latest_openmeteo_avg_c, temp_unit))
                        if latest_openmeteo_precip is None or pd.isna(latest_openmeteo_precip):
                            m4.metric("Precip", "—")
                        else:
                            m4.metric("Precip", f"{float(latest_openmeteo_precip):.1f}")
                        if latest_openmeteo_date:
                            st.caption(f"Day: {latest_openmeteo_date}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Card 2: NOAA ML (target day)
                    with c2:
                        st.markdown(
                            """
                            <div style="padding:14px;border-radius:12px;border:1px solid rgba(0,0,0,0.08);">
                              <div style="font-size:14px;font-weight:650;">NOAA ML (target day)</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        ml1, ml2, ml3, ml4 = st.columns(4)
                        ml1.metric("Pred avg", format_temp(ml_pred_c, temp_unit))
                        ml2.metric("Open-Meteo ref", format_temp(om_ref.get("tmean_c"), temp_unit))

                        if delta_c is None:
                            ml3.metric("Delta", "—")
                        else:
                            if temp_unit == "°F":
                                ml3.metric("Delta", f"{delta_f:+.1f} °F")
                            elif temp_unit == "°C":
                                ml3.metric("Delta", f"{delta_c:+.1f} °C")
                            else:
                                ml3.metric("Delta", f"{delta_c:+.1f} °C / {delta_f:+.1f} °F")

                        ml4.metric("MAE (°C)", "—" if mae_c is None else f"{mae_c:.2f}")

                        st.caption(
                            f"Target day: {target_date or '—'}  |  NOAA last obs: {noaa_last_date or '—'}  |  Rows: {rows_total} (train {rows_train}, test {rows_test})"
                        )
                        st.caption(f"Station: {station_name}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("Debug details"):
                        st.write(
                            {
                                "ui_city": selected_name,
                                "noaa_city_query": noaa_city_query,
                                "training_days": ml_out["days_used"],
                                "noaa_last_date": ml_out.get("noaa_last_date"),
                                "target_date": ml_out.get("target_date"),
                                "noaa_station_id": ml_out["station_id"],
                                "noaa_station_name": ml_out["station_name"],
                                "ml_predicted_avg_temp_c": ml_out["predicted_avg_temp_c"],
                                "open_meteo_target_date": om_ref.get("date"),
                                "open_meteo_target_avg_c": om_ref.get("tmean_c"),
                                "delta_c": delta_c,
                                "delta_f": delta_f,
                                "noaa_parquet_path": ml_out["parquet_path"],
                            }
                        )

                except Exception as e:
                    st.error(f"ML forecast failed: {e}")
                    st.info("Tip: try 90 days in the sidebar, or use a more specific city like 'San Jose, CA'.")
        else:
            st.info("Pick a city to see forecast details.")

        st.divider()

        # ----------------------------
        # Charts
        # ----------------------------
        temp_cols = [c for c in ["temp_min", "temp_avg", "temp_max"] if c in dfd.columns]
        dfd = add_temp_cols(dfd, temp_cols, temp_unit)

        show_band = st.toggle("Show min/max band", value=True)

        st.markdown("### Temperature trend (daily)")
        if "date" in dfd.columns and temp_cols:
            plot_cols = {c: f"{c}_u" for c in temp_cols if f"{c}_u" in dfd.columns}
            y_title = "Temperature (°C)" if temp_unit == "°C" else "Temperature (°F)" if temp_unit == "°F" else "Temperature (°C)"

            base = alt.Chart(dfd).encode(x=alt.X("date:T", title="Date"))

            # Band between min and max
            if show_band and "temp_min" in plot_cols and "temp_max" in plot_cols:
                band = base.mark_area(opacity=0.18).encode(
                    y=alt.Y(f"{plot_cols['temp_min']}:Q", title=y_title),
                    y2=alt.Y2(f"{plot_cols['temp_max']}:Q"),
                    tooltip=[
                        "date:T",
                        alt.Tooltip(f"{plot_cols['temp_min']}:Q", title="Min"),
                        alt.Tooltip(f"{plot_cols['temp_max']}:Q", title="Max"),
                    ],
                )
            else:
                # fallback avg area
                area_field = plot_cols.get("temp_avg") or plot_cols.get("temp_max")
                band = base.mark_area(opacity=0.20).encode(
                    y=alt.Y(f"{area_field}:Q", title=y_title),
                    tooltip=["date:T", alt.Tooltip(f"{area_field}:Q", title="Temp")],
                )

            # Lines
            lines = []
            for label, col in plot_cols.items():
                pretty = label.replace("temp_", "").upper()
                lines.append(
                    base.mark_line().encode(
                        y=alt.Y(f"{col}:Q"),
                        tooltip=["date:T", alt.Tooltip(f"{col}:Q", title=pretty)],
                    )
                )

            chart = band
            for ln in lines:
                chart = chart + ln

            st.altair_chart(chart.interactive(), use_container_width=True)
        else:
            st.info("Temperature fields not found in daily data.")

        st.markdown("### Precipitation (daily)")
        if "date" in dfd.columns and "precip_sum" in dfd.columns:
            precip_bars = (
                alt.Chart(dfd)
                .mark_bar()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("precip_sum:Q", title="Precipitation"),
                    tooltip=["date:T", alt.Tooltip("precip_sum:Q", title="Precip")],
                )
            )
            st.altair_chart(precip_bars.interactive(), use_container_width=True)
        else:
            st.info("Precipitation field not found in daily data.")

        with st.expander("Show raw daily table"):
            st.dataframe(dfd.sort_values("date", ascending=False), use_container_width=True)

# ============================
# Hourly tab
# ============================
with tab2:
    if not hourly_path.exists():
        st.info("Hourly data not created yet. Run the pipeline.")
    else:
        dfh = pd.read_parquet(hourly_path)
        dfh, selected_name = city_filter_ui(dfh, key_prefix="hourly", label="Select city for hourly view")

        dfh["time_dt"] = safe_datetime_series(dfh, "time")
        dfh = dfh.sort_values("time_dt")

        if "temperature_2m" in dfh.columns:
            if temp_unit == "°F":
                dfh["temperature_u"] = dfh["temperature_2m"].apply(lambda x: c_to_f(float(x)) if pd.notna(x) else None)
                y_title = "Temperature (°F)"
            else:
                dfh["temperature_u"] = dfh["temperature_2m"]
                y_title = "Temperature (°C)"
        else:
            y_title = "Temperature"

        st.markdown("### Temperature (hourly)")
        if "time_dt" in dfh.columns and "temperature_u" in dfh.columns:
            temp_area = (
                alt.Chart(dfh)
                .mark_area(opacity=0.22)
                .encode(
                    x=alt.X("time_dt:T", title="Time"),
                    y=alt.Y("temperature_u:Q", title=y_title),
                    tooltip=["time_dt:T", alt.Tooltip("temperature_u:Q", title="Temp")],
                )
            )
            temp_line = (
                alt.Chart(dfh)
                .mark_line()
                .encode(
                    x="time_dt:T",
                    y="temperature_u:Q",
                )
            )
            st.altair_chart((temp_area + temp_line).interactive(), use_container_width=True)
        else:
            st.info("Hourly temperature fields not found.")

        st.markdown("### Precipitation (hourly)")
        if "time_dt" in dfh.columns and "precipitation" in dfh.columns:
            pr_line = (
                alt.Chart(dfh)
                .mark_line()
                .encode(
                    x=alt.X("time_dt:T", title="Time"),
                    y=alt.Y("precipitation:Q", title="Precipitation"),
                    tooltip=["time_dt:T", alt.Tooltip("precipitation:Q", title="Precip")],
                )
            )
            st.altair_chart(pr_line.interactive(), use_container_width=True)
        else:
            st.info("Hourly precipitation field not found.")

        st.markdown("### Wind (hourly)")
        if "time_dt" in dfh.columns and "windspeed_10m" in dfh.columns:
            wind_line = (
                alt.Chart(dfh)
                .mark_line()
                .encode(
                    x=alt.X("time_dt:T", title="Time"),
                    y=alt.Y("windspeed_10m:Q", title="Wind speed"),
                    tooltip=["time_dt:T", alt.Tooltip("windspeed_10m:Q", title="Wind")],
                )
            )
            st.altair_chart(wind_line.interactive(), use_container_width=True)

        with st.expander("Show raw hourly table (latest 200 rows)"):
            st.dataframe(
                dfh.sort_values("time_dt", ascending=False).head(200).drop(columns=["time_dt"], errors="ignore"),
                use_container_width=True,
            )
