from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.config import CITIES, RAW_DIR, CURATED_DIR
from pipeline.ingest import ingest_cities
from pipeline.transform import build_curated, build_daily_aggregates
from pipeline.dq import run_dq

st.set_page_config(page_title="Weather Data Pipeline", layout="wide")

st.title("Weather Data Engineering Project")
st.caption("Open-Meteo ingestion → curated parquet → quality checks → dashboard")

hourly_path = CURATED_DIR / "weather_hourly.parquet"
daily_path = CURATED_DIR / "weather_daily.parquet"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline")
    st.write("Cities")
    st.write([c.name for c in CITIES])

    if st.button("Run pipeline now"):
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
        st.dataframe(dfd.sort_values(["date", "city"], ascending=[False, True]), use_container_width=True)
    else:
        st.info("Daily aggregates not created yet.")

with tab2:
    if hourly_path.exists():
        dfh = pd.read_parquet(hourly_path)
        st.dataframe(dfh.tail(200), use_container_width=True)
    else:
        st.info("Hourly data not created yet.")
