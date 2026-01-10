# Weather Data Engineering Project

A small end to end weather data pipeline with a Streamlit dashboard.

It ingests weather data from Open-Meteo, writes raw JSON to a landing zone, transforms into curated Parquet tables, builds daily aggregates, and visualizes everything with charts and metrics. It also includes an optional NOAA based ML forecast for next day average temperature.

---

## What this project shows

- API ingestion using Open-Meteo
- Raw JSON landing zone (local files)
- Curated Parquet tables (hourly + daily)
- Daily aggregates and summary metrics
- Basic data quality handling (nulls, type coercion, date parsing)
- Streamlit dashboard with:
  - City search and selection
  - Temperature unit toggle (°C, °F, both)
  - Daily temperature trend with area fill
  - Daily precipitation line chart
  - Hourly temperature chart with area fill + line
  - Hourly precipitation line chart
  - Hourly wind line chart
- NOAA ML forecast (next day avg temp) using GHCN Daily station history
  - Training windows: 30, 90, 365 days
  - Shows MAE and station metadata
  - Compares ML predicted tomorrow avg vs Open-Meteo tomorrow forecast avg

---

## Tech stack

- Python
- Pandas
- Streamlit
- Altair
- Requests
- Geopy (NOAA city geocoding)
- Scikit-learn (simple regression model)
- Parquet storage

---

## Project structure (typical)

```text
.
├── app.py
├── requirements.txt
├── pipeline/
│   ├── config.py
│   ├── ingest.py
│   ├── transform.py
│   └── weather_ml.py
├── data/                  # generated (local)
├── models/                # optional (local)
└── .cache_noaa/            # NOAA station + dly cache (local)



Setup
1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

2) Install dependencies
pip install -r requirements.txt

3) Run the Streamlit app
streamlit run app.py

How to use the app
Pipeline run

Search a US city (example: New Castle)

Pick the exact city from the dropdown

Click Run pipeline now

The app writes:

Raw JSON into the raw directory

Curated Parquet into the curated directory

Daily forecast tab

Shows latest day metrics: max, min, avg, precipitation

Shows NOAA ML next day average prediction

Shows Open-Meteo tomorrow average (forecast) as a reference

Shows delta: ML minus Open-Meteo

Charts:

Temperature trend (daily) with area fill

Precipitation (daily) line chart

Table view of daily rows

Hourly forecast tab

Hourly temperature chart (area + line)

Hourly precipitation line chart

Hourly wind line chart

Latest rows table (trimmed)

NOAA ML forecast details

What it is:

Uses NOAA GHCN Daily station history near the selected city

Trains a lightweight regression model using lag features

Predicts tomorrow’s average temperature

What it is not:

It is not expected to exactly match Open-Meteo or Google weather.

Differences happen because data sources and modeling methods differ.

If it fails:

Some nearby stations may not have usable rows for the requested window.

The ML module tries multiple nearby stations, but some cities may still fail.