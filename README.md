# ⚡ Weather Intelligence Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Altair](https://img.shields.io/badge/Altair-Visualization-orange?style=flat)](https://altair-viz.github.io/)

> **A modern, end-to-end data engineering project featuring a "Glassmorphism" UI, automated pipeline lineage, and hybrid ML forecasting.**

---

## 🚀 Overview

The **Weather Intelligence Pipeline** is a robust data engineering showcase that ingests, transforms, and visualizes weather data in real-time. Unlike standard dashboards, this project focuses on **Pipeline Observability** and **Data Quality**, providing a "Control Room" experience for monitoring ingestion flows and artifact integrity.

It combines **Open-Meteo** API data with a custom **NOAA-trained ML model** to provide unique, hybrid forecasts with accuracy tracking.

<!-- User can add screenshot here -->
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-5.png)
![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-6.png)
![alt text](image-7.png)

<!-- ![Dashboard Overview](docs/images/dashboard_main.png) -->

---

## ✨ Key Features

### 🔧 1. Data Engineering Suite
A dedicated "Control Room" tab provides deep visibility into the backend processes:
- **Visual Pipeline Lineage**: Real-time Mermaid DAG showing data flow from API to Parquet.
- **Execution Telemetry**: Trace execution time and status for every run.
- **Artifact Explorer**: Built-in JSON and Parquet viewer to inspect raw vs curated data without leaving the UI.
- **Data Quality Inspector**: Automated checks for null values, data freshness, and completeness.

### 📊 2. Interactive Analytics
- **Daily & Hourly Views**: Switch between long-term trends and high-resolution hourly data.
- **Multi-Metric Visualization**:
    - Temperature Trends (Area Charts)
    - Precipitation Levels (Bar Charts)
    - Wind Patterns (Area/Line Charts)
- **Dynamic Filtering**: Instant city search with geocoding and history.

### 🤖 3. Hybrid ML Forecasting
- **Proprietary Model**: Trains a light-weight regression model on 30/90/365 days of historical data from NOAA GHCN stations.
- **Model vs API Comparison**: Benchmarks the custom ML prediction against Open-Meteo's forecast to surface divergences.
- **Transparent Accuracy**: Displays Mean Absolute Error (MAE) and training metadata for every prediction.

---

## 🏗️ Architecture

The pipeline follows a modern ETL pattern:

```mermaid
graph LR
    A[Open-Meteo API] -->|Ingest JSON| B(Raw Landing Zone)
    B -->|Pandas Transform| C{Curated Zone}
    C -->|Aggregations| D[Hourly/Daily Parquet]
    D -->|Feature Eng| E[ML Model Training]
    E -->|Inference| F[Streamlit Dashboard]
    
    style A fill:#0f172a,stroke:#334155,color:#fff
    style B fill:#1e293b,stroke:#7c3aed,color:#fff
    style D fill:#1e293b,stroke:#38bdf8,color:#fff
    style F fill:#312e81,stroke:#a78bfa,color:#fff
```

1.  **Ingest**: Fetches raw hourly weather data for the selected location (Open-Meteo).
2.  **Transform**: Cleanses data, handles type casting, and standardizes timestamps.
3.  **Store**: Saves intermediate artifacts as optimized **Parquet** files (`weather_hourly.parquet`, `weather_daily.parquet`).
4.  **Serve**: Streamlit loads the curated data for visualization and interactive filtering.

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Altair (Declarative Statistical Visualization)
- **App Framework**: Streamlit (with Custom CSS/Glassmorphism)
- **External APIs**: Open-Meteo (Weather), Geopy/Open-Meteo (Geocoding)

---

## ⚡ Getting Started

### Prerequisites
- Python 3.8 or higher

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/sairaghu538/weather-pipeline.git
    cd weather-pipeline
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
.
├── app.py                 # Main Streamlit Dashboard application
├── pipeline/              # ETL Logic Module
│   ├── config.py          # Configuration constants
│   ├── ingest.py          # API fetching & Raw storage
│   ├── transform.py       # Pandas transformations & Parquet writing
│   └── weather_ml.py      # NOAA ML Model & Inference
├── data/                  # Local data storage (Gitignored)
│   ├── raw/               # JSON Landing Zone
│   └── curated/           # Parquet Tables
├── .streamlit/            # App theming and config
└── requirements.txt       # Python dependencies
```