# Weather Data Engineering Project 

## What this shows
- API ingestion (Open-Meteo)
- Raw JSON landing zone
- Curated Parquet tables
- Daily aggregates
- Data quality checks
- Streamlit dashboard

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
