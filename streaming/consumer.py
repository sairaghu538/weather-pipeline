"""
Multi-City Kafka Consumer for Weather Data Pipeline

Consumes weather events from Kafka and writes to Parquet.
Supports multiple cities in a single file with city column for filtering.
"""
import json
import time
from pathlib import Path
import pandas as pd
from kafka import KafkaConsumer

# Configuration
KAFKA_TOPIC = "weather.raw.hourly"
KAFKA_SERVER = "localhost:9092"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "streamed"

def process_event(event):
    """
    Parses the raw Open-Meteo JSON and extracts the current hour's data.
    """
    raw = event.get("payload", {})
    hourly = raw.get("hourly", {})
    
    if not hourly or "time" not in hourly:
        return None
    
    # Get first hour data
    row = {
        "city": event.get("city"),
        "state": event.get("state"),
        "lat": event.get("lat"),
        "lon": event.get("lon"),
        "ingested_at": event.get("ingested_at"),
        "forecast_time": hourly["time"][0] if hourly["time"] else None,
        "temperature_2m": hourly.get("temperature_2m", [None])[0],
        "precipitation": hourly.get("precipitation", [None])[0],
        "windspeed_10m": hourly.get("windspeed_10m", [None])[0]
    }
    return row

def run():
    print(f"🎧 Starting Multi-City Consumer on {KAFKA_TOPIC}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / "weather_stream.parquet"
    
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
    except Exception as e:
        print(f"❌ Failed to connect to Kafka. Is it running?")
        print(e)
        return

    event_count = 0
    batch = []
    BATCH_SIZE = 10  # Write every 10 events for efficiency
    
    for message in consumer:
        event = message.value
        clean_row = process_event(event)
        
        if clean_row:
            event_count += 1
            batch.append(clean_row)
            print(f"📥 [{event_count}] {clean_row['city']} - {clean_row['temperature_2m']}°C")
            
            # Write in batches for efficiency
            if len(batch) >= BATCH_SIZE:
                df_new = pd.DataFrame(batch)
                
                if parquet_path.exists():
                    df_old = pd.read_parquet(parquet_path)
                    df_final = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_final = df_new
                
                df_final.to_parquet(parquet_path)
                print(f"   💾 Saved batch ({len(df_final)} total rows)")
                batch = []

if __name__ == "__main__":
    run()
