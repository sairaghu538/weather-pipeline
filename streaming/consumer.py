import json
import time
from pathlib import Path
import pandas as pd
from kafka import KafkaConsumer

# Configuration
KAFKA_TOPIC = "weather.raw.hourly"
KAFKA_SERVER = "localhost:9092"
OUTPUT_DIR = Path("../data/streamed")

def process_event(event):
    """
    Parses the raw Open-Meteo JSON and extracts the current hour's data.
    """
    raw = event.get("payload", {})
    hourly = raw.get("hourly", {})
    
    # Simple logic: take the very first row (current hour approx)
    # real logic would find the exact matching hour index
    if not hourly or "time" not in hourly:
        return None
        
    row = {
        "city": event.get("city"),
        "ingested_at": event.get("ingested_at"),
        "forecast_time": hourly["time"][0],
        "temperature_2m": hourly["temperature_2m"][0],
        "precipitation": hourly["precipitation"][0],
        "windspeed_10m": hourly["windspeed_10m"][0]
    }
    return row

def run():
    print(f"🎧 Starting Consumer on {KAFKA_TOPIC}...")
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

    buffer = []
    
    for message in consumer:
        event = message.value
        clean_row = process_event(event)
        
        if clean_row:
            print(f"📥 Received update for {clean_row['city']} ({clean_row['forecast_time']})")
            buffer.append(clean_row)
            
            # Write to parquet every update (for visual effect) or batch it
            # For this demo, we append immediately
            df_new = pd.DataFrame([clean_row])
            
            if parquet_path.exists():
                df_old = pd.read_parquet(parquet_path)
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_final = df_new
                
            df_final.to_parquet(parquet_path)
            print("   💾 Saved to parquet")

if __name__ == "__main__":
    run()
