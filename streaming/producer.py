import json
import time
import requests
from kafka import KafkaProducer

# Configuration
KAFKA_TOPIC = "weather.raw.hourly"
KAFKA_SERVER = "localhost:9092"
CITY_LAT = 37.3382   # San Jose, CA (Example)
CITY_LON = -121.8863
CITY_NAME = "San Jose, CA"

def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "hourly": "temperature_2m,precipitation,rain,windspeed_10m",
        "timezone": "UTC", 
        "forecast_days": 1 
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    # Return full JSON response ("raw" event)
    return r.json()

def run():
    print(f"🚀 Starting Producer for {CITY_NAME}...")
    print(f"Target Topic: {KAFKA_TOPIC}")
    
    # Initialize Producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"❌ Failed to connect to Kafka at {KAFKA_SERVER}. Is it running?")
        print(e)
        return

    while True:
        try:
            # 1. Fetch
            data = get_weather()
            
            # 2. Enrich
            event = {
                "city": CITY_NAME,
                "lat": CITY_LAT,
                "lon": CITY_LON,
                "ingested_at": time.time(),
                "payload": data
            }
            
            # 3. Publish
            producer.send(KAFKA_TOPIC, event)
            producer.flush()
            
            print(f"✅ Sent event at {time.strftime('%H:%M:%S')}")
            
            # 4. Wait
            time.sleep(60) # Poll every minute
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run()
