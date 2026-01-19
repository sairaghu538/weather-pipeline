"""
Multi-City Kafka Producer for Weather Data Pipeline

Streams weather data for 50 US cities to Kafka every 60 seconds.
Each city is fetched sequentially with a small delay to avoid API rate limits.
"""
import json
import time
from pathlib import Path
import requests
from kafka import KafkaProducer

# Configuration
KAFKA_TOPIC = "weather.raw.hourly"
KAFKA_SERVER = "localhost:9092"
CITIES_FILE = Path(__file__).parent / "cities.json"
POLL_INTERVAL = 900  # 15 minutes between full cycles (API rate limit safe)
API_DELAY = 1  # seconds between API calls (rate limiting)

def load_cities():
    """Load cities from JSON file."""
    with open(CITIES_FILE) as f:
        return json.load(f)

def get_weather(lat: float, lon: float):
    """Fetch weather from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,rain,windspeed_10m",
        "timezone": "UTC",
        "forecast_days": 1
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def run():
    cities = load_cities()
    print(f"🚀 Starting Multi-City Producer ({len(cities)} cities)")
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

    cycle = 0
    while True:
        cycle += 1
        print(f"\n📍 Cycle {cycle} - Fetching {len(cities)} cities...")
        start_time = time.time()
        
        for city in cities:
            city_name = f"{city['name']}, {city['state']}"
            try:
                # Fetch weather
                data = get_weather(city['lat'], city['lon'])
                
                # Create event
                event = {
                    "city": city_name,
                    "state": city['state'],
                    "lat": city['lat'],
                    "lon": city['lon'],
                    "ingested_at": time.time(),
                    "payload": data
                }
                
                # Publish
                producer.send(KAFKA_TOPIC, event)
                print(f"  ✅ {city_name}")
                
                # Rate limit
                time.sleep(API_DELAY)
                
            except Exception as e:
                print(f"  ⚠️ {city_name}: {e}")
        
        producer.flush()
        elapsed = time.time() - start_time
        print(f"✨ Cycle {cycle} complete in {elapsed:.1f}s")
        
        # Wait for next cycle
        wait_time = max(0, POLL_INTERVAL - elapsed)
        if wait_time > 0:
            print(f"⏳ Waiting {wait_time:.0f}s for next cycle...")
            time.sleep(wait_time)

if __name__ == "__main__":
    run()
