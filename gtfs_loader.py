# gtfs_loader.py

import pandas as pd
import requests
import io

def fetch_csv(url):
    res = requests.get(url)
    res.raise_for_status()
    return pd.read_csv(io.StringIO(res.content.decode("utf-8")))

# Supabase base URL (note: double slashes are preserved from your links)
BASE_URL = "https://wgcyfjhtsmtaizsqveuw.supabase.co/storage/v1/object/public/gtfs//"

def load_gtfs_tables() -> dict:
    return {
        "agency": fetch_csv(BASE_URL + "agency.txt"),
        "calendar_dates": fetch_csv(BASE_URL + "calendar_dates.txt"),
        "routes": fetch_csv(BASE_URL + "routes.txt"),
        "shapes": fetch_csv(BASE_URL + "shapes.txt"),
        "stops": fetch_csv(BASE_URL + "stops.txt"),
        "trips": fetch_csv(BASE_URL + "trips.txt"),
        # Note: stop_times.txt is missing; can be added when uploaded
    }

# Load once when the module is imported
gtfs = load_gtfs_tables()

