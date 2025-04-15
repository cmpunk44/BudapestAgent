import pandas as pd

BASE_URL = "https://wgcyfjhtsmtaizsqveuw.supabase.co/storage/v1/object/public/gtfs"

def load_gtfs_tables(base_url: str) -> dict:
    return {
        "stops": pd.read_csv(f"{base_url}/stops.txt"),
        "routes": pd.read_csv(f"{base_url}/routes.txt"),
        "trips": pd.read_csv(f"{base_url}/trips.txt"),
        "stop_times": pd.read_csv(f"{base_url}/stop_times.txt"),
        "calendar": pd.read_csv(f"{base_url}/calendar.txt"),
    }

# Preload once when the module is imported
gtfs = load_gtfs_tables(BASE_URL)
