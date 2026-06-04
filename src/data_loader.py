import json
import geopandas as gpd
import pandas as pd

BASE = "/Users/p.silva/Documents/GitHub/EC-Project/data/raw"

# CSVs
service_requests_311 = pd.read_csv(f"{BASE}/3-1-1-service-requests.csv", delimiter=';', low_memory=False)
# austin_full = pd.read_csv(f"{BASE}/austin_full.csv", low_memory=False)
# austin_crime = pd.read_csv(f"{BASE}/austin_texas_crime_data.csv", low_memory=False)
vancouver_crime = pd.read_csv(f"{BASE}/vancouver_crime_data.csv", low_memory=False)
austin_crime = pd.read_excel(f"{BASE}/2025_Data.xlsx")

# GeoJSONs
city_boundary = gpd.read_file(f"{BASE}/city-boundary.geojson")
local_area_boundary = gpd.read_file(f"{BASE}/local-area-boundary.geojson")
property_parcels = gpd.read_file(f"{BASE}/property-parcel-polygons.geojson")
street_network = gpd.read_file(f"{BASE}/street_network.geojson")

# JSONs
with open(f"{BASE}/properties.json") as f: properties = json.load(f)
with open(f"{BASE}/sharp_street_segments.json") as f: sharp_street_segs = json.load(f)
with open(f"{BASE}/street_lights.json") as f: street_lights = json.load(f)