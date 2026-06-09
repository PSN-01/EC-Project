import json
import geopandas as gpd
import pandas as pd

BASE_RAW = "/Users/p.silva/Documents/GitHub/EC-Project/data/raw"
BASE_V = f"{BASE_RAW}/Vancouver Data"
BASE_ATX = f"{BASE_RAW}/Austin Tx Data"

# VANCOUVER DATA

# Vancouver CSVs
service_requests_311_v = pd.read_csv(f"{BASE_V}/3-1-1-service-requests.csv", delimiter=';', low_memory=False)
vancouver_crime = pd.read_csv(f"{BASE_V}/vancouver_crime_data.csv", low_memory=False)

# Vancouver GeoJSONs
city_boundary_v = gpd.read_file(f"{BASE_V}/city-boundary.geojson")
local_area_boundary_v = gpd.read_file(f"{BASE_V}/local-area-boundary.geojson")
property_parcels_v = gpd.read_file(f"{BASE_V}/property-parcel-polygons.geojson")
street_network_v = gpd.read_file(f"{BASE_V}/street_network.geojson")

# Vancouver JSONs
with open(f"{BASE_V}/properties.json") as f:
    properties_v = json.load(f)
with open(f"{BASE_V}/sharp_street_segments.json") as f:
    sharp_street_segs_v = json.load(f)
with open(f"{BASE_V}/street_lights.json") as f:
    street_lights_v = json.load(f)


# AUSTIN TX DATA

# Austin Tx CSVs
austin_311_public = pd.read_csv(f"{BASE_ATX}/Austin_311_Public_Data_20260604.csv", low_memory=False)
austin_crime = pd.read_csv(f"{BASE_ATX}/combined_data.csv", low_memory=False)

# Austin Tx Demo Data
austin_demo_data = pd.read_csv(f"{BASE_ATX}/merged_output.csv", low_memory=False)
austin_demo_data = austin_demo_data[['geoid', 'income_household_median', 'county_name', 'lat', 'lng', 'population']]

# Austin Tx GeoJSONs
neighborhoods_atx = gpd.read_file(f"{BASE_ATX}/Boundaries__City_of_Austin_Neighborhoods_20260605.geojson")
census_tracts_atx = gpd.read_file(f"{BASE_ATX}/Boundaries__State_of_Texas_Census_Tracts_(Based_off_2020_Census)_20260605.geojson")
zip_codes_atx = gpd.read_file(f"{BASE_ATX}/Boundaries__US_Zip_Codes_20260605.geojson")
jurisdictions_atx = gpd.read_file(f"{BASE_ATX}/BOUNDARIES_jurisdictions_20260605.geojson")
census_blocks_atx = gpd.read_file(f"{BASE_ATX}/Census_Block_Groups_20260605.geojson")
downtown_districts_atx = gpd.read_file(f"{BASE_ATX}/Downtown_Austin_Plan_Districts_GEOJSON.geojson")
land_use_atx = gpd.read_file(f"{BASE_ATX}/Land_Use_Inventory_Detailed_20260605.geojson")
street_centerline_atx = gpd.read_file(f"{BASE_ATX}/Street_Centerline_20260605.geojson")
travis_county_boundary = gpd.read_file(f"{BASE_ATX}/Travis_County_Boundary_Polygon.geojson")


