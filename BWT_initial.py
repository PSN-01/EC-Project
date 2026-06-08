#%%
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, HeatMapWithTime
from folium.features import DivIcon

from src.data_loader import (
    austin_crime,
    census_tracts_atx,
    street_centerline_atx,
    austin_311_public
)

#%%

# Census Tracts
gdf_boundaries = census_tracts_atx.to_crs("EPSG:4326")
boundary_col = 'geoid'

# Crime 2014-2026
df_crime = austin_crime.copy()
df_crime['Occurred Date'] = pd.to_datetime(df_crime['Occurred Date'], errors='coerce')
df_crime = df_crime[(df_crime['Occurred Date'].dt.year >= 2014) & (df_crime['Occurred Date'].dt.year <= 2026)].copy()

df_crime['Latitude'] = pd.to_numeric(df_crime['Latitude'], errors='coerce')
df_crime['Longitude'] = pd.to_numeric(df_crime['Longitude'], errors='coerce')
df_crime_clean = df_crime.dropna(subset=['Latitude', 'Longitude'])

gdf_crime = gpd.GeoDataFrame(
    df_crime_clean,
    geometry=gpd.points_from_xy(df_crime_clean['Longitude'], df_crime_clean['Latitude']),
    crs="EPSG:4326"
)

# Bylaw 2014-2026
df_311 = austin_311_public.copy()
df_311['Created Date'] = pd.to_datetime(df_311['Created Date'], errors='coerce')
df_311 = df_311[(df_311['Created Date'].dt.year >= 2014) & (df_311['Created Date'].dt.year <= 2026)].copy()

df_311['Latitude Coordinate'] = pd.to_numeric(df_311['Latitude Coordinate'], errors='coerce')
df_311['Longitude Coordinate'] = pd.to_numeric(df_311['Longitude Coordinate'], errors='coerce')
df_311_clean = df_311.dropna(subset=['Latitude Coordinate', 'Longitude Coordinate'])

gdf_311 = gpd.GeoDataFrame(
    df_311_clean,
    geometry=gpd.points_from_xy(df_311_clean['Longitude Coordinate'], df_311_clean['Latitude Coordinate']),
    crs="EPSG:4326"
)

# 4. Spatial Joins
joined_crime = gpd.sjoin(gdf_crime, gdf_boundaries[[boundary_col, 'geometry']], how="inner", predicate="within")
joined_311 = gpd.sjoin(gdf_311, gdf_boundaries[[boundary_col, 'geometry']], how="inner", predicate="within")