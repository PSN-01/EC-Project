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

# Spatial Joins
joined_crime = gpd.sjoin(gdf_crime, gdf_boundaries[[boundary_col, 'geometry']], how="inner", predicate="within")
joined_311 = gpd.sjoin(gdf_311, gdf_boundaries[[boundary_col, 'geometry']], how="inner", predicate="within")


#%%
"""
CRIME MAP: CHOROPLETH ONLY
"""

crime_counts = joined_crime.groupby(boundary_col).size().reset_index()
crime_counts.columns = ['Boundary_Name', 'Total_Crimes']
quantiles_crime = crime_counts['Total_Crimes'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()

crime_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=gdf_boundaries[[boundary_col, 'geometry']],
    name='Crime Density (2014-2026)',
    data=crime_counts,
    columns=['Boundary_Name', 'Total_Crimes'],
    key_on=f'feature.properties.{boundary_col}',
    fill_color='YlOrRd',
    fill_opacity=0.6,
    line_opacity=0.5,
    legend_name='Total Crimes',
    bins=quantiles_crime
).add_to(crime_map)

folium.LayerControl().add_to(crime_map)
crime_map.save("maps/crime_map_austin.html")


#%%
"""
311 REPORTS MAP: CHOROPLETH ONLY
"""

sr_counts = joined_311.groupby(boundary_col).size().reset_index()
sr_counts.columns = ['Boundary_Name', 'Total_Reports']
quantiles_311 = sr_counts['Total_Reports'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()

sr_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=gdf_boundaries[[boundary_col, 'geometry']],
    name='311 Report Density (2014-2026)',
    data=sr_counts,
    columns=['Boundary_Name', 'Total_Reports'],
    key_on=f'feature.properties.{boundary_col}',
    fill_color='PuBu',
    fill_opacity=0.6,
    line_opacity=0.5,
    legend_name='Total 311 Reports',
    bins=quantiles_311
).add_to(sr_map)

folium.LayerControl().add_to(sr_map)
sr_map.save("maps/311_reports_map_austin.html")