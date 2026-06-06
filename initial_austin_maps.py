import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, HeatMapWithTime
from folium.features import DivIcon

from src.data_loader import (
    austin_crime,
    census_tracts_atx,
    street_centerline_atx
)

#%%
"""
DATA PREPARATION (CENSUS TRACTS)
"""
gdf_streets = street_centerline_atx.to_crs("EPSG:4326")
gdf_boundaries = census_tracts_atx.to_crs("EPSG:4326")

for col in gdf_boundaries.select_dtypes(include=['datetime64', 'datetimetz']).columns:
    gdf_boundaries[col] = gdf_boundaries[col].astype(str)

for col in gdf_streets.select_dtypes(include=['datetime64', 'datetimetz']).columns:
    gdf_streets[col] = gdf_streets[col].astype(str)

df_austin = austin_crime.copy()
df_austin['Occurred Date'] = pd.to_datetime(df_austin['Occurred Date'], errors='coerce')
df_austin = df_austin[df_austin['Occurred Date'].dt.year == 2025].copy()

df_austin['Latitude'] = pd.to_numeric(df_austin['Latitude'], errors='coerce')
df_austin['Longitude'] = pd.to_numeric(df_austin['Longitude'], errors='coerce')

df_clean = df_austin[
    (df_austin['Latitude'].notnull()) &
    (df_austin['Longitude'].notnull())
].copy()

gdf = gpd.GeoDataFrame(
    df_clean,
    geometry=gpd.points_from_xy(df_clean['Longitude'], df_clean['Latitude']),
    crs="EPSG:4326"
)

#%%
"""
SIMPLE INTERACTIVE MAP
"""
austin_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')
cluster = MarkerCluster().add_to(austin_map)

for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color="crimson",
        fill=True,
        fill_opacity=0.7,
        popup=f"<b>Type:</b> {row.get('Highest Offense Description', 'N/A')}<br><b>Address:</b> {row.get('Address', 'N/A')}"
    ).add_to(cluster)

# austin_map.save("Austin_Map.html")


#%%
"""
UNIFIED CHOROPLETH AND POINTS MAP
"""
gdf_joined = gpd.sjoin(gdf, gdf_boundaries, how="inner", predicate="within")

boundary_col = 'geoid'
crime_counts = gdf_joined.groupby(boundary_col).size().reset_index()
crime_counts.columns = ['Boundary_Name', 'Total_Crimes']

quantiles = crime_counts['Total_Crimes'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()

unified_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=gdf_boundaries,
    name='Census Tract Density',
    data=crime_counts,
    columns=['Boundary_Name', 'Total_Crimes'],
    key_on=f'feature.properties.{boundary_col}',
    fill_color='YlOrRd',
    fill_opacity=0.6,
    line_opacity=0.5,
    legend_name='Total Crimes (2025)',
    bins=quantiles
).add_to(unified_map)

gdf_merged = gdf_boundaries.merge(crime_counts, left_on=boundary_col, right_on='Boundary_Name')

for _, row in gdf_merged.iterrows():
    centroid = row.geometry.centroid
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=DivIcon(
            icon_size=(150, 36),
            icon_anchor=(75, 18),
            html=f'<div style="font-size: 11pt; font-weight: bold; text-shadow: 1px 1px 2px white; text-align: center;">{row["Total_Crimes"]}</div>'
        )
    ).add_to(unified_map)

points_group = folium.FeatureGroup(name="Crimes").add_to(unified_map)

for idx, row in gdf_joined.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color='#b3ecec',
        fill=True,
        fill_opacity=0.7,
        weight=0,
        popup=f"Boundary: {row[boundary_col]}<br>Type: {row.get('Highest Offense Description', 'N/A')}"
    ).add_to(points_group)

folium.LayerControl().add_to(unified_map)
unified_map.save("unified_map_austin.html")


#%%
"""
INFRASTRUCTURE MAP (CENSUS TRACTS & STREETS)
"""
infrastructure_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')

folium.GeoJson(
    gdf_boundaries,
    name="Census Tract Boundaries",
    style_function=lambda feature: {
        'fillColor': '#3186cc',
        'color': '#3186cc',
        'weight': 2,
        'fillOpacity': 0.15
    }
).add_to(infrastructure_map)

folium.GeoJson(
    gdf_streets,
    name="Street Network",
    style_function=lambda feature: {
        'color': '#ff7800',
        'weight': 0.8,
        'opacity': 0.6
    }
).add_to(infrastructure_map)

folium.LayerControl(position='topright').add_to(infrastructure_map)
infrastructure_map.save("network_map_austin.html")
