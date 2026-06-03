import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, HeatMapWithTime
from folium.features import DivIcon

from src.data_loader import (
    vancouver_crime,
    service_requests_311,
    street_network,
    local_area_boundary,
    property_parcels
)

#%%
gdf_streets = street_network.to_crs("EPSG:4326")
gdf_parcels = property_parcels.to_crs("EPSG:4326")
gdf_neighborhoods = local_area_boundary.to_crs("EPSG:4326")

#%%
"""
SIMPLE INTERACTIVE MAP
"""
df_vancouver = vancouver_crime.copy()
df_vancouver['YEAR'] = pd.to_numeric(df_vancouver['YEAR'], errors='coerce')

df_filtered = df_vancouver[df_vancouver['YEAR'] == 2025].copy()

df_clean = df_filtered[
    (df_filtered['X'].notnull()) &
    (df_filtered['Y'].notnull()) &
    (df_filtered['X'] != 0.0) &
    (df_filtered['Y'] != 0.0)
].copy()

gdf = gpd.GeoDataFrame(
    df_clean,
    geometry=gpd.points_from_xy(df_clean['X'], df_clean['Y']),
    crs="EPSG:26910"
).to_crs("EPSG:4326")

basic_map = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles='CartoDB positron')
cluster = MarkerCluster().add_to(basic_map)

for idx, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color="crimson",
        fill=True,
        fill_opacity=0.7,
        popup=f"<b>Type:</b> {row['TYPE']}<br><b>Block:</b> {row['HUNDRED_BLOCK']}"
    ).add_to(cluster)

# basic_map.save("basic_map.html")


#%%
"""
HEATMAP - KERNEL DENSITY ESTIMATION
"""
heat_map = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles='CartoDB positron')
coordinates = [[row.geometry.y, row.geometry.x] for index, row in gdf.iterrows()]

HeatMap(
    coordinates,
    radius=15,
    blur=20,
    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
).add_to(heat_map)

# heat_map.save("heat_map.html")


#%%
"""
TIME HEATMAP (JANUARY)
"""
df_time = df_vancouver.copy()

df_time['YEAR'] = pd.to_numeric(df_time['YEAR'], errors='coerce')
df_time['MONTH'] = pd.to_numeric(df_time['MONTH'], errors='coerce')
df_time['DAY'] = pd.to_numeric(df_time['DAY'], errors='coerce')

df_january = df_time[
    (df_time['YEAR'] == 2025) &
    (df_time['MONTH'] == 1)
].copy()

df_january = df_january[
    (df_january['X'].notnull()) &
    (df_january['Y'].notnull()) &
    (df_january['X'] != 0.0) &
    (df_january['Y'] != 0.0)
].copy()

time_map = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles='CartoDB positron')

days_in_month = sorted(df_january['DAY'].dropna().unique())

data_by_time = []
time_index = []

for day in days_in_month:
    df_day = df_january[df_january['DAY'] == day]

    gdf_day = gpd.GeoDataFrame(
        df_day,
        geometry=gpd.points_from_xy(df_day['X'], df_day['Y']),
        crs="EPSG:26910"
    ).to_crs("EPSG:4326")

    coords_day = [[row.geometry.y, row.geometry.x] for index, row in gdf_day.iterrows()]

    data_by_time.append(coords_day)
    time_index.append(f"January {int(day)}")

HeatMapWithTime(
    data=data_by_time,
    index=time_index,
    auto_play=True,
    radius=20,
    display_index=True
).add_to(time_map)

# time_map.save("time_map.html")


#%%
"""
UNIFIED CHOROPLETH AND POINTS MAP
"""
gdf_joined = gpd.sjoin(gdf, gdf_neighborhoods, how="inner", predicate="within")

neighborhood_col = 'name'
crime_counts = gdf_joined.groupby(neighborhood_col).size().reset_index()
crime_counts.columns = ['Neighborhood', 'Total_Crimes']

quantiles = crime_counts['Total_Crimes'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()

unified_map = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=gdf_neighborhoods,
    name='Neighborhood Density',
    data=crime_counts,
    columns=['Neighborhood', 'Total_Crimes'],
    key_on=f'feature.properties.{neighborhood_col}',
    fill_color='YlOrRd',
    fill_opacity=0.6,
    line_opacity=0.5,
    legend_name='Total Crimes (2025)',
    bins=quantiles
).add_to(unified_map)

gdf_merged = gdf_neighborhoods.merge(crime_counts, left_on=neighborhood_col, right_on='Neighborhood')

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
        color='#00ffff',
        fill=True,
        fill_opacity=0.7,
        weight=0,
        popup=f"Neighborhood: {row[neighborhood_col]}<br>Type: {row['TYPE']}"
    ).add_to(points_group)

folium.LayerControl().add_to(unified_map)
# unified_map.save("unified_map.html")


#%%
"""
INFRASTRUCTURE MAP (NEIGHBORHOODS, PARCELS & STREETS)
"""
infrastructure_map = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles='CartoDB positron')

folium.GeoJson(
    gdf_neighborhoods,
    name="Neighborhood Boundaries",
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

folium.GeoJson(
    gdf_parcels,
    name="Property Parcels",
    style_function=lambda feature: {
        'color': '#ffffff',
        'weight': 0.3,
        'fillOpacity': 0.05
    }
).add_to(infrastructure_map)

folium.LayerControl(position='topright').add_to(infrastructure_map)
# infrastructure_map.save("infrastructure_map.html")