#%%

import importlib
import src.data_loader as data_loader
import src.data_wrangling as data_wrangling

importlib.reload(data_loader)
importlib.reload(data_wrangling)

import folium
import mapclassify
from src.data_wrangling import (
    joined_crime,
    joined_311,
    gdf_boundaries,
    boundary_col,
    demo_data
)

gdf_boundaries['area_km2'] = gdf_boundaries.to_crs(epsg=2277).area / 10**6

#%%
"""
CRIME MAP: DENSITY WITH NATURAL BREAKS (FISHER-JENKS)
"""
crime_counts = joined_crime.groupby(boundary_col).size().reset_index()
crime_counts.columns = [boundary_col, 'Total_Crimes']
gdf_crime_analysis = gdf_boundaries[[boundary_col, 'geometry', 'area_km2']].merge(
    crime_counts, on=boundary_col, how='left'
).fillna({'Total_Crimes': 0})
gdf_crime_analysis['Crime_Density'] = gdf_crime_analysis['Total_Crimes'] / gdf_crime_analysis['area_km2']

classifier_crime = mapclassify.NaturalBreaks(gdf_crime_analysis['Crime_Density'], k=5)
jenks_bins_crime = [gdf_crime_analysis['Crime_Density'].min()] + classifier_crime.bins.tolist()

crime_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')
folium.Choropleth(
    geo_data=gdf_boundaries[[boundary_col, 'geometry']],
    name='Crime Density per Km² (Natural Breaks)',
    data=gdf_crime_analysis,
    columns=[boundary_col, 'Crime_Density'],
    key_on=f'feature.properties.{boundary_col}',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.4,
    legend_name='Crimes per Sq Kilometer',
    bins=jenks_bins_crime
).add_to(crime_map)

folium.LayerControl().add_to(crime_map)
crime_map.save("maps/crime_map_austin.html")

#%%
"""
311 REPORTS MAP: DENSITY WITH NATURAL BREAKS (FISHER-JENKS)
"""
sr_counts = joined_311.groupby(boundary_col).size().reset_index()
sr_counts.columns = [boundary_col, 'Total_Reports']
gdf_sr_analysis = gdf_boundaries[[boundary_col, 'geometry', 'area_km2']].merge(
    sr_counts, on=boundary_col, how='left'
).fillna({'Total_Reports': 0})
gdf_sr_analysis['SR_Density'] = gdf_sr_analysis['Total_Reports'] / gdf_sr_analysis['area_km2']

classifier_sr = mapclassify.NaturalBreaks(gdf_sr_analysis['SR_Density'], k=5)
jenks_bins_sr = [gdf_sr_analysis['SR_Density'].min()] + classifier_sr.bins.tolist()

sr_map = folium.Map(location=[30.2672, -97.7431], zoom_start=11, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=gdf_boundaries[[boundary_col, 'geometry']],
    name='311 Report Density per Km² (Natural Breaks)',
    data=gdf_sr_analysis,
    columns=[boundary_col, 'SR_Density'],
    key_on=f'feature.properties.{boundary_col}',
    fill_color='PuBu',
    fill_opacity=0.6,
    line_opacity=0.5,
    legend_name='311 Reports per Sq Kilometer',
    bins=jenks_bins_sr
).add_to(sr_map)

folium.LayerControl().add_to(sr_map)
sr_map.save("maps/311_reports_map_austin.html")