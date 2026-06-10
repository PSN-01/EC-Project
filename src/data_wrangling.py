import pandas as pd
import geopandas as gpd

from src.data_loader import (
    austin_crime,
    census_tracts_atx,
    street_centerline_atx,
    austin_311_public,
    jurisdictions_atx,
    austin_demo_data
)

# Census Tracts and Austin Filter
gdf_boundaries = census_tracts_atx.to_crs("EPSG:4326")
gdf_jurisdiction = jurisdictions_atx.to_crs("EPSG:4326")
boundary_col = 'geoid'

# Keep Austin Tracts
gdf_boundaries = gpd.sjoin(
    gdf_boundaries,
    gdf_jurisdiction[['geometry']],
    how='inner',
    predicate='intersects'
).drop_duplicates(subset=[boundary_col])


"""
MERGE CENSUS DEMOGRAPHICS
"""

gdf_boundaries[boundary_col] = gdf_boundaries[boundary_col].astype(str)
austin_demo_data['GEOID'] = austin_demo_data['GEOID'].astype(str)

austin_valid_geoids = gdf_boundaries[boundary_col].unique().tolist()
demo_data = austin_demo_data[austin_demo_data['GEOID'].isin(austin_valid_geoids)].copy()

"""
FILTERING RELEVANT CATEGORIES (PRIMERO)
"""

# 311 Filters
bwt_311_categories = [
    'Graffiti Abatement',
    'APH - Graffiti Abatement - Public Property',
    'DSD - Graffiti Abatement - Private Property',
    'Street Light Issue- Address',
    'Street Light Issue- Multiple poles/multiple streets',
    'AE Street Light Issue - Address',
    'AE Street Light Issue - No Address',
    'Pothole Repair',
    'SBO - Pothole Repair',
    'TPW - Pothole Repair',
    'Debris in Street',
    'SBO - Debris in Street',
    'TPW - Debris in Street',
    'APD - Vehicle Abatement Report',
    'Sign - Traffic Sign Maintenance',
    'Sidewalk Repair',
    'SBO - Sidewalk Repair',
    'TPW - Sidewalk Repair'
]

df_311 = austin_311_public.copy()
df_311 = df_311[df_311['SR Description'].isin(bwt_311_categories)].copy()

# Crime Filters
bwt_crime_categories = [
    'BURGLARY OF VEHICLE', 'BURGLARY NON RESIDENCE', 'BURGLARY OF RESIDENCE',
    'CRIMINAL MISCHIEF', 'THEFT', 'AUTO THEFT', 'THEFT OF AUTO PARTS',
    'THEFT OF BICYCLE', 'GRAFFITI', 'CRIMINAL TRESPASS', 'CRIMINAL TRESPASS/TRANSIENT',
    'THEFT CATALYTIC CONVERTER', 'THEFT OF METAL', 'ARSON', 'DAMAGE CITY PROP',
    'ROBBERY BY ASSAULT', 'ROBBERY BY THREAT', 'AGG ROBBERY/DEADLY WEAPON',
    'AGG ASSAULT', 'ASSAULT WITH INJURY', 'ASSAULT BY THREAT', 'ASSAULT BY CONTACT',
    'DEADLY CONDUCT', 'PURSE SNATCHING', 'POCKET PICKING',
    'PROSTITUTION', 'DOC FIGHTING', 'URINATING IN PUBLIC PLACE',
    'VOCO - ALCOHOL CONSUMPTION', 'PUBLIC LEWDNESS', 'CAMPING IN PARK',
    'POSS CONTROLLED SUB/NARCOTIC', 'POSSESSION OF MARIJUANA', 'POSS OF DRUG PARAPHERNALIA',
    'DOC UNREASONABLE NOISE', 'PROWLER', 'LOITERING IN PUBLIC PARK'
]

df_crime = austin_crime.copy()
df_crime = df_crime[df_crime['Highest Offense Description'].isin(bwt_crime_categories)].copy()

"""
DATE FILTER (DESPUÉS DEL CATEGORY FILTER)
"""

# Crime date
df_crime['Occurred Date'] = pd.to_datetime(df_crime['Occurred Date'], errors='coerce')
df_crime = df_crime[
    (df_crime['Occurred Date'].dt.year >= 2014) &
    (df_crime['Occurred Date'].dt.year <= 2026)
].copy()

# 311 dates
df_311['Created Date'] = pd.to_datetime(df_311['Created Date'], errors='coerce')
df_311['Close Date'] = pd.to_datetime(df_311['Close Date'], errors='coerce')

df_311 = df_311[
    (df_311['Created Date'].dt.year >= 2014) &
    (df_311['Created Date'].dt.year <= 2026)
].copy()

"""
CLEANING + GEO (DESPUÉS)
"""

# Crime coords
df_crime['Latitude'] = pd.to_numeric(df_crime['Latitude'], errors='coerce')
df_crime['Longitude'] = pd.to_numeric(df_crime['Longitude'], errors='coerce')
df_crime = df_crime.dropna(subset=['Latitude', 'Longitude'])

gdf_crime = gpd.GeoDataFrame(
    df_crime,
    geometry=gpd.points_from_xy(df_crime['Longitude'], df_crime['Latitude']),
    crs="EPSG:4326"
)

# 311 coords
df_311['Latitude Coordinate'] = pd.to_numeric(df_311['Latitude Coordinate'], errors='coerce')
df_311['Longitude Coordinate'] = pd.to_numeric(df_311['Longitude Coordinate'], errors='coerce')
df_311 = df_311.dropna(subset=['Latitude Coordinate', 'Longitude Coordinate'])

gdf_311 = gpd.GeoDataFrame(
    df_311,
    geometry=gpd.points_from_xy(df_311['Longitude Coordinate'], df_311['Latitude Coordinate']),
    crs="EPSG:4326"
)

"""
SPATIAL JOIN (AL FINAL)
"""

joined_crime = gpd.sjoin(
    gdf_crime,
    gdf_boundaries[[boundary_col, 'geometry']],
    how="inner",
    predicate="within"
)

joined_311 = gpd.sjoin(
    gdf_311,
    gdf_boundaries[[boundary_col, 'geometry']],
    how="inner",
    predicate="within"
)