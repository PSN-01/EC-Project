import panel_builder as pb
import panel_regressions_1st as p
import src.data_wrangling as dw
import src.data_loader as dl
import event_study as es
import pandas as pd

#%%

from src.data_loader import austin_crime
from src.data_loader import  austin_311_public

#%%


import src.data_loader as dl
import src.data_wrangling as dw

print("--- CALCULANDO LOS DATOS FALTANTES [XX] PARA EL PAPER ---\n")

# 1. Property tax parcels / Demographic units (raw)
# Son los datos que sacaste de la API del Censo
raw_demo_units = len(dl.austin_demo_data)
print(f"Property tax parcels / Demographic units (raw): {raw_demo_units:,}")

# 2. State of Texas census tracts (raw)
# Es el shapefile original completo de Texas antes de hacer el clip con Austin
raw_census_tracts = len(dl.census_tracts_atx)
print(f"State of Texas census tracts (raw): {raw_census_tracts:,}")

# 3. 311 requests matched to BWT/Infrastructure categories
# Tickets filtrados por fecha (2014-2026), categoría y con coordenadas válidas
valid_311_requests = len(dw.gdf_311)
# Tickets que además cayeron estrictamente dentro de los polígonos de Austin
joined_311_requests = len(dw.joined_311)

print(f"\n311 requests matched (con coordenadas válidas): {valid_311_requests:,}")
print(f"311 requests matched (dentro de Austin - joined): {joined_311_requests:,}")
print("-> Usa el de 'dentro de Austin' si quieres ser más estricto en la tabla.")

# 4. BWT-relevant crime incidents with valid coordinates
# Crímenes filtrados por fecha (2014-2026), categoría y con coordenadas válidas
valid_crimes = len(dw.gdf_crime)
# Crímenes que además cayeron estrictamente dentro de los polígonos de Austin
joined_crimes = len(dw.joined_crime)

print(f"\nBWT-relevant crime incidents (con coordenadas válidas): {valid_crimes:,}")
print(f"BWT-relevant crime incidents (dentro de Austin - joined): {joined_crimes:,}")
print("-> Igual, te recomiendo usar el 'joined' para la tabla final.")