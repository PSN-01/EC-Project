import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import importlib
import os

import src.data_loader as data_loader
import src.data_wrangling as data_wrangling

importlib.reload(data_loader)
importlib.reload(data_wrangling)

from src.data_wrangling import joined_311, joined_crime

# parametros iniciales
CRS_METRIC = "EPSG:32614"
CUTOFF_DATE = pd.to_datetime('2026-06-10')
CELL_SIZE_METERS = 200

os.makedirs('data/clean', exist_ok=True)
os.makedirs('figures', exist_ok=True)

#%%

# preparacion de datos
gdf_disorder = joined_311[joined_311['ticket_type'] == 'disorder'].copy()
gdf_disorder['Created Date'] = pd.to_datetime(gdf_disorder['Created Date'])
gdf_disorder['Close Date'] = pd.to_datetime(gdf_disorder['Close Date']).fillna(CUTOFF_DATE)
gdf_disorder = gdf_disorder.to_crs(CRS_METRIC)

gdf_crime = joined_crime.copy().to_crs(CRS_METRIC)
gdf_crime['Occurred Date'] = pd.to_datetime(gdf_crime['Occurred Date'])

print(f"\n[1] Construyendo la cuadricula espacial de {CELL_SIZE_METERS}x{CELL_SIZE_METERS} metros...")

xmin, ymin, xmax, ymax = gdf_disorder.total_bounds
grid_cells = []

for x0 in np.arange(xmin, xmax, CELL_SIZE_METERS):
    for y0 in np.arange(ymin, ymax, CELL_SIZE_METERS):
        poly = Polygon([
            (x0, y0),
            (x0 + CELL_SIZE_METERS, y0),
            (x0 + CELL_SIZE_METERS, y0 + CELL_SIZE_METERS),
            (x0, y0 + CELL_SIZE_METERS)
        ])
        grid_cells.append(poly)

grid = gpd.GeoDataFrame(geometry=grid_cells, crs=CRS_METRIC)
grid['grid_id'] = grid.index

# filtramos los cuadritos para quedarnos solo con los que tocan austin
austin_hull = gpd.GeoDataFrame(geometry=[gdf_disorder.union_all().convex_hull], crs=CRS_METRIC)
grid = gpd.sjoin(grid, austin_hull, how='inner', predicate='intersects').drop(columns=['index_right'])

print(f"    Total de micro-zonas (cuadritos) creadas: {len(grid)}")

print("\n[2] Asignando tickets y crimenes a sus respectivos cuadritos...")

# Borramos la columna residual del wrangling anterior si es que existe
gdf_disorder = gdf_disorder.drop(columns=['index_right'], errors='ignore')
gdf_crime = gdf_crime.drop(columns=['index_right'], errors='ignore')

# Ahora si cruzamos sin pedos
disorder_grid = gpd.sjoin(gdf_disorder, grid[['grid_id', 'geometry']], how='inner', predicate='within')
crime_grid = gpd.sjoin(gdf_crime, grid[['grid_id', 'geometry']], how='inner', predicate='within')

print("\n[3] Construyendo el panel espacial historico mes a mes (2014 - 2026)... esto tomara un minuto.")

min_year = disorder_grid['Created Date'].dt.year.min()
max_year = disorder_grid['Created Date'].dt.year.max()

panel_rows = []

for year in range(min_year, max_year + 1):
    for month in range(1, 13):
        if year == 2026 and month >= 6:
            continue

        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)

        # calculo ex-ante de los tickets de este mes
        mask_311 = (disorder_grid['Created Date'] <= month_end) & (disorder_grid['Close Date'] >= month_start)
        active_t = disorder_grid[mask_311].copy()

        if not active_t.empty:
            start_calc = active_t['Created Date'].clip(lower=month_start)
            end_calc = active_t['Close Date'].clip(upper=month_end)
            active_t['active_days'] = (end_calc - start_calc).dt.days + 1

            grid_stats = active_t.groupby('grid_id').agg(
                Avg_Repair_Days=('active_days', 'mean'),
                ticket_count=('active_days', 'count')
            ).reset_index()
        else:
            grid_stats = pd.DataFrame(columns=['grid_id', 'Avg_Repair_Days', 'ticket_count'])

        # crímenes del mes t (placebo)
        mask_crime_t = (crime_grid['Occurred Date'].dt.year == year) & (crime_grid['Occurred Date'].dt.month == month)
        crime_t = crime_grid[mask_crime_t].groupby('grid_id').size().reset_index(name='Crime_t')

        # crímenes del mes t+1 (efecto real)
        t1_year = year + 1 if month == 12 else year
        t1_month = 1 if month == 12 else month + 1
        mask_crime_t1 = (crime_grid['Occurred Date'].dt.year == t1_year) & (
                    crime_grid['Occurred Date'].dt.month == t1_month)
        crime_t1 = crime_grid[mask_crime_t1].groupby('grid_id').size().reset_index(name='Crime_t1')

        # unimos todo para el mes actual
        month_df = pd.merge(grid_stats, crime_t, on='grid_id', how='outer')
        month_df = pd.merge(month_df, crime_t1, on='grid_id', how='outer')

        # solo guardamos las zonas que tuvieron ALGO de actividad (basura o crimen) para no inflar la memoria
        month_df = month_df.dropna(how='all', subset=['Avg_Repair_Days', 'Crime_t', 'Crime_t1'])

        month_df['year'] = year
        month_df['month'] = month
        month_df['date'] = month_start

        panel_rows.append(month_df)

micro_panel = pd.concat(panel_rows, ignore_index=True)
micro_panel = micro_panel.fillna(0)

# guardamos el panel limpio
micro_panel.to_csv('data/clean/micro_spatial_panel.csv', index=False)
print("\n[+] Panel micro-espacial guardado exitosamente en 'data/clean/micro_spatial_panel.csv'")
print(f"    Total de observaciones panel (cuadrito-mes): {len(micro_panel)}")

print("\n[4] Corriendo Regresion de Efectos Fijos Espaciales de Prueba...")

# preparamos el panel para la libreria linearmodels
df_model = micro_panel.copy()
df_model['date'] = pd.to_datetime(df_model['date'])
df_model = df_model.set_index(['grid_id', 'date'])

# separamos la negligencia en los umbrales para que sea igual a tu panel viejo
conds = [
    df_model['Avg_Repair_Days'] == 0,
    (df_model['Avg_Repair_Days'] > 0) & (df_model['Avg_Repair_Days'] <= 7),
    (df_model['Avg_Repair_Days'] > 7) & (df_model['Avg_Repair_Days'] <= 21),
    df_model['Avg_Repair_Days'] > 21
]
labels = ['0_Zero', '1_Low_1_7', '2_Med_8_21', '3_High_21plus']
df_model['Negligence_Level'] = np.select(conds, labels, default='0_Zero')

dummies = pd.get_dummies(df_model['Negligence_Level'], drop_first=True).astype(float)
dummies = dummies[['1_Low_1_7', '2_Med_8_21', '3_High_21plus']]

# definimos las variables (Lagged para aislar causalidad como en tu panel original)
exog = sm.add_constant(dummies)
endog = df_model['Crime_t1']

# corremos el modelo con efectos fijos (absorbiendo las caracteristicas estaticas de cada cuadrito y el tiempo)
try:
    results_fe = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type='clustered', cluster_entity=True)
    print("\n================ RESULTADOS DEL PANEL ESPACIAL EXACTO ================")
    print(results_fe.summary)
except Exception as e:
    print(f"\nError al correr la regresion: {e}")

# dibujito rapido para ver el mapa
fig, ax = plt.subplots(figsize=(10, 10))
austin_hull.plot(ax=ax, color='white', edgecolor='black', linewidth=1)
grid.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.3, linewidth=0.2)
gdf_disorder.sample(min(10000, len(gdf_disorder))).plot(ax=ax, color='red', markersize=0.5, alpha=0.5)
ax.set_title("Malla Espacial de 200x200m sobre Puntos de Desorden", fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('figures/micro_grid_map.png', dpi=150)