import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import importlib

import src.data_loader as data_loader
import src.data_wrangling as data_wrangling

importlib.reload(data_loader)
importlib.reload(data_wrangling)

from src.data_wrangling import joined_311, joined_crime

# Parametros iniciales
RADII_METERS = [50, 100, 200]
PROOF_YEAR = 2019
PROOF_MONTH = 1
CRS_METRIC = "EPSG:32614"
CUTOFF_DATE = pd.to_datetime('2026-06-10')

# Preparacion de datos espaciales
gdf_disorder = joined_311[joined_311['ticket_type'] == 'disorder'].copy()
gdf_disorder['Created Date'] = pd.to_datetime(gdf_disorder['Created Date'])
gdf_disorder['Close Date'] = pd.to_datetime(gdf_disorder['Close Date']).fillna(CUTOFF_DATE)
gdf_disorder = gdf_disorder.to_crs(CRS_METRIC)

gdf_crime = joined_crime.copy().to_crs(CRS_METRIC)
gdf_crime['Occurred Date'] = pd.to_datetime(gdf_crime['Occurred Date'])

austin_area_m2 = gdf_disorder.unary_union.convex_hull.area

#%%

def get_ex_ante_tickets(gdf, year, month):
    # Definimos los limites del mes t
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)

    # Filtramos tickets que estuvieron abiertos en algun momento del mes t
    mask = (gdf['Created Date'] <= month_end) & (gdf['Close Date'] >= month_start)
    active = gdf[mask].copy()

    if active.empty:
        return active

    # Calculamos los dias activos EXCLUSIVAMENTE hasta el final del mes t
    start_calc = active['Created Date'].clip(lower=month_start)
    end_calc = active['Close Date'].clip(upper=month_end)

    active['active_days_month'] = (end_calc - start_calc).dt.days + 1

    # Clasificamos usando los umbrales del panel FE
    conds = [
        active['active_days_month'] <= 7,
        (active['active_days_month'] > 7) & (active['active_days_month'] <= 21),
        active['active_days_month'] > 21
    ]
    labels = ['1_7_days', '8_21_days', '21_plus_days']
    active['neglect_category'] = np.select(conds, labels, default='unknown')

    return active


def get_crimes_in_month(gdf, year, month):
    mask = (gdf['Occurred Date'].dt.year == year) & (gdf['Occurred Date'].dt.month == month)
    return gdf[mask].copy()


def calculate_ccr(tickets_gdf, crimes_gdf, radius, total_area):
    if tickets_gdf.empty or crimes_gdf.empty:
        return np.nan, 0, 0, 0

    buffers = tickets_gdf.copy()
    buffers['geometry'] = buffers.geometry.buffer(radius)
    dissolved = buffers[['geometry']].dissolve()

    buffer_area = dissolved.geometry.iloc[0].area
    area_ratio = buffer_area / total_area

    crimes_inside = gpd.sjoin(
        crimes_gdf[['geometry']],
        dissolved.reset_index()[['geometry']],
        how='inner',
        predicate='within'
    )

    n_inside = len(crimes_inside)
    n_total = len(crimes_gdf)
    crime_ratio = n_inside / n_total if n_total > 0 else 0

    ccr = crime_ratio / area_ratio if area_ratio > 0 else np.nan
    return ccr, n_inside, n_total, buffer_area


#%%
# Configuracion temporal
t_year, t_month = PROOF_YEAR, PROOF_MONTH
t1_year, t1_month = (t_year + 1, 1) if t_month == 12 else (t_year, t_month + 1)

# Extraccion de datos
tickets_t = get_ex_ante_tickets(gdf_disorder, t_year, t_month)
crimes_t1 = get_crimes_in_month(gdf_crime, t1_year, t1_month)
crimes_t = get_crimes_in_month(gdf_crime, t_year, t_month)

print(f"\nEvaluando Umbrales Ex-Ante: {t_year}-{t_month:02d} -> {t1_year}-{t1_month:02d}")
categories = ['1_7_days', '8_21_days', '21_plus_days']

results = []

for radius in RADII_METERS:
    for cat in categories:
        cat_tickets = tickets_t[tickets_t['neglect_category'] == cat]

        # Original: tickets en mes t vs crimen en mes t+1
        ccr_main, n_in_main, n_tot_main, area_main = calculate_ccr(
            cat_tickets, crimes_t1, radius, austin_area_m2
        )

        # Placebo: tickets en mes t vs crimen en mes t
        ccr_placebo, n_in_plac, _, _ = calculate_ccr(
            cat_tickets, crimes_t, radius, austin_area_m2
        )

        results.append({
            'radius_m': radius,
            'category': cat,
            'tickets_n': len(cat_tickets),
            'area_km2': area_main / 1e6,
            'CCR_original': ccr_main,
            'CCR_placebo': ccr_placebo
        })

df_results = pd.DataFrame(results)

print("\n--- CCR POR DURACION DE NEGLIGENCIA (EX-ANTE) ---")
print(
    df_results[['radius_m', 'category', 'tickets_n', 'area_km2', 'CCR_original', 'CCR_placebo']].to_string(index=False))

#%%

# Generacion de graficos
fig, ax = plt.subplots(figsize=(10, 6))

colors = {'1_7_days': '#69BE28', '8_21_days': '#A5ACAF', '21_plus_days': '#002244'}
labels_map = {'1_7_days': '1-7 dias', '8_21_days': '8-21 dias', '21_plus_days': '21+ dias'}

for cat in categories:
    sub = df_results[df_results['category'] == cat]

    ax.plot(sub['radius_m'], sub['CCR_original'], marker='o',
            color=colors[cat], label=f"{labels_map[cat]} (Original)", linewidth=2.5)

    ax.plot(sub['radius_m'], sub['CCR_placebo'], marker='s', linestyle='--',
            color=colors[cat], alpha=0.5, label=f"{labels_map[cat]} (Placebo)")

ax.axhline(1, color='tomato', linestyle=':', linewidth=2, label='Sin Concentracion (CCR=1)')

ax.set_title(f'Concentracion de Crimen por Duracion de Negligencia\nAustin {t_year}-{t_month:02d}', fontweight='bold')
ax.set_xlabel('Radio del Buffer (m)')
ax.set_ylabel('CCR (Crime Concentration Ratio)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
import os;

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/spatial_threshold_ex_ante.png', dpi=150)
plt.show()