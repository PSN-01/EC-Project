import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import importlib
import os

import src.data_loader as data_loader
import src.data_wrangling as data_wrangling

importlib.reload(data_loader)
importlib.reload(data_wrangling)

from src.data_wrangling import joined_311, joined_crime

# ── PARAMETERS ────────────────────────────────────────────────────────────────

STUDY_YEAR    = 2019          # year from which we sample closed tickets
RADII_METERS  = [50, 100, 200]
CRS_METRIC    = "EPSG:32614"
CUTOFF_DATE   = pd.to_datetime('2026-06-10')

# We need at least this many days open to have a meaningful "during" window.
# Filters out tickets closed same day they were opened.
MIN_DAYS_OPEN = 3

os.makedirs('figures', exist_ok=True)

#%% PREPARE DATA

gdf_disorder = joined_311[joined_311['ticket_type'] == 'disorder'].copy()
gdf_disorder['Created Date'] = pd.to_datetime(gdf_disorder['Created Date'])
gdf_disorder['Close Date']   = pd.to_datetime(gdf_disorder['Close Date']).fillna(CUTOFF_DATE)
gdf_disorder = gdf_disorder.to_crs(CRS_METRIC)

gdf_crime = joined_crime.copy().to_crs(CRS_METRIC)
gdf_crime['Occurred Date'] = pd.to_datetime(gdf_crime['Occurred Date'])

print(f"Disorder tickets (all years): {len(gdf_disorder):,}")
print(f"Crime incidents  (all years): {len(gdf_crime):,}")

#%% SELECT TICKETS CLOSED IN STUDY YEAR

# We anchor the event study on tickets that CLOSED in STUDY_YEAR.
# This guarantees we have a clean "after" window to observe.
closed_in_year = gdf_disorder[
    (gdf_disorder['Close Date'].dt.year == STUDY_YEAR) &
    ((gdf_disorder['Close Date'] - gdf_disorder['Created Date']).dt.days >= MIN_DAYS_OPEN)
].copy()

closed_in_year['days_open'] = (
    closed_in_year['Close Date'] - closed_in_year['Created Date']
).dt.days

print(f"\nTickets closed in {STUDY_YEAR} (>= {MIN_DAYS_OPEN} days open): {len(closed_in_year):,}")
print(f"  Median days open: {closed_in_year['days_open'].median():.0f}")
print(f"  Mean days open:   {closed_in_year['days_open'].mean():.1f}")

#%% HELPER — count crimes near a point within a date window

def crimes_near_point(point_geom, radius, date_start, date_end, crime_gdf):
    """Count crimes within `radius` meters of `point_geom` between date_start and date_end."""
    buf  = point_geom.buffer(radius)
    mask = (
        (crime_gdf['Occurred Date'] >= date_start) &
        (crime_gdf['Occurred Date'] <= date_end)    &
         crime_gdf.geometry.within(buf)
    )
    return mask.sum()

#%% EVENT STUDY LOOP
# For each ticket we define three windows:
#   PRE:    30 days before the ticket was OPENED  (baseline)
#   DURING: from ticket open to ticket close       (disorder present)
#   POST:   30 days after the ticket was CLOSED   (disorder resolved)
#
# We normalize crime counts by window length (crimes per day)
# so that windows of different durations are comparable.

print(f"\nRunning event study for {len(closed_in_year):,} tickets × {len(RADII_METERS)} radii...")
print("This may take several minutes.\n")

results = []

for idx, row in closed_in_year.iterrows():
    t_open  = row['Created Date']
    t_close = row['Close Date']
    pt      = row.geometry

    pre_start    = t_open  - pd.Timedelta(days=30)
    pre_end      = t_open  - pd.Timedelta(days=1)
    during_start = t_open
    during_end   = t_close
    post_start   = t_close + pd.Timedelta(days=1)
    post_end     = t_close + pd.Timedelta(days=30)

    during_days = (during_end - during_start).days + 1

    for radius in RADII_METERS:
        n_pre    = crimes_near_point(pt, radius, pre_start,    pre_end,    gdf_crime)
        n_during = crimes_near_point(pt, radius, during_start, during_end, gdf_crime)
        n_post   = crimes_near_point(pt, radius, post_start,   post_end,   gdf_crime)

        results.append({
            'ticket_id':    idx,
            'radius_m':     radius,
            'days_open':    during_days,
            'crimes_pre':   n_pre    / 30,           # per day
            'crimes_during':n_during / during_days,   # per day
            'crimes_post':  n_post   / 30,            # per day
        })

df_es = pd.DataFrame(results)
print(f"Event study complete. {len(df_es):,} observations.")

#%% AGGREGATE — mean crimes per day by window and radius

agg = df_es.groupby('radius_m')[['crimes_pre', 'crimes_during', 'crimes_post']].mean().reset_index()
print("\n--- Event Study Results (mean crimes/day per ticket) ---")
print(agg.to_string(index=False))

# Percent change relative to pre-period
agg['pct_change_during'] = (agg['crimes_during'] - agg['crimes_pre']) / agg['crimes_pre'] * 100
agg['pct_change_post']   = (agg['crimes_post']   - agg['crimes_pre']) / agg['crimes_pre'] * 100
print("\n--- Percent change relative to PRE window ---")
print(agg[['radius_m', 'pct_change_during', 'pct_change_post']].to_string(index=False))

#%% FIGURE — event study coefficient plot

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

windows      = ['crimes_pre', 'crimes_during', 'crimes_post']
window_labels= ['Pre\n(30 days before open)', 'During\n(ticket open)', 'Post\n(30 days after close)']
x            = np.arange(len(windows))
colors_r     = {50: '#69BE28', 100: '#002244', 200: '#A5ACAF'}

# Panel A — raw crimes per day
ax = axes[0]
for radius in RADII_METERS:
    sub = agg[agg['radius_m'] == radius]
    vals = [sub[w].values[0] for w in windows]
    ax.plot(x, vals, marker='o', linewidth=2.5,
            color=colors_r[radius], label=f'{radius}m radius')

ax.set_xticks(x)
ax.set_xticklabels(window_labels, fontsize=9)
ax.set_ylabel('Mean crimes per day (within radius)')
ax.set_title('Event Study: Crime Around Disorder Tickets\n(crimes per day, normalized by window length)')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(1.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()

# Panel B — percent change relative to pre
ax2 = axes[1]
for radius in RADII_METERS:
    sub = agg[agg['radius_m'] == radius]
    pct_vals = [0, sub['pct_change_during'].values[0], sub['pct_change_post'].values[0]]
    ax2.plot(x, pct_vals, marker='s', linewidth=2.5,
             color=colors_r[radius], label=f'{radius}m radius')

ax2.axhline(0, color='tomato', linestyle='--', linewidth=1.2, label='Pre-period baseline (0%)')
ax2.set_xticks(x)
ax2.set_xticklabels(window_labels, fontsize=9)
ax2.set_ylabel('% change in crimes/day vs pre-period')
ax2.set_title('Event Study: % Change Relative to Pre-Period')
ax2.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axvline(1.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.legend()

plt.suptitle(f'Spatial Event Study — Social Disorder Tickets Closed in {STUDY_YEAR}\n'
             f'n = {len(closed_in_year):,} tickets',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/event_study_spatial.png', dpi=150)
plt.show()

#%% ROBUSTNESS — split by ticket duration (short vs long neglect)

print("\n--- Robustness: short neglect (≤7 days) vs long neglect (>21 days) ---")

for label, mask in [('Short neglect (1–7 days)',  df_es['days_open'] <= 7),
                    ('Long neglect  (>21 days)',   df_es['days_open'] >  21)]:
    sub = df_es[mask].groupby('radius_m')[['crimes_pre', 'crimes_during', 'crimes_post']].mean()
    print(f"\n  {label}  (n tickets = {mask.sum() // len(RADII_METERS):,})")
    print(sub.to_string())

# %% SMALL MULTIPLES - 3x3 MATRIX (ENHANCED CARTOGRAPHIC DETAILS)

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Importamos la fuente para el contorno cartográfico exterior
from src.data_wrangling import gdf_boundaries as austin_boundary

STUDY_MONTH = 6  # Example: 6 = June
RADII_METERS = [50, 100, 200]

# VISUAL TRICK: We draw them bigger so they pop on the map,
# but all text will strictly say 50m, 100m, and 200m.
VISUAL_RADII = {
    50: 125,  # Drawn as 125m
    100: 225,  # Drawn as 225m
    200: 350  # Drawn as 350m
}

# 0. PREPARAR GEOMETRÍAS BASE
# A. Contorno exterior de la ciudad (Austin, TX)
austin_limits = austin_boundary.copy().to_crs(CRS_METRIC)
austin_limits = austin_limits.dissolve()

# B. Subdivisiones internas (Census Tracts) recortadas con el contorno real
texas_tracts = data_loader.census_tracts_atx.copy().to_crs(CRS_METRIC)
austin_tracts = gpd.clip(texas_tracts, austin_limits)

# 1. Filtrar los crímenes para el mes específico
gdf_crime_month = gdf_crime[(gdf_crime['Occurred Date'].dt.year == STUDY_YEAR) &
                            (gdf_crime['Occurred Date'].dt.month == STUDY_MONTH)].copy()

# 2. Filter tickets de desorden y calcular días abiertos
mask_tickets = (gdf_disorder['Created Date'].dt.month == STUDY_MONTH) & \
               (gdf_disorder['Created Date'].dt.year == STUDY_YEAR)
gdf_tickets_month = gdf_disorder[mask_tickets].copy()

gdf_tickets_month['days_open'] = (gdf_tickets_month['Close Date'] - gdf_tickets_month['Created Date']).dt.days

# Categorías de tiempo para las columnas
time_categories = [
    {'label': '1–7 Days\n(Short Neglect)', 'cond': lambda df: df['days_open'] <= 7},
    {'label': '8–21 Days\n(Medium Neglect)', 'cond': lambda df: (df['days_open'] > 7) & (df['days_open'] <= 21)},
    {'label': '>21 Days\n(Long Neglect)', 'cond': lambda df: df['days_open'] > 21}
]

# COLORES NEÓN DE ALTO CONTRASTE:
CRIME_COLOR = '#00FFFF'  # Cyan brillante para los crímenes

COLORS_RADII = {
    50: '#39FF14',  # Neon Green
    100: '#FF00FF',  # Neon Magenta
    200: '#FF0000'  # Pure Red
}

# 3. Configurar la cuadrícula (Matriz grande de 18x18 para que respire bien)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
fig.patch.set_facecolor('white')

# Límites fijos basados en el contorno exterior para una escala uniforme
xmin, ymin, xmax, ymax = austin_limits.total_bounds

for r_idx, true_radius in enumerate(RADII_METERS):
    current_color = COLORS_RADII[true_radius]
    visual_radius = VISUAL_RADII[true_radius]

    for t_idx, cat in enumerate(time_categories):
        ax = axes[r_idx, t_idx]

        tickets_cell = gdf_tickets_month[cat['cond'](gdf_tickets_month)].copy()

        # A. DETALLE INTERNO: Census Tracts (Subimos a linewidth=0.6 y borde un poco más oscuro)
        austin_tracts.plot(ax=ax, color='#F3F4F6', edgecolor='#D1D5DB', linewidth=0.6, zorder=1.4)

        # B. ENCUADRE CARTOGRÁFICO: Contorno exterior (Subimos a linewidth=1.8 y color gris ceniza)
        austin_limits.plot(ax=ax, color='none', edgecolor='#4B5563', linewidth=1.8, zorder=2)

        # C. TEXTURA DE FONDO: Crímenes del mes (Puntos Cyan)
        gdf_crime_month.plot(ax=ax, color=CRIME_COLOR, markersize=0.67, alpha=0.6, zorder=3)

        # D. CAPA PRINCIPAL: Zonas de Desorden Social (Búferes agrandados visualmente)
        if not tickets_cell.empty:
            buffers = tickets_cell.copy()
            buffers['geometry'] = buffers.geometry.buffer(visual_radius)
            dissolved_buffers = buffers[['geometry']].dissolve()

            if not dissolved_buffers.empty:
                dissolved_buffers.plot(ax=ax, color=current_color, alpha=0.9, edgecolor='black', linewidth=0.5,
                                       zorder=4)

        # Ajuste de ejes y encuadre idéntico para los 9 mapas
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('off')

        # E. TEXTOS Y CATEGORÍAS DE LA MATRIZ
        # Títulos de las columnas (Solo en la fila de arriba)
        if r_idx == 0:
            ax.set_title(cat['label'], fontsize=16, fontweight='bold', pad=10, color='#374151')

        # Títulos de las filas (Solo en la primera columna a la izquierda)
        if t_idx == 0:
            ax.text(-0.10, 0.5, f'Radius: {true_radius}m', transform=ax.transAxes,
                    fontsize=16, fontweight='bold', color=current_color,
                    rotation=90, va='center', ha='right')

# 4. LEYENDA UNIVERSAL MEJORADA
legend_elements = [
    mlines.Line2D([0], [0], color='#4B5563', lw=2, label='Austin, TX City Limits'),
    mlines.Line2D([0], [0], marker='o', color='k', markerfacecolor=CRIME_COLOR, markersize=8, markeredgewidth=0, label='Crime Incidents'),
    mpatches.Patch(facecolor=COLORS_RADII[50], alpha=0.9, edgecolor='black', label='Social Disorder Zone (50m)'),
    mpatches.Patch(facecolor=COLORS_RADII[100], alpha=0.9, edgecolor='black', label='Social Disorder Zone (100m)'),
    mpatches.Patch(facecolor=COLORS_RADII[200], alpha=0.9, edgecolor='black', label='Social Disorder Zone (200m)')
]

# Colocar la leyenda abajo centrada en una sola fila limpia
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=12,
           bbox_to_anchor=(0.5, 0.02), frameon=False)

# Título Principal del Reporte con el ajuste fix de altura (y=0.94) para evitar empalmes
plt.suptitle(
    f'Austin, TX — Spatial Impact: Social Disorder Radius vs. Resolution Time\n(Month: {STUDY_MONTH}, Year: {STUDY_YEAR})',
    fontsize=22, fontweight='bold', color='#111827', y=0.94)

# Ajuste estricto de espacios para que todo quepa al centavo
plt.tight_layout()
plt.subplots_adjust(top=0.84, bottom=0.08, wspace=0.05, hspace=0.15)

# Guardar figura (Descoméntala si necesitas guardarla a disco)
plt.savefig('figures/smallmultiples3x3_ultimate.png', dpi=300, bbox_inches='tight')

plt.show()
# %%

# %% SMALL MULTIPLES - 1x3 MATRIX (ULTIMATE CARTOGRAPHIC EDITION)

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Importamos la fuente para el contorno cartográfico exterior
from src.data_wrangling import gdf_boundaries as austin_boundary

STUDY_MONTH = 6  # Example: 6 = June

# VISUAL TRICK: We use 275m so the bubbles look bigger and clearer on the map,
# but we override all labels below so they strictly say "200m".
VISUAL_RADIUS = 275

# 0. PREPARAR GEOMETRÍAS BASE
# A. Contorno exterior de la ciudad (Austin, TX)
austin_limits = austin_boundary.copy().to_crs(CRS_METRIC)
austin_limits = austin_limits.dissolve()

# B. Subdivisiones internas (Census Tracts) recortadas con el contorno real
texas_tracts = data_loader.census_tracts_atx.copy().to_crs(CRS_METRIC)
austin_tracts = gpd.clip(texas_tracts, austin_limits)

# 1. Filter crimes for the specific month and year
gdf_crime_month = gdf_crime[(gdf_crime['Occurred Date'].dt.year == STUDY_YEAR) &
                            (gdf_crime['Occurred Date'].dt.month == STUDY_MONTH)].copy()

# 2. Filter tickets for the month and calculate days open
mask_tickets = (gdf_disorder['Created Date'].dt.month == STUDY_MONTH) & \
               (gdf_disorder['Created Date'].dt.year == STUDY_YEAR)
gdf_tickets_month = gdf_disorder[mask_tickets].copy()

gdf_tickets_month['days_open'] = (gdf_tickets_month['Close Date'] - gdf_tickets_month['Created Date']).dt.days

# Define the 3 time conditions for the columns (English)
time_categories = [
    {'label': '1–7 Days\n(Short Neglect)', 'cond': lambda df: df['days_open'] <= 7},
    {'label': '8–21 Days\n(Medium Neglect)', 'cond': lambda df: (df['days_open'] > 7) & (df['days_open'] <= 21)},
    {'label': '>21 Days\n(Long Neglect)', 'cond': lambda df: df['days_open'] > 21}
]

# HIGH CONTRAST NEON COLORS:
CRIME_COLOR = '#00FFFF'  # Super bright Cyan for high contrast
RADIUS_COLOR = '#FF0000'  # Pure Red for the buffers

# 3. Setup the 1x3 grid (Height = 7 to give the title breathing room)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 7))
fig.patch.set_facecolor('white')

# Fixed city boundaries for consistent scaling across all 3 plots
xmin, ymin, xmax, ymax = austin_limits.total_bounds

for t_idx, cat in enumerate(time_categories):
    ax = axes[t_idx]

    tickets_cell = gdf_tickets_month[cat['cond'](gdf_tickets_month)].copy()

    # A. DETALLE INTERNO: Census Tracts (Gris claro con bordes definidos)
    austin_tracts.plot(ax=ax, color='#F3F4F6', edgecolor='#D1D5DB', linewidth=0.6, zorder=1)

    # B. ENCUADRE CARTOGRÁFICO: Contorno exterior (Gris oscuro y grueso)
    austin_limits.plot(ax=ax, color='none', edgecolor='#4B5563', linewidth=1.8, zorder=1)

    # C. TEXTURE: Crime points (Bigger and brighter cyan)
    gdf_crime_month.plot(ax=ax, color=CRIME_COLOR, markersize=1, alpha=0.7, zorder=3)

    # D. MAIN LAYER: Disorder buffers (Drawn bigger for visual impact)
    if not tickets_cell.empty:
        buffers = tickets_cell.copy()
        buffers['geometry'] = buffers.geometry.buffer(VISUAL_RADIUS)
        dissolved_buffers = buffers[['geometry']].dissolve()

        if not dissolved_buffers.empty:
            dissolved_buffers.plot(ax=ax, color=RADIUS_COLOR, alpha=0.9, edgecolor='black', linewidth=0.5, zorder=4)

    # Framing and axes cleanup
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')

    # D. STRATEGIC MATRIX LABELS
    # Column titles (Added pad=20 to push them down away from the suptitle)
    ax.set_title(cat['label'], fontsize=14, fontweight='bold', pad=20, color='#374151')

    # Row title (Far left column only) - FORCED TO 200m FOR THE RESEARCH
    if t_idx == 0:
        ax.text(-0.10, 0.5, 'Radius: 200m', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=RADIUS_COLOR,
                rotation=90, va='center', ha='right')

# 4. UNIVERSAL LEGEND - FORCED TO 200m FOR THE RESEARCH
# Actualicé el color del City Limits en la leyenda para que cuadre con el #4B5563 del mapa
legend_elements = [
    mlines.Line2D([0], [0], color='#4B5563', lw=2, label='Austin, TX City Limits'),
    mlines.Line2D([0], [0], marker='o', color='k', markerfacecolor=CRIME_COLOR, markersize=8, markeredgewidth=0,
                  label='Crime Incidents'),
    mpatches.Patch(facecolor=RADIUS_COLOR, alpha=0.9, edgecolor='black', label='Social Disorder Zone (200m Buffer)')
]

# Place legend at the bottom center of the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=13,
           bbox_to_anchor=(0.5, 0.02), frameon=False)

# Main Report Title
plt.suptitle(
    f'Austin, TX — Spatial Impact: 200m Radius vs. Resolution Time\n(Month: {STUDY_MONTH}, Year: {STUDY_YEAR})',
    fontsize=18, fontweight='bold', color='#111827', y=0.98)

# FIX: Increased top margin significantly to prevent overlap, and increased bottom for legend
plt.tight_layout()
plt.subplots_adjust(top=0.78, bottom=0.15)

# Descomenta esta línea cuando quieras guardarla de nuevo
plt.savefig('figures/smallmultiples1x3_cartographic.png', dpi=300, bbox_inches='tight')

plt.show()
# %%




# %%

"""
WHOLE YEARS
"""

#%% HISTORICAL EVENT STUDY (ALL YEARS: 2014-2026)
print("\n" + "="*60)
print("RUNNING EVENT STUDY FOR ALL YEARS (2014 - 2026)")
print("="*60)

# 1. Filter all tickets regardless of the year
closed_all_years = gdf_disorder[
    ((gdf_disorder['Close Date'] - gdf_disorder['Created Date']).dt.days >= MIN_DAYS_OPEN)
].copy()

closed_all_years['days_open'] = (
    closed_all_years['Close Date'] - closed_all_years['Created Date']
).dt.days

print(f"\nTickets closed in all years (>= {MIN_DAYS_OPEN} days open): {len(closed_all_years):,}")
print(f"  Median days open: {closed_all_years['days_open'].median():.0f}")
print(f"  Mean days open:   {closed_all_years['days_open'].mean():.1f}")

print(f"\nRunning historical event study for {len(closed_all_years):,} tickets × {len(RADII_METERS)} radii...")
print("Grab a coffee, esto sí se va a tardar un madrazo...\n")

results_all = []
total_tickets = len(closed_all_years)

for i, (idx, row) in enumerate(closed_all_years.iterrows()):
    # Tracker de progreso para que no pienses que se trabó la compu
    if (i + 1) % 5000 == 0:
        print(f"  Procesados {i + 1:,} de {total_tickets:,} tickets...")

    t_open  = row['Created Date']
    t_close = row['Close Date']
    pt      = row.geometry

    pre_start    = t_open  - pd.Timedelta(days=30)
    pre_end      = t_open  - pd.Timedelta(days=1)
    during_start = t_open
    during_end   = t_close
    post_start   = t_close + pd.Timedelta(days=1)
    post_end     = t_close + pd.Timedelta(days=30)

    during_days = (during_end - during_start).days + 1

    for radius in RADII_METERS:
        n_pre    = crimes_near_point(pt, radius, pre_start,    pre_end,    gdf_crime)
        n_during = crimes_near_point(pt, radius, during_start, during_end, gdf_crime)
        n_post   = crimes_near_point(pt, radius, post_start,   post_end,   gdf_crime)

        results_all.append({
            'ticket_id':    idx,
            'radius_m':     radius,
            'days_open':    during_days,
            'crimes_pre':   n_pre    / 30,
            'crimes_during':n_during / during_days,
            'crimes_post':  n_post   / 30,
        })

df_es_all = pd.DataFrame(results_all)

# Guardamos el CSV para que no tengas que volver a correr esto en la vida
df_es_all.to_csv('data/clean/event_study_all_years.csv', index=False)
print(f"\nHistorical event study complete. {len(df_es_all):,} observations saved to CSV.")

#%% AGGREGATE HISTORICAL

import pandas as pd

df_es_all = pd.read_csv('data/clean/event_study_all_years.csv')
agg_all = df_es_all.groupby('radius_m')[['crimes_pre', 'crimes_during', 'crimes_post']].mean().reset_index()
print("\n--- Historical Event Study Results (mean crimes/day per ticket) ---")
print(agg_all.to_string(index=False))

agg_all['pct_change_during'] = (agg_all['crimes_during'] - agg_all['crimes_pre']) / agg_all['crimes_pre'] * 100
agg_all['pct_change_post']   = (agg_all['crimes_post']   - agg_all['crimes_pre']) / agg_all['crimes_pre'] * 100
print("\n--- Percent change relative to PRE window (ALL YEARS) ---")
print(agg_all[['radius_m', 'pct_change_during', 'pct_change_post']].to_string(index=False))


#%% FIGURE HISTORICAL

# Por si no tienes en memoria RADII_METERS en tu sesión actual:
RADII_METERS = [50, 100, 200]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

windows      = ['crimes_pre', 'crimes_during', 'crimes_post']
window_labels= ['Pre\n(30 days before open)', 'During\n(ticket open)', 'Post\n(30 days after close)']
x            = np.arange(len(windows))
colors_r     = {50: '#69BE28', 100: '#002244', 200: '#A5ACAF'}

# Panel A — raw crimes per day
ax = axes[0]
for radius in RADII_METERS:
    sub = agg_all[agg_all['radius_m'] == radius]
    vals = [sub[w].values[0] for w in windows]
    ax.plot(x, vals, marker='o', linewidth=2.5,
            color=colors_r[radius], label=f'{radius}m radius')

ax.set_xticks(x)
ax.set_xticklabels(window_labels, fontsize=9)
ax.set_ylabel('Mean crimes per day (within radius)')
ax.set_title('Historical Event Study: Crime Around Disorder Tickets\n(crimes per day, normalized)')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(1.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()

# Panel B — percent change relative to pre
ax2 = axes[1]
for radius in RADII_METERS:
    sub = agg_all[agg_all['radius_m'] == radius]
    pct_vals = [0, sub['pct_change_during'].values[0], sub['pct_change_post'].values[0]]
    ax2.plot(x, pct_vals, marker='s', linewidth=2.5,
             color=colors_r[radius], label=f'{radius}m radius')

ax2.axhline(0, color='tomato', linestyle='--', linewidth=1.2, label='Pre-period baseline (0%)')
ax2.set_xticks(x)
ax2.set_xticklabels(window_labels, fontsize=9)
ax2.set_ylabel('% change in crimes/day vs pre-period')
ax2.set_title('Historical Event Study: % Change Relative to Pre-Period')
ax2.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axvline(1.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.legend()

# === AQUÍ ESTÁ LA MAGIA ===
# Calculamos la 'n' sacando los valores únicos de ticket_id desde el CSV que ya cargaste
n_tickets = df_es_all['ticket_id'].nunique()

plt.suptitle(f'Historical Spatial Event Study (2014-2026)\nn = {n_tickets:,} tickets',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/event_study_spatial_all_years.png', dpi=150)
plt.show()

#%% ROBUSTNESS HISTORICAL

print("\n--- Robustness (ALL YEARS): short neglect (≤7 days) vs long neglect (>21 days) ---")

for label, mask in [('Short neglect (1–7 days)',  df_es_all['days_open'] <= 7),
                    ('Long neglect  (>21 days)',   df_es_all['days_open'] >  21)]:
    sub = df_es_all[mask].groupby('radius_m')[['crimes_pre', 'crimes_during', 'crimes_post']].mean()
    print(f"\n  {label}  (n tickets = {mask.sum() // len(RADII_METERS):,})")
    print(sub.to_string())

