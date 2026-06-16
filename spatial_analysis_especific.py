import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib
import os

import src.data_loader as data_loader
import src.data_wrangling as data_wrangling

importlib.reload(data_loader)
importlib.reload(data_wrangling)

from src.data_wrangling import joined_311, joined_crime

# ── PARAMETERS ────────────────────────────────────────────────────────────────

RADII_METERS  = [50, 100, 200]
CRS_METRIC    = "EPSG:32614"
CUTOFF_DATE   = pd.to_datetime('2026-06-10')
SNAPSHOT_YEAR = 2019   # year used for the map and single-year CCR table
SNAPSHOT_MONTH = 1     # month t for the map (January 2019 → February 2019)
MAP_RADIUS    = 100    # radius used in the map figure

os.makedirs('data/clean', exist_ok=True)
os.makedirs('figures', exist_ok=True)

#%% PREPARE DATA

gdf_disorder = joined_311[joined_311['ticket_type'] == 'disorder'].copy()
gdf_disorder['Created Date'] = pd.to_datetime(gdf_disorder['Created Date'])
gdf_disorder['Close Date']   = pd.to_datetime(gdf_disorder['Close Date']).fillna(CUTOFF_DATE)
gdf_disorder = gdf_disorder.to_crs(CRS_METRIC)

gdf_crime = joined_crime.copy().to_crs(CRS_METRIC)
gdf_crime['Occurred Date'] = pd.to_datetime(gdf_crime['Occurred Date'])

austin_area_m2 = gdf_disorder.union_all().convex_hull.area

print(f"Disorder tickets (all years): {len(gdf_disorder):,}")
print(f"Crime incidents  (all years): {len(gdf_crime):,}")

#%% HELPER FUNCTIONS

def get_ex_ante_tickets(gdf, year, month):
    """Active disorder tickets in month t, classified by days open so far (ex-ante)."""
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end   = month_start + pd.offsets.MonthEnd(0)

    mask   = (gdf['Created Date'] <= month_end) & (gdf['Close Date'] >= month_start)
    active = gdf[mask].copy()
    if active.empty:
        return active

    start_calc = active['Created Date'].clip(lower=month_start)
    end_calc   = active['Close Date'].clip(upper=month_end)
    active['active_days_month'] = (end_calc - start_calc).dt.days + 1

    conds  = [active['active_days_month'] <= 7,
              (active['active_days_month'] > 7) & (active['active_days_month'] <= 21),
               active['active_days_month'] > 21]
    labels = ['1_7_days', '8_21_days', '21_plus_days']
    active['neglect_category'] = np.select(conds, labels, default='unknown')
    return active


def get_crimes_in_month(gdf, year, month):
    mask = (gdf['Occurred Date'].dt.year == year) & (gdf['Occurred Date'].dt.month == month)
    return gdf[mask].copy()


def next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)


def compute_ccr(tickets_gdf, crimes_gdf, radius, total_area):
    """Returns (CCR, n_inside, n_total, buffer_area_m2). Returns NaN if inputs empty."""
    if tickets_gdf.empty or crimes_gdf.empty:
        return np.nan, 0, len(crimes_gdf), 0

    buffers  = tickets_gdf.copy()
    buffers['geometry'] = buffers.geometry.buffer(radius)
    dissolved = buffers[['geometry']].dissolve()

    buffer_area  = dissolved.geometry.iloc[0].area
    area_ratio   = buffer_area / total_area

    crimes_inside = gpd.sjoin(
        crimes_gdf[['geometry']],
        dissolved.reset_index()[['geometry']],
        how='inner', predicate='within'
    )
    n_inside    = len(crimes_inside)
    n_total     = len(crimes_gdf)
    crime_ratio = n_inside / n_total if n_total > 0 else 0
    ccr         = crime_ratio / area_ratio if area_ratio > 0 else np.nan
    return ccr, n_inside, n_total, buffer_area

#%% SINGLE-YEAR CCR TABLE — 2019

print(f"\nComputing CCR summary for {SNAPSHOT_YEAR}...")

categories    = ['1_7_days', '8_21_days', '21_plus_days']
labels_map    = {'1_7_days': '1–7 days', '8_21_days': '8–21 days', '21_plus_days': '21+ days'}
snapshot_rows = []

for month in range(1, 13):
    yr, mo   = SNAPSHOT_YEAR, month
    yr1, mo1 = next_month(yr, mo)
    if yr1 > SNAPSHOT_YEAR:
        break

    tickets_t  = get_ex_ante_tickets(gdf_disorder, yr, mo)
    crimes_t1  = get_crimes_in_month(gdf_crime, yr1, mo1)
    crimes_t   = get_crimes_in_month(gdf_crime, yr, mo)

    if tickets_t.empty or crimes_t.empty:
        continue

    for radius in RADII_METERS:
        for cat in categories:
            cat_tickets = tickets_t[tickets_t['neglect_category'] == cat]
            ccr_main, n_in, n_tot, area = compute_ccr(cat_tickets, crimes_t1, radius, austin_area_m2)
            ccr_plac, *_               = compute_ccr(cat_tickets, crimes_t,  radius, austin_area_m2)

            snapshot_rows.append({
                'month': mo, 'radius_m': radius, 'category': cat,
                'tickets_n': len(cat_tickets),
                'n_inside': n_in, 'n_total': n_tot, 'buffer_area_m2': area,
                'CCR_original': ccr_main, 'CCR_placebo': ccr_plac
            })

df_snap = pd.DataFrame(snapshot_rows)

# Aggregate over all months of 2019
df_snap_agg = df_snap.groupby(['radius_m', 'category']).agg(
    tickets_n   = ('tickets_n',    'sum'),
    n_inside    = ('n_inside',     'sum'),
    n_total     = ('n_total',      'sum'),
    buffer_area = ('buffer_area_m2','sum'),
    months      = ('month',        'count')
).reset_index()

df_snap_agg['area_pct']      = df_snap_agg['buffer_area'] / (austin_area_m2 * df_snap_agg['months']) * 100
df_snap_agg['crime_pct']     = df_snap_agg['n_inside']    / df_snap_agg['n_total'] * 100
df_snap_agg['CCR']           = (df_snap_agg['crime_pct'] / df_snap_agg['area_pct'])

print(f"\n--- {SNAPSHOT_YEAR} CCR by Neglect Duration ---")
print(df_snap_agg[['radius_m', 'category', 'tickets_n', 'area_pct', 'crime_pct', 'CCR']].to_string(index=False))

#%% HISTORICAL CCR — 2014 to 2026

print("\nRunning full historical analysis (2014–2026)... this will take a few minutes.")

hist_rows = []
min_year  = gdf_disorder['Created Date'].dt.year.min()
max_year  = gdf_disorder['Created Date'].dt.year.max()

for year in range(min_year, max_year + 1):
    for month in range(1, 13):
        if year == 2026 and month >= 6:
            continue
        yr1, mo1   = next_month(year, month)
        tickets_t  = get_ex_ante_tickets(gdf_disorder, year, month)
        crimes_t1  = get_crimes_in_month(gdf_crime, yr1, mo1)
        crimes_t   = get_crimes_in_month(gdf_crime, year, month)

        if tickets_t.empty or crimes_t.empty:
            continue

        for radius in RADII_METERS:
            for cat in categories:
                cat_tickets = tickets_t[tickets_t['neglect_category'] == cat]
                ccr_m, n_in, n_tot, area = compute_ccr(cat_tickets, crimes_t1, radius, austin_area_m2)
                ccr_p, *_               = compute_ccr(cat_tickets, crimes_t,  radius, austin_area_m2)
                hist_rows.append({
                    'year': year, 'month': month, 'radius_m': radius, 'category': cat,
                    'tickets_n': len(cat_tickets),
                    'n_inside': n_in, 'n_total': n_tot, 'buffer_area_m2': area,
                    'CCR_original': ccr_m, 'CCR_placebo': ccr_p
                })

df_hist = pd.DataFrame(hist_rows)
df_hist.to_csv('data/clean/ccr_historical_full.csv', index=False)

# Grand total
df_grand = df_hist.groupby(['radius_m', 'category']).agg(
    n_inside    = ('n_inside',      'sum'),
    n_total     = ('n_total',       'sum'),
    buffer_area = ('buffer_area_m2','sum'),
    months      = ('month',         'count')
).reset_index()

df_grand['area_pct']  = df_grand['buffer_area'] / (austin_area_m2 * df_grand['months']) * 100
df_grand['crime_pct'] = df_grand['n_inside']    / df_grand['n_total'] * 100
df_grand['CCR']       = df_grand['crime_pct']   / df_grand['area_pct']

print("\n--- Grand Total CCR (2014–2026) ---")
print(df_grand[['radius_m', 'category', 'area_pct', 'crime_pct', 'CCR']].to_string(index=False))

#%% FIGURE 1 — Historical CCR by neglect duration

fig, ax = plt.subplots(figsize=(10, 6))
colors = {'1_7_days': '#69BE28', '8_21_days': '#A5ACAF', '21_plus_days': '#002244'}

for cat in categories:
    sub = df_grand[df_grand['category'] == cat]
    ax.plot(sub['radius_m'], sub['CCR'], marker='o',
            color=colors[cat], label=labels_map[cat], linewidth=2.5)

ax.axhline(1, color='tomato', linestyle='--', linewidth=1.5, label='No concentration (CCR = 1)')
ax.set_title('Crime Concentration by Neglect Duration\nAustin 2014–2026 (all months aggregated)', fontweight='bold')
ax.set_xlabel('Buffer radius (m)')
ax.set_ylabel('Crime Concentration Ratio (CCR)')
ax.legend()
plt.tight_layout()
plt.savefig('figures/ccr_historical_by_duration.png', dpi=150)
plt.show()

#%% FIGURE 2 — 3-panel map: 50m, 100m, 200m

radii = [50, 100, 200]

print(f"\nBuilding 3-panel map for {SNAPSHOT_YEAR}-{SNAPSHOT_MONTH:02d}...")

t_year, t_month   = SNAPSHOT_YEAR, SNAPSHOT_MONTH
t1_year, t1_month = next_month(t_year, t_month)

tickets_map = get_ex_ante_tickets(gdf_disorder, t_year, t_month)
crimes_map  = get_crimes_in_month(gdf_crime, t1_year, t1_month)

from src.data_wrangling import gdf_boundaries as austin_boundary
austin_base = austin_boundary.to_crs("EPSG:4326")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cat_colors_fill = {
    '1_7_days': '#69BE28',
    '8_21_days': '#f5a623',
    '21_plus_days': '#002244'
}

crime_colors = {
    '1_7_days': '#69BE28',
    '8_21_days': '#f5a623',
    '21_plus_days': '#002244',
    'outside': '#A5ACAF'
}

for i, radius in enumerate(radii):

    ax = axes[i]

    # --- Build dissolved buffers ---
    dissolved_layers = {}
    for cat in categories:
        cat_t = tickets_map[tickets_map['neglect_category'] == cat]
        if cat_t.empty:
            continue

        buf = cat_t.copy()
        buf = buf.to_crs(CRS_METRIC)
        buf['geometry'] = buf.geometry.buffer(radius)

        dissolved_layers[cat] = buf[['geometry']].dissolve()

    # --- Tag crimes ---
    crimes_tagged = crimes_map[['geometry']].copy().to_crs(CRS_METRIC)
    crimes_tagged['inside_cat'] = 'outside'

    for cat in ['21_plus_days', '8_21_days', '1_7_days']:
        if cat not in dissolved_layers:
            continue

        inside = gpd.sjoin(
            crimes_tagged[crimes_tagged['inside_cat'] == 'outside'][['geometry']],
            dissolved_layers[cat].reset_index()[['geometry']],
            how='inner', predicate='within'
        ).index

        crimes_tagged.loc[inside, 'inside_cat'] = cat

    # --- Plot base map ---
    austin_base.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.4)

    # --- Plot buffers ---
    for cat, diss in dissolved_layers.items():
        diss.to_crs("EPSG:4326").plot(
            ax=ax, color=cat_colors_fill[cat], alpha=0.20
        )
        diss.to_crs("EPSG:4326").boundary.plot(
            ax=ax, color=cat_colors_fill[cat], linewidth=0.5
        )

    # --- Plot crimes ---
    crime_plot = crimes_tagged.to_crs("EPSG:4326")

    for tag in ['outside', '1_7_days', '8_21_days', '21_plus_days']:
        sub = crime_plot[crime_plot['inside_cat'] == tag]
        if sub.empty:
            continue

        ax.scatter(
            sub.geometry.x,
            sub.geometry.y,
            s=6 if tag != 'outside' else 2,
            c=crime_colors[tag],
            alpha=0.9 if tag != 'outside' else 0.3,
            zorder=3
        )

    ax.set_title(f"{radius}m buffer", fontsize=11, fontweight='bold')
    ax.set_axis_off()

# --- Shared legend ---
patches = [
    mpatches.Patch(color='#69BE28', alpha=0.7, label='1–7 days'),
    mpatches.Patch(color='#f5a623', alpha=0.7, label='8–21 days'),
    mpatches.Patch(color='#002244', alpha=0.7, label='21+ days'),
    mpatches.Patch(color='#A5ACAF', alpha=0.5, label='Outside buffer'),
]

fig.legend(handles=patches, loc='lower center', ncol=4, frameon=False)

plt.suptitle(
    f"Disorder Buffers vs Crime\nJan {SNAPSHOT_YEAR} (Disorder) → Feb {SNAPSHOT_YEAR} (Crime)",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('figures/map_disorder_buffers_3panel.png', dpi=150)
plt.show()

print("\nDone. 3-panel figure saved.")