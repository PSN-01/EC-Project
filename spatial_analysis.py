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

# ── PARAMETERS (customize here) ───────────────────────────────────────────────

RADII_METERS  = [50, 100, 200]   # buffer sizes to test
PROOF_YEAR    = 2019             # year for proof of concept
PROOF_MONTH   = 1                # month t (January 2019)
MIN_TICKETS   = 30               # minimum active tickets to include a month
CRS_METRIC    = "EPSG:32614"     # projected CRS for Austin (meters)

#%% PREPARE DATA

# Filter 311 to social disorder only
gdf_disorder = joined_311[joined_311['ticket_type'] == 'disorder'].copy()
gdf_disorder['Created Date'] = pd.to_datetime(gdf_disorder['Created Date'])
gdf_disorder['Close Date']   = pd.to_datetime(gdf_disorder['Close Date'])

# Project to metric CRS
gdf_disorder = gdf_disorder.to_crs(CRS_METRIC)
gdf_crime    = joined_crime.copy().to_crs(CRS_METRIC)
gdf_crime['Occurred Date'] = pd.to_datetime(gdf_crime['Occurred Date'])

# Austin total area (for CCR denominator)
austin_area_m2 = gdf_disorder.unary_union.convex_hull.area  # approximation

print(f"Total disorder tickets (all years): {len(gdf_disorder):,}")
print(f"Total crime incidents  (all years): {len(gdf_crime):,}")

#%% PROOF OF CONCEPT — January 2019 → February 2019

def get_active_tickets(gdf, year, month):
    """Returns disorder tickets that were open at any point during (year, month)."""
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end   = month_start + pd.offsets.MonthEnd(0)
    mask = (gdf['Created Date'] <= month_end) & (gdf['Close Date'] >= month_start)
    return gdf[mask].copy()

def get_crimes_in_month(gdf, year, month):
    """Returns crimes that occurred in (year, month)."""
    mask = (
        (gdf['Occurred Date'].dt.year  == year) &
        (gdf['Occurred Date'].dt.month == month)
    )
    return gdf[mask].copy()

def next_month(year, month):
    if month == 12:
        return year + 1, 1
    return year, month + 1

# ── Pull data for proof month ──────────────────────────────────────────────────
t_year, t_month   = PROOF_YEAR, PROOF_MONTH
t1_year, t1_month = next_month(t_year, t_month)

active_tickets = get_active_tickets(gdf_disorder, t_year, t_month)
next_crimes    = get_crimes_in_month(gdf_crime, t1_year, t1_month)

print(f"\nProof of concept: {t_year}-{t_month:02d} → {t1_year}-{t1_month:02d}")
print(f"Active disorder tickets (month t):  {len(active_tickets):,}")
print(f"Crime incidents (month t+1):        {len(next_crimes):,}")

if len(active_tickets) < MIN_TICKETS:
    print(f"WARNING: fewer than {MIN_TICKETS} tickets — results may not be meaningful.")

#%% CCR FOR EACH RADIUS

results_poc = []

for radius in RADII_METERS:

    # 1. Build buffers
    buffers = active_tickets.copy()
    buffers['geometry'] = buffers.geometry.buffer(radius)

    # 2. Dissolve overlapping buffers → single polygon, no double counting
    dissolved = buffers[['geometry']].dissolve()

    # 3. Area ratio
    buffer_area  = dissolved.geometry.iloc[0].area
    area_ratio   = buffer_area / austin_area_m2

    # 4. Spatial join: which crimes fall inside the dissolved buffer?
    crimes_inside = gpd.sjoin(
        next_crimes[['geometry']],
        dissolved.reset_index()[['geometry']],
        how='inner',
        predicate='within'
    )

    n_inside  = len(crimes_inside)
    n_total   = len(next_crimes)
    n_outside = n_total - n_inside
    crime_ratio = n_inside / n_total if n_total > 0 else 0

    # 5. Crime Concentration Ratio
    ccr = crime_ratio / area_ratio if area_ratio > 0 else np.nan

    results_poc.append({
        'radius_m':      radius,
        'active_tickets': len(active_tickets),
        'buffer_area_km2': buffer_area / 1e6,
        'area_ratio_pct':  area_ratio * 100,
        'crimes_inside':   n_inside,
        'crimes_outside':  n_outside,
        'crimes_total':    n_total,
        'crime_ratio_pct': crime_ratio * 100,
        'CCR':             ccr
    })

    print(f"\nRadius {radius}m:")
    print(f"  Buffer area:      {buffer_area/1e6:.2f} km²  ({area_ratio*100:.1f}% of Austin)")
    print(f"  Crimes inside:    {n_inside} / {n_total} ({crime_ratio*100:.1f}%)")
    print(f"  CCR:              {ccr:.3f}  {'← concentration' if ccr > 1 else '← no concentration'}")

df_poc = pd.DataFrame(results_poc)
print("\n", df_poc[['radius_m', 'area_ratio_pct', 'crime_ratio_pct', 'CCR']].to_string(index=False))

#%% PRUEBA 1 — INFRASTRUCTURE VS DISORDER

gdf_infra = joined_311[joined_311['ticket_type'] == 'infrastructure'].copy()
gdf_infra['Created Date'] = pd.to_datetime(gdf_infra['Created Date'])
gdf_infra['Close Date']   = pd.to_datetime(gdf_infra['Close Date'])
gdf_infra = gdf_infra.to_crs(CRS_METRIC)

active_infra = get_active_tickets(gdf_infra, t_year, t_month)
print(f"\nPRUEBA 1 — Infrastructure vs Disorder (same month, same crimes)")
print(f"Active infrastructure tickets: {len(active_infra):,}")

comparison = []
for radius in RADII_METERS:
    for ticket_type, tickets, label in [
        ('disorder',       active_tickets, 'Disorder'),
        ('infrastructure', active_infra,   'Infrastructure'),
    ]:
        buffers = tickets.copy()
        buffers['geometry'] = buffers.geometry.buffer(radius)
        dissolved = buffers[['geometry']].dissolve()

        buffer_area = dissolved.geometry.iloc[0].area
        area_ratio  = buffer_area / austin_area_m2

        crimes_inside = gpd.sjoin(
            next_crimes[['geometry']],
            dissolved.reset_index()[['geometry']],
            how='inner', predicate='within'
        )
        crime_ratio = len(crimes_inside) / len(next_crimes)
        ccr = crime_ratio / area_ratio if area_ratio > 0 else np.nan
        comparison.append({'radius_m': radius, 'type': label,
                           'area_pct': area_ratio * 100,
                           'crime_pct': crime_ratio * 100, 'CCR': ccr})

df_comparison = pd.DataFrame(comparison)
print(df_comparison.to_string(index=False))
print("\nExpected if BWT is real: Disorder CCR >> Infrastructure CCR")

#%% PRUEBA 2 — PLACEBO TEMPORAL

future_tickets = get_active_tickets(gdf_disorder, t1_year, t1_month)
prev_crimes    = get_crimes_in_month(gdf_crime, t_year, t_month)

print(f"\nPRUEBA 2 — Temporal Placebo (Feb tickets → Jan crimes)")
print(f"Future tickets: {len(future_tickets):,} | Past crimes: {len(prev_crimes):,}")

placebo_results = []
for radius in RADII_METERS:
    buffers = future_tickets.copy()
    buffers['geometry'] = buffers.geometry.buffer(radius)
    dissolved = buffers[['geometry']].dissolve()

    buffer_area = dissolved.geometry.iloc[0].area
    area_ratio  = buffer_area / austin_area_m2

    crimes_inside = gpd.sjoin(
        prev_crimes[['geometry']],
        dissolved.reset_index()[['geometry']],
        how='inner', predicate='within'
    )
    crime_ratio = len(crimes_inside) / len(prev_crimes)
    ccr = crime_ratio / area_ratio if area_ratio > 0 else np.nan

    original_ccr = df_poc[df_poc['radius_m'] == radius]['CCR'].values[0]
    placebo_results.append({'radius_m': radius, 'CCR_original': original_ccr, 'CCR_placebo': ccr})
    print(f"  Radius {radius}m — Original: {original_ccr:.3f} | Placebo: {ccr:.3f}")

df_placebo = pd.DataFrame(placebo_results)
print("\nExpected if BWT is real: Placebo CCR << Original CCR")

#%% PRUEBA 3 — RANDOM PLACEBO

from shapely.geometry import Point

minx, miny, maxx, maxy = gdf_disorder.total_bounds
n_random = len(active_tickets)
austin_hull = gpd.GeoDataFrame(
    geometry=[gdf_disorder.union_all().convex_hull], crs=CRS_METRIC
)

print(f"\nPRUEBA 3 — Random Placebo ({n_random} random points, 10 runs each)")

random_results = []
np.random.seed(42)
for radius in RADII_METERS:
    ccr_runs = []
    for _ in range(10):
        rand_x = np.random.uniform(minx, maxx, n_random)
        rand_y = np.random.uniform(miny, maxy, n_random)
        rand_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in zip(rand_x, rand_y)],
            crs=CRS_METRIC
        )
        rand_gdf = gpd.sjoin(rand_gdf, austin_hull, how='inner', predicate='within')

        buffers = rand_gdf[['geometry']].copy()
        buffers['geometry'] = buffers.geometry.buffer(radius)
        dissolved = buffers.dissolve()

        buffer_area = dissolved.geometry.iloc[0].area
        area_ratio  = buffer_area / austin_area_m2

        crimes_inside = gpd.sjoin(
            next_crimes[['geometry']],
            dissolved.reset_index()[['geometry']],
            how='inner', predicate='within'
        )
        crime_ratio = len(crimes_inside) / len(next_crimes)
        ccr_runs.append(crime_ratio / area_ratio if area_ratio > 0 else np.nan)

    avg_ccr = np.nanmean(ccr_runs)
    random_results.append({'radius_m': radius, 'CCR_random_avg': avg_ccr})
    print(f"  Radius {radius}m — Random CCR (avg 10 runs): {avg_ccr:.3f}")

print("\nExpected if CCR is well-calibrated: Random CCR ≈ 1.0")

#%% PRUEBA 4 — ESTABILIDAD TEMPORAL (12 meses de 2019)

print(f"\nPRUEBA 4 — Temporal stability across all months of {PROOF_YEAR}")
STABILITY_RADIUS = 100

monthly_results = []
for month in range(1, 13):
    yr, mo    = PROOF_YEAR, month
    yr1, mo1  = next_month(yr, mo)
    if yr1 > PROOF_YEAR:
        break

    tickets_m = get_active_tickets(gdf_disorder, yr, mo)
    crimes_m1 = get_crimes_in_month(gdf_crime, yr1, mo1)

    if len(tickets_m) < MIN_TICKETS or len(crimes_m1) == 0:
        print(f"  {yr}-{mo:02d}: skipped")
        continue

    buffers = tickets_m.copy()
    buffers['geometry'] = buffers.geometry.buffer(STABILITY_RADIUS)
    dissolved = buffers[['geometry']].dissolve()

    buffer_area   = dissolved.geometry.iloc[0].area
    area_ratio    = buffer_area / austin_area_m2
    crimes_inside = gpd.sjoin(
        crimes_m1[['geometry']],
        dissolved.reset_index()[['geometry']],
        how='inner', predicate='within'
    )
    crime_ratio = len(crimes_inside) / len(crimes_m1)
    ccr = crime_ratio / area_ratio if area_ratio > 0 else np.nan

    monthly_results.append({'month': f"{yr}-{mo:02d}", 'tickets': len(tickets_m),
                            'crimes': len(crimes_m1), 'CCR': ccr})
    print(f"  {yr}-{mo:02d}→{yr1}-{mo1:02d} | tickets={len(tickets_m):,} | crimes={len(crimes_m1):,} | CCR={ccr:.3f}")

df_monthly = pd.DataFrame(monthly_results)
print(f"\n  Mean CCR: {df_monthly['CCR'].mean():.3f} | Min: {df_monthly['CCR'].min():.3f} | Max: {df_monthly['CCR'].max():.3f}")

#%% FIGURE — Summary of all 4 tests

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Disorder vs Infrastructure by radius
for label, color in [('Disorder', '#002244'), ('Infrastructure', '#A5ACAF')]:
    sub = df_comparison[df_comparison['type'] == label]
    axes[0].plot(sub['radius_m'], sub['CCR'], marker='o', label=label, color=color, linewidth=2)
axes[0].axhline(1, color='tomato', linestyle='--', linewidth=1.2)
axes[0].set_title('Test 1: Disorder vs Infrastructure')
axes[0].set_xlabel('Buffer radius (m)')
axes[0].set_ylabel('CCR')
axes[0].legend()

# Panel B: Original vs Placebo
axes[1].plot(df_placebo['radius_m'], df_placebo['CCR_original'], marker='o',
             label='Original (Jan→Feb)', color='#002244', linewidth=2)
axes[1].plot(df_placebo['radius_m'], df_placebo['CCR_placebo'], marker='s',
             label='Placebo (Feb→Jan)', color='#69BE28', linewidth=2, linestyle='--')
axes[1].axhline(1, color='tomato', linestyle='--', linewidth=1.2)
axes[1].set_title('Test 2: Temporal Placebo')
axes[1].set_xlabel('Buffer radius (m)')
axes[1].set_ylabel('CCR')
axes[1].legend()

# Panel C: Monthly stability
axes[2].bar(range(len(df_monthly)), df_monthly['CCR'], color='#002244', alpha=0.85)
axes[2].axhline(1, color='tomato', linestyle='--', linewidth=1.5, label='CCR=1 (no concentration)')
axes[2].set_xticks(range(len(df_monthly)))
axes[2].set_xticklabels(df_monthly['month'], rotation=45, fontsize=8)
axes[2].set_title(f'Test 4: CCR Stability 2019 ({STABILITY_RADIUS}m)')
axes[2].set_ylabel('CCR')
axes[2].legend()

plt.suptitle('Spatial Validation Tests — Austin 2019', fontsize=13, fontweight='bold')
plt.tight_layout()
import os; os.makedirs('figures', exist_ok=True)
plt.savefig('figures/spatial_validation_tests.png', dpi=150)
plt.show()

#%% FIGURE — proof of concept map (radius = 100m)

radius_map = 100
buffers_map = active_tickets.copy()
buffers_map['geometry'] = buffers_map.geometry.buffer(radius_map)
dissolved_map = buffers_map[['geometry']].dissolve()

crimes_inside_map = gpd.sjoin(
    next_crimes[['geometry']].copy(),
    dissolved_map.reset_index()[['geometry']],
    how='left',
    predicate='within'
)
crimes_inside_map['inside'] = crimes_inside_map['index_right'].notna()

fig, ax = plt.subplots(figsize=(10, 10))

# Buffer zones
dissolved_map.to_crs("EPSG:4326").boundary.plot(
    ax=ax, color='#69BE28', linewidth=0.6, label='Disorder buffer (100m)'
)
dissolved_map.to_crs("EPSG:4326").plot(
    ax=ax, color='#69BE28', alpha=0.15
)

# Crimes outside
crimes_outside_map = crimes_inside_map[~crimes_inside_map['inside']].to_crs("EPSG:4326")
crimes_inside_only = crimes_inside_map[crimes_inside_map['inside']].to_crs("EPSG:4326")

crimes_outside_map.plot(ax=ax, color='#A5ACAF', markersize=3, alpha=0.5, label='Crime outside buffer')
crimes_inside_only.plot(ax=ax, color='#002244', markersize=5, alpha=0.8, label='Crime inside buffer')

ax.set_title(
    f"Social Disorder Buffers (Jan 2019) vs Crime (Feb 2019)\n"
    f"radius={radius_map}m | {len(active_tickets)} active tickets | "
    f"CCR={df_poc[df_poc['radius_m']==radius_map]['CCR'].values[0]:.2f}",
    fontsize=12
)
ax.legend(loc='upper right')
ax.set_axis_off()
plt.tight_layout()

import os; os.makedirs('figures', exist_ok=True)
plt.savefig('figures/spatial_poc_2019_01.png', dpi=150)
plt.show()