import pandas as pd
import importlib
import src.data_loader as data_loader
import src.data_wrangling as data_wrangling

importlib.reload(data_loader)
importlib.reload(data_wrangling)

from src.data_wrangling import (
    joined_311,
    joined_crime,
    demo_data,
    austin_valid_geoids,
    boundary_col
)


df_311 = joined_311.copy()

cutoff_date = pd.to_datetime('2026-06-10')
df_311['Close Date'] = df_311['Close Date'].fillna(cutoff_date)
df_311 = df_311[df_311['Close Date'] >= df_311['Created Date']].copy()

# Monthly
df_311['Start_Month'] = df_311['Created Date'].dt.to_period('M')
df_311['End_Month'] = df_311['Close Date'].dt.to_period('M')


def slice_ticket(row):
    """
    Takes a report ticket and generates a list of dictionaries
    one by one for each month in which the report ticket was still on, with its active days.
    """
    # Rango de meses que tocó este ticket
    months = pd.period_range(start=row['Start_Month'], end=row['End_Month'], freq='M')
    slices = []

    for m in months:
        # Calculate exact limits of that month
        m_start = max(row['Created Date'], m.start_time)
        m_end = min(row['Close Date'], m.end_time)

        # ensures that the report ticket closed on the same day counts as one day
        days = (m_end - m_start).days + 1
        slices.append({'Month': m, 'Active_Days': days})

    return slices


slicer_results = df_311.apply(slice_ticket, axis=1)

# Flatten the dictionary list (explode)
df_exploded = pd.DataFrame([item for sublist in slicer_results for item in sublist])
# Heredar el GEOID a las rebanadas
df_exploded[boundary_col] = df_311[boundary_col].repeat(slicer_results.str.len()).values

# Calculates average of negligencia by GEOID and Month
panel_311 = df_exploded.groupby([boundary_col, 'Month'])['Active_Days'].mean().reset_index()
panel_311.rename(columns={'Active_Days': 'Avg_Repair_Days'}, inplace=True)

df_c = joined_crime.copy()
df_c['Month'] = df_c['Occurred Date'].dt.to_period('M')
panel_crime = df_c.groupby([boundary_col, 'Month']).size().reset_index(name='Total_Crime')


# Force that every tract has every month, just in case
all_months = pd.period_range(start='2014-01', end='2026-05', freq='M')

multi_idx = pd.MultiIndex.from_product(
    [austin_valid_geoids, all_months],
    names=[boundary_col, 'Month']
)
skeleton = pd.DataFrame(index=multi_idx).reset_index()

panel = pd.merge(skeleton, panel_311, on=[boundary_col, 'Month'], how='left')
panel = pd.merge(panel, panel_crime, on=[boundary_col, 'Month'], how='left')

# If there is no match, 0 crimes or 0 days of negligence
panel['Total_Crime'] = panel['Total_Crime'].fillna(0)
panel['Avg_Repair_Days'] = panel['Avg_Repair_Days'].fillna(0)

# Lag
# Organice by space and time before making the shift
panel = panel.sort_values(by=[boundary_col, 'Month'])

# move the negligence one month down by each tract
panel['Lag_Avg_Repair_Days'] = panel.groupby(boundary_col)['Avg_Repair_Days'].shift(1)

# Demo data
final_panel = pd.merge(panel, demo_data, left_on=boundary_col, right_on='GEOID', how='left')


# 2014-01 will always have NaN as lag, cause there is no data b4
final_panel = final_panel.dropna(subset=['Lag_Avg_Repair_Days']).copy()
final_panel.to_csv("data/clean/final_panel.csv", index=False)