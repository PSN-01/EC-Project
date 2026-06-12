import pandas as pd
import importlib
import numpy as np
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

df_311['Start_Month'] = df_311['Created Date'].dt.to_period('M')
df_311['End_Month']   = df_311['Close Date'].dt.to_period('M')

disorder_categories = [
    'Graffiti Abatement',
    'APH - Graffiti Abatement - Public Property',
    'DSD - Graffiti Abatement - Private Property',
    'Debris in Street',
    'SBO - Debris in Street',
    'TPW - Debris in Street',
    'APD - Vehicle Abatement Report',
]
df_311['ticket_type'] = np.where(
    df_311['SR Description'].isin(disorder_categories),
    'disorder',
    'infrastructure'
)


def slice_ticket(row):
    months = pd.period_range(start=row['Start_Month'], end=row['End_Month'], freq='M')
    slices = []
    for m in months:
        m_start = max(row['Created Date'], m.start_time)
        m_end   = min(row['Close Date'],   m.end_time)
        days    = (m_end - m_start).days + 1
        slices.append({'Month': m, 'Active_Days': days})
    return slices


slicer_results = df_311.apply(slice_ticket, axis=1)

df_exploded = pd.DataFrame([item for sublist in slicer_results for item in sublist])
df_exploded[boundary_col]    = df_311[boundary_col].repeat(slicer_results.str.len()).values
df_exploded['ticket_type']   = df_311['ticket_type'].repeat(slicer_results.str.len()).values

# Overall average
panel_311 = df_exploded.groupby([boundary_col, 'Month'])['Active_Days'].mean().reset_index()
panel_311.rename(columns={'Active_Days': 'Avg_Repair_Days'}, inplace=True)

# By ticket type
panel_311_typed = (
    df_exploded.groupby([boundary_col, 'Month', 'ticket_type'])['Active_Days']
    .mean()
    .unstack('ticket_type')
    .reset_index()
)
panel_311_typed.columns.name = None
panel_311_typed = panel_311_typed.rename(columns={
    'disorder':       'Avg_Repair_Days_disorder',
    'infrastructure': 'Avg_Repair_Days_infra'
})

df_c = joined_crime.copy()
df_c['Month'] = df_c['Occurred Date'].dt.to_period('M')
panel_crime = df_c.groupby([boundary_col, 'Month']).size().reset_index(name='Total_Crime')

all_months = pd.period_range(start='2014-01', end='2026-05', freq='M')
multi_idx  = pd.MultiIndex.from_product(
    [austin_valid_geoids, all_months],
    names=[boundary_col, 'Month']
)
skeleton = pd.DataFrame(index=multi_idx).reset_index()

panel = pd.merge(skeleton,  panel_311,       on=[boundary_col, 'Month'], how='left')
panel = pd.merge(panel,     panel_311_typed,  on=[boundary_col, 'Month'], how='left')
panel = pd.merge(panel,     panel_crime,      on=[boundary_col, 'Month'], how='left')

panel['Total_Crime']              = panel['Total_Crime'].fillna(0)
panel['Avg_Repair_Days']          = panel['Avg_Repair_Days'].fillna(0)
panel['Avg_Repair_Days_disorder'] = panel['Avg_Repair_Days_disorder'].fillna(0)
panel['Avg_Repair_Days_infra']    = panel['Avg_Repair_Days_infra'].fillna(0)

panel = panel.sort_values(by=[boundary_col, 'Month'])

panel['Lag_Avg_Repair_Days']          = panel.groupby(boundary_col)['Avg_Repair_Days'].shift(1)
panel['Lag_Avg_Repair_Days_disorder'] = panel.groupby(boundary_col)['Avg_Repair_Days_disorder'].shift(1)
panel['Lag_Avg_Repair_Days_infra']    = panel.groupby(boundary_col)['Avg_Repair_Days_infra'].shift(1)

final_panel = pd.merge(panel, demo_data, left_on=boundary_col, right_on='GEOID', how='left')
final_panel = final_panel.dropna(subset=['Lag_Avg_Repair_Days']).copy()
final_panel.to_csv("data/clean/final_panel.csv", index=False)