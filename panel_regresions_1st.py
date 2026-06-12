import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.stats.outliers_influence import variance_inflation_factor

import panel_builder
from panel_builder import final_panel
importlib.reload(panel_builder)

df_model = final_panel.copy()
df_model['geoid'] = df_model['geoid'].astype(str)
df_model['Month_str'] = df_model['Month'].astype(str)

#%% DIAGNOSTICS

mean_crime = df_model['Total_Crime'].mean()
var_crime  = df_model['Total_Crime'].var()
print(f"Mean: {mean_crime:.4f} | Variance: {var_crime:.4f}")
print("OVERDISPERSION — use Negative Binomial." if var_crime > mean_crime else "No overdispersion — Poisson viable.")

control_vars = [
    'Lag_Avg_Repair_Days', 'population', 'median_income', 'poverty_rate',
    'rent_rate', 'vacancy_rate', 'hs_rate', 'bachelors_rate',
    'long_commute_rate', 'unemployment_rate', 'young_males_rate',
    'residential_mobility_rate', 'single_parent_rate'
]
X_vif = sm.add_constant(df_model[control_vars].dropna())
vif_data = (
    pd.DataFrame({'Feature': X_vif.columns,
                  'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]})
    .sort_values('VIF', ascending=False)
    .reset_index(drop=True)
)
print(vif_data)

zero_pct = (df_model['Total_Crime'] == 0).mean() * 100
print(f"Zeros: {zero_pct:.2f}% — {'High zero-inflation.' if zero_pct > 30 else 'Within normal range.'}")

#%% CLEAN DATASET

cols_to_check = control_vars + ['Total_Crime', 'geoid', 'Month_str']
df_clean = df_model.dropna(subset=cols_to_check).copy()
df_clean['median_income_10k'] = df_clean['median_income'] / 10_000
df_clean['population_1k']     = df_clean['population']    / 1_000

df_lm = df_clean.copy()
df_lm['Month_dt'] = pd.to_datetime(df_lm['Month_str'])
df_lm = df_lm.set_index(['geoid', 'Month_dt'])

endog = df_lm['Total_Crime']

#%% BASELINE MODELS

exog_ols = sm.add_constant(df_lm[['Lag_Avg_Repair_Days']])

results_fe = PanelOLS(endog, exog_ols, entity_effects=True, time_effects=True, drop_absorbed=True)\
    .fit(cov_type='clustered', cluster_entity=True)
# print(results_fe.summary)

results_re = RandomEffects(endog, exog_ols)\
    .fit(cov_type='clustered', cluster_entity=True)
# print(results_re.summary)

#%% HAUSMAN TEST

# Compares FE vs RE to decide which is appropriate.
# H0: RE is consistent (no correlation between entity effects and regressors).
# A significant result (p < 0.05) favors Fixed Effects.
b_fe = results_fe.params
b_re = results_re.params
common = b_fe.index.intersection(b_re.index)

diff = b_fe[common] - b_re[common]
cov_diff = results_fe.cov[common].loc[common] - results_re.cov[common].loc[common]

hausman_stat = float(diff @ np.linalg.inv(cov_diff.values) @ diff)
hausman_df   = len(common)
hausman_p    = 1 - __import__('scipy.stats', fromlist=['chi2']).chi2.cdf(hausman_stat, df=hausman_df)

print(f"Hausman stat: {hausman_stat:.4f} | df: {hausman_df} | p-value: {hausman_p:.4f}")
print("Conclusion: Fixed Effects preferred." if hausman_p < 0.05 else "Conclusion: Random Effects consistent.")

#%% MAIN MODEL — negligence thresholds

conditions = [
    df_lm['Lag_Avg_Repair_Days'] == 0,
    (df_lm['Lag_Avg_Repair_Days'] > 0)  & (df_lm['Lag_Avg_Repair_Days'] <= 7),
    (df_lm['Lag_Avg_Repair_Days'] > 7)  & (df_lm['Lag_Avg_Repair_Days'] <= 21),
     df_lm['Lag_Avg_Repair_Days'] > 21,
]
labels = ['0_Zero', '1_Low_1_7', '2_Med_8_21', '3_High_21plus']
df_lm['Negligence_Level'] = np.select(conditions, labels, default='0_Zero')

demo_controls = [
    'population_1k', 'median_income_10k', 'poverty_rate', 'rent_rate',
    'vacancy_rate', 'hs_rate', 'bachelors_rate', 'long_commute_rate',
    'unemployment_rate', 'young_males_rate', 'residential_mobility_rate',
    'single_parent_rate'
]
dummies     = pd.get_dummies(df_lm['Negligence_Level'], drop_first=True).astype(float)
exog_thresh = pd.concat([sm.add_constant(df_lm[demo_controls]), dummies], axis=1)

results_thresh_re = RandomEffects(endog, exog_thresh)\
    .fit(cov_type='clustered', cluster_entity=True)
# print(results_thresh_re.summary)

#%% MAIN MODEL — negligence thresholds with Fixed Effects

# FE absorbs all time-invariant tract characteristics (demographics included),
# so demo_controls are dropped — they would be collinear with entity effects.
results_thresh_fe = PanelOLS(endog, exog_thresh, entity_effects=True, time_effects=True, drop_absorbed=True)\
    .fit(cov_type='clustered', cluster_entity=True)
print(results_thresh_fe.summary)

#%% FIGURE — RE vs FE threshold coefficients

threshold_keys   = ['1_Low_1_7', '2_Med_8_21', '3_High_21plus']
threshold_labels = ['1–7 days', '8–21 days', '21+ days']
x = np.arange(len(threshold_keys))
width = 0.35

def get_coefs_ci(results, keys):
    coefs  = results.params[keys].values
    ci_low = results.conf_int()['lower'][keys].values
    ci_hi  = results.conf_int()['upper'][keys].values
    return coefs, ci_low, ci_hi

re_coefs, re_low, re_hi = get_coefs_ci(results_thresh_re, threshold_keys)
fe_coefs, fe_low, fe_hi = get_coefs_ci(results_thresh_fe, threshold_keys)

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

ax.errorbar(x - width/2, re_coefs, yerr=[re_coefs - re_low, re_hi - re_coefs],
            fmt='o', color='steelblue', capsize=5, capthick=1.5, linewidth=1.5, label='Random Effects')
ax.errorbar(x + width/2, fe_coefs, yerr=[fe_coefs - fe_low, fe_hi - fe_coefs],
            fmt='s', color='tomato', capsize=5, capthick=1.5, linewidth=1.5, label='Fixed Effects')

ax.set_xticks(x)
ax.set_xticklabels(threshold_labels)
ax.set_ylabel('Coefficient (vs. 0 repair days)')
ax.set_xlabel('Lag negligence level')
ax.set_title('Effect of repair delay on crime — RE vs FE')
ax.legend()
plt.tight_layout()
import os; os.makedirs('figures', exist_ok=True)
# plt.savefig('figures/threshold_coefs.png', dpi=150)
plt.show()

#%% PLACEBO TEST

# Uses the lead (future month) instead of the lag.
# If significant, the causal direction is suspect.
# If near zero, the lag identification is valid.
df_lm['Lead_Avg_Repair_Days'] = df_lm.groupby(level=0)['Avg_Repair_Days'].shift(-1)

placebo_idx   = df_lm['Lead_Avg_Repair_Days'].dropna().index
exog_placebo  = sm.add_constant(df_lm.loc[placebo_idx, ['Lead_Avg_Repair_Days']])
endog_placebo = endog.loc[placebo_idx]

results_placebo = PanelOLS(endog_placebo, exog_placebo,
                           entity_effects=True, time_effects=True, drop_absorbed=True)\
    .fit(cov_type='clustered', cluster_entity=True)
print(results_placebo.summary)

#%% DISORDER VS INFRASTRUCTURE

threshold_levels = ['1_Low_1_7', '2_Med_8_21', '3_High_21plus']

for col, label in [('Lag_Avg_Repair_Days_disorder', 'Disorder (graffiti, debris, vehicles)'),
                   ('Lag_Avg_Repair_Days_infra',    'Infrastructure (potholes, lights, sidewalks)')]:

    conds = [
        df_lm[col] == 0,
        (df_lm[col] > 0)  & (df_lm[col] <= 7),
        (df_lm[col] > 7)  & (df_lm[col] <= 21),
         df_lm[col] > 21,
    ]
    col_cat        = col + '_level'
    df_lm[col_cat] = np.select(conds, labels, default='0_Zero')

    dummies_t = pd.get_dummies(df_lm[col_cat], drop_first=True).astype(float)
    exog_t    = pd.concat([sm.add_constant(df_lm[demo_controls]), dummies_t], axis=1)

    res = PanelOLS(endog, exog_t, entity_effects=True, time_effects=True, drop_absorbed=True)\
        .fit(cov_type='clustered', cluster_entity=True)

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(pd.DataFrame({
        'coef':    res.params[threshold_levels],
        'p-value': res.pvalues[threshold_levels]
    }))