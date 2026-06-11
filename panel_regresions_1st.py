import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

from panel_builder import final_panel

df_model = final_panel.copy()

#%%

mean_crime = df_model['Total_Crime'].mean()
var_crime = df_model['Total_Crime'].var()

print(f"Mean of Total_Crime: {mean_crime:.4f}")
print(f"Variance of Total_Crime: {var_crime:.4f}")

if var_crime > mean_crime:
    print("Diagnosis: Variance is strictly greater than Mean. OVERDISPERSION DETECTED.")
    print("Action: Negative Binomial model is required. Standard Poisson will underestimate standard errors.")
else:
    print("Diagnosis: No severe overdispersion.")
    print("Action: Standard Poisson model is viable.")

#%%

control_vars = [
    'Lag_Avg_Repair_Days',
    'population',
    'median_income',
    'poverty_rate',
    'rent_rate',
    'vacancy_rate',
    'hs_rate',
    'bachelors_rate',
    'long_commute_rate',
    'unemployment_rate',
    'young_males_rate',
    'residential_mobility_rate',
    'single_parent_rate'
]

# Drop rows with NaNs in these columns to calculate VIF properly
df_vif_check = df_model[control_vars].dropna()

# Add constant for VIF calculation
X = sm.add_constant(df_vif_check)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Sort and display features with concerning VIF
vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
print(vif_data)
print("\nNote: Any Feature (excluding const) with VIF > 5 should ideally be dropped to prevent multicollinearity.")

print("\n--- 3. ZERO-INFLATION CHECK ---")

zeros_count = (df_model['Total_Crime'] == 0).sum()
total_obs = len(df_model)
zero_pct = (zeros_count / total_obs) * 100

print(f"Total observations: {total_obs}")
print(f"Zero crime observations: {zeros_count} ({zero_pct:.2f}%)")

if zero_pct > 30:
    print("Diagnosis: High zero-inflation detected.")
    print("Action: Consider Zero-Inflated Negative Binomial (ZINB) if standard NB fails to converge.")
else:
    print("Diagnosis: Zeros are within expected range for tract-level monthly panel.")

#%%

cols_to_check = [
    'Total_Crime', 'Lag_Avg_Repair_Days', 'population', 'median_income',
    'poverty_rate', 'rent_rate', 'vacancy_rate', 'hs_rate', 'bachelors_rate',
    'long_commute_rate', 'unemployment_rate', 'young_males_rate',
    'residential_mobility_rate', 'single_parent_rate', 'geoid', 'Month_str'
]

# 2. FIX DEL CRASH: Tirar NaNs explícitamente para que las longitudes cuadren
df_clean = df_model.dropna(subset=cols_to_check).copy()

# 3. FIX DEL OVERFLOW: Escalar números gigantes para no reventar la función exponencial
df_clean['median_income_10k'] = df_clean['median_income'] / 10000
df_clean['population_1k'] = df_clean['population'] / 1000

# Fórmula actualizada con las variables escaladas
formula = (
    "Total_Crime ~ Lag_Avg_Repair_Days + population_1k + median_income_10k + "
    "poverty_rate + rent_rate + vacancy_rate + hs_rate + bachelors_rate + "
    "long_commute_rate + unemployment_rate + young_males_rate + "
    "residential_mobility_rate + single_parent_rate + "
    "C(geoid) + C(Month_str)"
)

print(f"Filas listas para el modelo: {len(df_clean)}")
print("Ajustando el modelo Binomial Negativo TWFE (espera un momento por las dummies)...")

# Ajuste del modelo
bwt_model = smf.negativebinomial(formula=formula, data=df_clean).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_clean['geoid']},
    maxiter=200,      # Le damos un poco más de iteraciones para que converja suave
    method='lbfgs'
)

print("\n================ RESULTADOS FINALES ================")
params_to_show = [
    'Lag_Avg_Repair_Days', 'population_1k', 'median_income_10k', 'poverty_rate',
    'rent_rate', 'vacancy_rate', 'hs_rate', 'bachelors_rate',
    'long_commute_rate', 'unemployment_rate', 'young_males_rate',
    'residential_mobility_rate', 'single_parent_rate'
]

summary_df = pd.DataFrame({
    'Coeficiente': bwt_model.params[params_to_show],
    'P-Value': bwt_model.pvalues[params_to_show]
})

print(summary_df)

#%%

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# 1. Preparar el MultiIndex requerido por linearmodels
df_lm = df_clean.copy()
df_lm['Month_dt'] = pd.to_datetime(df_lm['Month_str'])
df_lm = df_lm.set_index(['geoid', 'Month_dt'])

# 2. FIX: Dejar ÚNICAMENTE la variable que cambia en el tiempo
# El censo ya fue absorbido por el efecto fijo.
exog_vars = ['Lag_Avg_Repair_Days']

# Agregar constante
exog = sm.add_constant(df_lm[exog_vars])
endog = df_lm['Total_Crime']

print("Ajustando Panel OLS con Efectos Fijos de Dos Vías (TWFE)...")

# 3. FIX: Le agregamos drop_absorbed=True por si acaso queda alguna constante escondida
model = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)

# Correr el modelo con los Errores Estándar Agrupados
results = model.fit(cov_type='clustered', cluster_entity=True)

print(results.summary)

#%%

from linearmodels.panel import RandomEffects

print("Ajustando Panel con Efectos Aleatorios (Random Effects)...")
# Aquí ya no usamos entity_effects=True, usamos el modelo RandomEffects
model_re = RandomEffects(endog, exog)

# Ajustamos con errores agrupados
results_re = model_re.fit(cov_type='clustered', cluster_entity=True)
print(results_re.summary)

#%%

import numpy as np

# Crear rangos de negligencia
condiciones = [
    (df_lm['Lag_Avg_Repair_Days'] == 0),
    (df_lm['Lag_Avg_Repair_Days'] > 0) & (df_lm['Lag_Avg_Repair_Days'] <= 7),
    (df_lm['Lag_Avg_Repair_Days'] > 7) & (df_lm['Lag_Avg_Repair_Days'] <= 21),
    (df_lm['Lag_Avg_Repair_Days'] > 21)
]
categorias = ['0_Zero', '1_Low_1_7', '2_Med_8_21', '3_High_21_plus']
df_lm['Negligence_Level'] = np.select(condiciones, categorias, default='0_Zero')

# Actualizar el modelo con la variable categórica (usando '0_Zero' como base)
exog_vars_cat = [
    'population_1k', 'median_income_10k', 'poverty_rate',
    'rent_rate', 'vacancy_rate', 'hs_rate', 'bachelors_rate',
    'long_commute_rate', 'unemployment_rate', 'young_males_rate',
    'residential_mobility_rate', 'single_parent_rate'
]

exog_cat = sm.add_constant(df_lm[exog_vars_cat])
# Generar dummies para los niveles, eliminando la base para evitar colinealidad perfecta
dummies = pd.get_dummies(df_lm['Negligence_Level'], drop_first=True).astype(float)
exog_cat = pd.concat([exog_cat, dummies], axis=1)

print("Ajustando Panel con Umbrales de Negligencia...")
model_thresh = RandomEffects(endog, exog_cat)
results_thresh = model_thresh.fit(cov_type='clustered', cluster_entity=True)
print(results_thresh.summary)