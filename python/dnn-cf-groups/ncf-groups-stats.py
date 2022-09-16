# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import pathlib

EXPERIMENTPATH='groups_2022_2/ml1m/'
DATAPATH='../rs-data-python/grupos/ml1m/'

#groups-4-biasedMF_k8_dsml1m_seed1234_indi_expert.csv
experiment_prefix='groups-4-'
experiment_subfix='_k8_dsml1m_seed1234'

models_dl=[
    'evodeep',
    'gmf',
    'mlp',
    'neumf',
]

models_ml = [
    'biasedMF'
]

models = models_dl + models_ml

# +
pd_maes = pd.DataFrame(columns=['mae'], index=models_dl)

for m in models_dl:
    with open(f'{EXPERIMENTPATH}{m}{experiment_subfix}.mae.result') as f:
        d = f.readlines()[0].split(';')[0]
    pd_maes['mae'][m]=d    

# -

# # MAE model regression on test data

pd_maes

# +
# Carga de los ficheros
groups = {}

fromngroups=2
tongroups=10
fromngroups=4
tongroups=4


for g in range(fromngroups,tongroups+1):
    groups[g] = pd.read_csv(f"{DATAPATH}groups-{g}.csv")
    groups[g]['g'] = str(g)
    groups[g]['middle'] = groups[g].filter(regex='^rating',axis=1).quantile(0.5, axis=1)
    groups[g]['mean'] = groups[g].filter(regex='^rating',axis=1).mean(axis=1, skipna=True)
    raw_cols = ['item'] + ['rating-'+str(i) for i in range(1, g+1)] + ['user-'+str(i) for i in range(1, g+1)]
    groups[g]=groups[g].drop(raw_cols, axis=1)

#groups[4]
# -

# # Models

ml_variations = [ m+experiment_subfix+v for m in models_ml for v in ['_indi_mean', '_indi_expert'] ]
dl_variations = [ m+experiment_subfix+v for m in models_dl for v in ['_indi_mean', '_indi_expert', '_0.250', '_expert'] ]
model_files = ml_variations + dl_variations
model_files

for g in range(fromngroups,tongroups+1):
    for m in model_files:
        groups[g][m] = pd.read_csv(
            f"{EXPERIMENTPATH}groups-{g}-{m}.csv",
            header=0,
            names=["data"]
        )['data']
#groups[4]


# +
# Generate diffs with mean

for g in range(fromngroups,tongroups+1):
    for m in model_files:
        groups[g]['diff_mean_'+m] = abs(groups[g]['mean'].subtract(groups[g][m],axis=0))
        groups[g]['diff_middle_'+m] = abs(groups[g]['middle'].subtract(groups[g][m],axis=0))

df_groups_diff_mean = groups[4].filter(regex="^diff_mean")
df_groups_diff_middle = groups[4].filter(regex="^diff_middle")
# -

pd.DataFrame(df_groups_diff_mean.mean().sort_values(), columns=['mean (diff_group_mean)'])

pd.DataFrame(df_groups_diff_middle.mean().sort_values(), columns=['mean (diff_group_middle)'])

"""
ANOVA
¿Diferencias entre modelos?
Factor: modelo
Variantes: todos los modelos
Variable respuesta: diff con media o con mediana
"""
import pingouin as pg
#import scipy.stats as stats


columns = df_groups_diff_mean.columns.to_list()+['g']
melted_data = groups[4][columns].melt(['g'], value_name='error', var_name='model')
melted_data
aov = pg.anova(data=melted_data, dv='error', between='model', detailed=True)
print(aov)

columns = df_groups_diff_middle.columns.to_list()+['g']
melted_data = groups[4][columns].melt(['g'], value_name='error', var_name='model')
aov = pg.anova(data=melted_data, dv='error', between='model', detailed=True)
print(aov)










