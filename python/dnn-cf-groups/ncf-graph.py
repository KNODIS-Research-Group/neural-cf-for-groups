import pandas as pd

# Carga de los ficheros
groups = {}

DATAPATH='../rs-data-python/grupos/ml1m-completeinfo/'
#EXPERIMENTPATH='groups_2022_04_good_index/ml1m/'
EXPERIMENTPATH='groups_2022_05_completeinfo/ml1m-completeinfo/'

# +
models = [
    #"pred-mfgu_af",
    #"pred-mfgu_bf",
    #"biasedmf-avg",
    #"mlp_ml1m_4_mean_indi",
    #"mlp_ml1m_4_trunc",
    #"mlp_ml1m_relu_10_mean_indi",
    #"mlp_ml1m_relu_10_trunc",
    #"boby_10_mean_indi",
    #"boby_10_trunc",
    #"evo_10_mean_indi",
    #"evo_10_trunc",
    
    #"biasedMF_k8_dsml1m_seed1234_indi_mean",
    #"biasedMF_k8_dsml1m_seed1234_indi_expert",
    #"mlp_k8_dsml1m_seed1234_indi_mean",
    #"mlp_k8_dsml1m_seed1234_indi_expert",
    #"mlp_k8_dsml1m_seed1234_0.000",
    #"mlp_k8_dsml1m_seed1234_0.250",
    #"mlp_k8_dsml1m_seed1234_1.000",
    #"mlp_k8_dsml1m_seed1234_expert",
    #"biasedmf-avg",
    #"pred-mfgu_af",
    #"pred-mfgu_bf",
    "biasedMF_k8_dsml1m_seed1234_indi_mean",
    "mlp_k8_dsml1m-completeinfo_seed1234_indi_mean",
    #"mlp_k8_dsml1m-completeinfo_seed1234_mo",
    

    "mlp_k8_dsml1m-completeinfo_seed1234_0.062",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.125",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.250",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.500",
    "mlp_k8_dsml1m-completeinfo_seed1234_1.000",
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_median", 
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_quantizied", 
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_quantizied_90", 
    
]
# -

fromngroups=2
tongroups=10
fromngroups=4
tongroups=4



for g in range(fromngroups,tongroups+1):
    groups[g] = pd.read_csv(f"{DATAPATH}groups-{g}.csv")
    groups[g]['g'] = str(g)
    for m in models:
        groups[g][m] = pd.read_csv(
            f"{EXPERIMENTPATH}groups-{g}-{m}.csv",
            header=0,
            names=["data"]
        )['data']

print("Tgrupo\tNaN\tNVotos\tPercen")
for g in range(fromngroups,tongroups+1):
    total_nan = groups[g].filter(regex="^rating-").isnull().sum(axis = 1).sum()
    print(g, "\t", total_nan, "\t", len(groups[g].index) * g, "\t", total_nan / (len(groups[g].index) * g))

from lets_plot import *
LetsPlot.setup_html()

"""
import math

def manual_mean(df,column_name, fromindex, toindex):
    for irow, _ in df.iterrows():
        totalsum = 0
        numnofnan = 0
        for icolumn in range(fromindex, toindex+1):
            val_rating = df.at[irow,'rating-'+str(icolumn)]
            if(math.isnan(val_rating)):
               abserror = math.nan
            else:
               val_test = df.at[irow,column_name]
               abserror = abs(val_rating-val_test)
               numnofnan = numnofnan + 1
               totalsum = totalsum + abserror
            df.at[irow,column_name+'-manual-error-'+str(icolumn)] = abserror
        df.at[irow,column_name+'-manual-error-MAE'] = totalsum/numnofnan

for g in range(fromngroups,tongroups+1):    # Each group
    for m in models:                        # Each model
        manual_mean(groups[g], m, 1, g)
"""

for g in range(fromngroups,tongroups+1):    # Each group
    for m in models:                        # Each model
        for gi in range(1,g+1):             # Each rating in group
            groups[g][m+'-error-'+str(gi)] = abs(groups[g]['rating-'+str(gi)] - groups[g][m])

for g in range(fromngroups,tongroups+1):    # Each group
    for m in models:                        # Each model
        groups[g][m+'-MAE'] = groups[g].filter(regex='^'+m+'-error-',axis=1).mean(axis=1, skipna=True)

"""

pd.set_option('display.max_columns', None)
print(groups[4][["biasedmf-avg-MAE","biasedmf-avg-manual-error-MAE"]])
print(groups[4][["mlp_k8_dsml1m_seed1234_indi_mean-MAE","mlp_k8_dsml1m_seed1234_indi_mean-manual-error-MAE"]])
print(groups[4][["mlp_k8_dsml1m_seed1234_mo-MAE","mlp_k8_dsml1m_seed1234_mo-manual-error-MAE"]])

"""

"""

print(groups[4][["biasedmf-avg-MAE","biasedmf-avg-manual-error-MAE"]].describe())
print(groups[4][["mlp_k8_dsml1m_seed1234_indi_mean-MAE","mlp_k8_dsml1m_seed1234_indi_mean-manual-error-MAE"]].describe())
print(groups[4][["mlp_k8_dsml1m_seed1234_mo-MAE","mlp_k8_dsml1m_seed1234_mo-manual-error-MAE"]].describe())


"""

all_data = pd.concat(groups, join='inner', ignore_index=True).drop(
    ['item', 'user-1', 'rating-1', 'user-2', 'rating-2'], axis=1
)


#groups[4][mean].describe().to_csv("lets-plot-images/info.csv")
"""
def generate_svg(data, column_list, file_name="test"):
    data_to_plot = data[column_list].melt(['g'], value_name='error', var_name='model')

    plt = ggplot(
        data_to_plot, 
        aes(x='g', y='error', color='model')
    ) + geom_boxplot() + \
        ggsize(800, 600) + \
        geom_boxplot() + \
        theme(axis_title_y='blank') + \
        ggtitle('Mean Absolute Error') + \
        xlab('Group size')

    ggsave(filename=file_name+'.svg', plot=plt)

generate_svg(all_data, max_difs, 'max_difs')
"""



all_data

all_data['g'] = pd.to_numeric(all_data['g'])

# +
models_in_ploitline = [
    "biasedmf-avg-MAE",
    #"biasedMF_k8_dsml1m_seed1234_indi_mean-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_indi_mean-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_mo-MAE",
    

    "mlp_k8_dsml1m-completeinfo_seed1234_0.062-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.125-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.250-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.500-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_1.000-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_indi_mean-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_median-MAE", 
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_quantizied-MAE", 
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_quantizied_90-MAE", 
]

processed = all_data.groupby("g", as_index=False).mean()[['g']+models_in_ploitline].melt(['g'], value_name='error', var_name='model')
# -

processed

# +
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,8))
sns.lineplot(data=processed, x="g", y="error", hue="model")
# -
all_data

# +
names_to_box_plot = [
    #"biasedMF_k8_dsml1m_seed1234_indi_mean-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_indi_mean-MAE",
    #"mlp_k8_dsml1m-completeinfo_seed1234_mo-MAE",
    

    "mlp_k8_dsml1m-completeinfo_seed1234_0.062-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.125-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.250-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_0.500-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_1.000-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_indi_mean-MAE",
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_median-MAE", 
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_quantizied-MAE", 
    "mlp_k8_dsml1m-completeinfo_seed1234_virtual_quantizied_90-MAE", 
]

to_boxplot = all_data[names_to_box_plot].melt(value_name='error', var_name='model')


from lets_plot import *
LetsPlot.setup_html()

plt = ggplot(
        to_boxplot, 
        aes(x='model', y='error', color='model')
    ) + geom_boxplot() + \
        ggsize(1000, 1000) + \
        theme(axis_title_y='blank') + \
        ggtitle('Mean Absolute Error') + \
        xlab('Group size')

plt.show()
# -


all_data[names_to_box_plot].describe()

all_data.iloc[2731]


