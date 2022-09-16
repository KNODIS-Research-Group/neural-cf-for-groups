from data_utils import init_random
init_random()

import argparse
import sys
import os
from pathlib import Path

text = 'Program to predict group data.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--m', type=str, required=True, help="Model path")

args = parser.parse_args()

# Dataset loading
def select_dataset(path):
    if path.find("/anime/") != -1:
        return "data_groups.GroupDataANIME"
    elif path.find("/ft/") != -1:
        return "data_groups.GroupDataFT"
    elif path.find("/ml/") != -1:
        return "data_groups.GroupDataML"
    elif path.find("/ml1m/") != -1:
        return "data_groups.GroupDataML1M"
    elif path.find("/ml1m-completeinfo/") != -1:
        return "data_groups.GroupDataML1MCompleteInfo"
    else:
        raise "Wrong dataset"


from data_groups import OneHotGenerator
from data_utils import dynamic_import
DynamicClass = dynamic_import(select_dataset(args.m))
dataset = DynamicClass()


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import csv
"""
Write prediction (data) in file 
"""
def write_file(dir, gsize, modelname, data, index=True):
    f=open(f'{dir}/groups-{gsize}-{modelname}.csv', 'w')
    f.write(f'{dir}/groups-{gsize}-{modelname}'+"\n")
    for r in data:
        if index:
            f.write(str(r[0])+"\n")
        else:
            f.write(str(r)+"\n")
    f.close()


model = keras.models.load_model(args.m)

modelfilepath = Path(args.m)
modelname = str(modelfilepath.stem)
outdir = str(modelfilepath.parent)

# Prediction on test
BATCH=64
(x_test, y_test) = dataset.get_test()
test_secuencer = OneHotGenerator(x_test, y_test, dataset.get_num_users(), dataset.get_num_items(), BATCH)
write_file(outdir,0,modelname+'_test',model.predict(test_secuencer))


"""
    GROUP experiments
"""
fromngroups=2
tongroups=10

fromngroups=4
tongroups=4


from ncf_virtual_users_functions import extract_embeddings_outputs, get_embeddings, get_fromembeddings_model

DATAPATH='../rs-data-python/grupos/ml1m-completeinfo/'
K=8

for i, ngrp in enumerate(range(fromngroups,tongroups+1)):
    groups_data = pd.read_csv(f"{DATAPATH}groups-{ngrp}.csv")
    
    (userembeddinglayer, itemembeddinglayer, userembeddinglayer2, itemembeddinglayer2) = get_embeddings(model)
    (user_embeddings, item_embeddings) = extract_embeddings_outputs(model, userembeddinglayer, itemembeddinglayer, dataset)
    
    #np.save(f"{outdir}/embedings_dump.dat", user_embeddings)
    #print(user_embeddings)
    
    #exit
    items_ids = groups_data['item'].to_numpy()
    items_embeddings = np.zeros((len(items_ids),K))
    
    for ix, item_id in enumerate(items_ids):
        items_embeddings[ix] = item_embeddings[item_id]
    
    userrows = groups_data.filter(regex='^user-',axis=1).to_numpy()
    virtualuser_embeddings_quantizied = np.zeros((len(userrows),K))
    virtualuser_embeddings_median = np.zeros((len(userrows),K))
    
    E_values_25 = np.percentile(user_embeddings, 10, axis=0)
    E_values_50 = np.percentile(user_embeddings, 50, axis=0)
    E_values_75 = np.percentile(user_embeddings, 90, axis=0)
    
    
    for row_x, row in enumerate(userrows):
        row_embeddings = np.zeros((len(row),K))
        for uin, uid in enumerate(row):
            row_embeddings[uin] = user_embeddings[uid]
        
        row_embeddings_comparations = row_embeddings.copy()
        row_embeddings_comparations[row_embeddings > E_values_50] = 1
        row_embeddings_comparations[row_embeddings == E_values_50] = 0 # never 
        row_embeddings_comparations[row_embeddings < E_values_50] = -1
        row_embeddings_comparations_sum = row_embeddings_comparations.sum(axis=0)
        
        for K_i in range(K):
            if row_embeddings_comparations_sum[K_i] > 0:
                virtualuser_embeddings_quantizied[row_x][K_i] = E_values_75[K_i]
            elif row_embeddings_comparations_sum[K_i] < 0:
                virtualuser_embeddings_quantizied[row_x][K_i] = E_values_25[K_i]
            else:
                virtualuser_embeddings_quantizied[row_x][K_i] = E_values_50[K_i]

        """
        print(row_embeddings_comparations_sum)
        print(virtualuser_embeddings_quantizied[row_x])
        print(E_values_25)
        print(E_values_50)
        print(E_values_75)
        print("*" * 20)
        exit
        """
        virtualuser_embeddings_median[row_x] = np.median(row_embeddings,axis=0).reshape(-1,8)
    
    
    virtual_model = get_fromembeddings_model(model, userembeddinglayer, itemembeddinglayer, None, None)
    y_pred = virtual_model.predict([virtualuser_embeddings_quantizied, items_embeddings])
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_virtual_quantizied_90", y_pred)
    
    y_pred = virtual_model.predict([virtualuser_embeddings_median, items_embeddings])
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_virtual_median", y_pred)

"""
for i, ngrp in enumerate(range(fromngroups,tongroups+1)):
    # Mean Individuals
    biased_model_name='biasedMF_k8_dsml1m_seed1234'
    # TODO: More groups
    dd = pd.read_csv(f'{outdir}/groups-{ngrp}-{biased_model_name}_indi.csv', header=0, names=[1,2,3,4])
    biasedmf = dd.to_numpy()
    
    # Mean Individuals
    y_pred = np.mean(biasedmf, axis=1)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir, ngrp, biased_model_name+'_indi_mean', y_pred, index=False)
    
    # Expand group as individual
    test_data = dataset.get_test_for_group_size_as_individual(ngrp)
    # Predict all
    y_pred_as_individual = model.predict(test_data)
    y_pred_as_individual = y_pred_as_individual.reshape((-1, ngrp))
    # [[predict_rating1, predict_rating2, ..., predict_ratingn], [predict_rating1, ...] ... ]
    write_file(outdir, ngrp, modelname+'_indi', y_pred_as_individual, index=False)
    
    # Mean Individuals
    y_pred = np.mean(y_pred_as_individual, axis=1)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir, ngrp, modelname+'_indi_mean', y_pred, index=False)
    
    #one_hot_activation = 1.0
    #one_hot_activation = 1.0 / group_size
    #one_hot_activation = 1.0 / group_size**2
    activations_oh = [
        #1.0,
        #(1.0/ngrp)*2,
        1.0/ngrp,
        #1.0/(ngrp*2),
        #1.0/(ngrp**2),
        #1.0/(ngrp**3),
        #1.0/(ngrp**4),
        #0.0,
    ]
    #pd.set_option('display.max_columns', None)
    for aoh in activations_oh:
        sss = format(aoh, '.3f')
        test_data = dataset.get_test_for_group_size(ngrp, aoh)
        y_pred = model.predict(test_data)
        y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
        write_file(outdir,ngrp, modelname+"_"+sss, y_pred)
    
"""