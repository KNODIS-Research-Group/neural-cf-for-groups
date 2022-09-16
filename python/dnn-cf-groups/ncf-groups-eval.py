from data_utils import init_random
init_random()

import argparse
import sys
import os
from pathlib import Path

text = 'Program to predict group data.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--m', type=str, required=True, help="Model path")

# Netflix is a very big DS. You can specify group size.
parser.add_argument('--groupsize', type=int, required=False, help="Group size")

args = parser.parse_args()

print(args.groupsize)

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
    elif path.find("/netflix/") != -1:
        return "data_groups.GroupDataNetflix"
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

"""
(x_test, y_test) = dataset.get_test()
test_secuencer = OneHotGenerator(x_test, y_test, dataset.get_num_users(), dataset.get_num_items(), BATCH)
write_file(outdir,0,modelname+'_test',model.predict(test_secuencer))
"""

"""
    GROUP experiments
"""
if args.groupsize is None:
    print("All groups")
    fromngroups=2
    tongroups=10
else:
    print("Group size:" + str(args.groupsize))
    fromngroups = args.groupsize
    tongroups = args.groupsize

#fromngroups=4
#tongroups=4

import gc


for i, ngrp in enumerate(range(fromngroups,tongroups+1)):
    """
        BiasedMF
    """
    # Mean Individuals
    #biased_model_name='biasedMF_k8_dsml1m_seed1234'
    
    #dd = pd.read_csv(f'{outdir}/groups-{ngrp}-{biased_model_name}_indi.csv', header=0, names=[i for i in range(1,ngrp)])
    #biasedmf = dd.to_numpy()
    
    # Mean Individuals
    #y_pred = np.mean(biasedmf, axis=1)
    #y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    #write_file(outdir, ngrp, biased_model_name+'_indi_mean', y_pred, index=False)
    
    
    """
        DL
    """
    # Expert Mean
    #group_rating_table = dataset.get_rating_count_table_for_group(ngrp)
    #group_rating_table_row_sum = np.sum(group_rating_table, axis=1)
    #group_multiply_factor = group_rating_table/group_rating_table_row_sum.reshape(-1,1)
    #y_pred = np.sum(biasedmf * group_multiply_factor, axis=1)
    #y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    #write_file(outdir, ngrp, biased_model_name+'_indi_expert', y_pred, index=False)
    
    
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

    # Expert Mean
    """
    group_rating_table = dataset.get_rating_count_table_for_group(ngrp)
    group_rating_table_row_sum = np.sum(group_rating_table, axis=1)
    group_rating_table_row_sum[group_rating_table_row_sum==0]=1
    group_multiply_factor = group_rating_table/group_rating_table_row_sum.reshape(-1,1)
    y_pred = np.sum(y_pred_as_individual * group_multiply_factor, axis=1)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir, ngrp, modelname+'_indi_expert', y_pred, index=False)
    del group_rating_table
    del group_rating_table_row_sum
    """
    
    del y_pred
    del y_pred_as_individual
    gc.collect()
    
    """
    print(group_rating_table)
    print(group_rating_table_row_sum)
    print(group_multiply_factor)
    print(y_pred_as_individual * group_multiply_factor)
    print(y_pred)
    print(np.sum(y_pred_as_individual * group_multiply_factor, axis=1))
    """
    
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
        #write_file(outdir,ngrp, modelname+"_"+sss, y_pred)
        write_file(outdir,ngrp, modelname+"_mo", y_pred)
        del test_data
        gc.collect()
    
    # Experts model

    """
    test_data = dataset.get_test_for_group_size_with_expert_info(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_mo_expert", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    test_data = dataset.get_test_for_group_size_with_max(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_max_expert", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    test_data = dataset.get_test_for_group_size_with_min(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_min_expert", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    test_data = dataset.get_test_for_group_size_with_inverse_expert_info(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_inverse_expert_info", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    test_data = dataset.get_test_for_group_size_with_raro(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_raro", y_pred)
    del test_data
    del y_pred
    gc.collect()
    """
    
    test_data = dataset.get_test_for_group_size_with_softmax(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_softmax", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    
    test_data = dataset.get_test_for_group_size_with_max(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_max", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    
    """
    test_data = dataset.get_test_for_group_size_with_inverse_softmax(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_inverse_softmax", y_pred)
    del test_data
    del y_pred
    gc.collect()
    
    test_data = dataset.get_test_for_group_size_with_softmax_raro(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_softmax_raro", y_pred)
    del test_data
    del y_pred
    gc.collect()
    """
