from data_utils import init_random
init_random()

import argparse
import sys
import os
from pathlib import Path

text = 'Program to evaluate group data.'

parser = argparse.ArgumentParser(description=text)
parser.add_argument('--m', type=str, required=True, help="Model path")
args = parser.parse_args()

OUTDIR='results'

from src.data.data import code_to_py

# Dataset loading
def select_dataset(path):
    if path.find("/anime/") != -1:
        return code_to_py("anime")
    elif path.find("/ft/") != -1:
        return code_to_py("ft")
    elif path.find("/ml1m/") != -1:
        return code_to_py("ml1m")
    else:
        raise "Wrong dataset"


from data_groups import OneHotGenerator
from data_utils import dynamic_import
DynamicClass = dynamic_import(select_dataset(args.m))
dataset = DynamicClass()


outdir=OUTDIR+'/'+dataset.get_data_code()
if not os.path.exists(outdir):
    os.makedirs(outdir)
    print("The new directory %s is created!" % outdir)


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

BATCH=64


"""
    GROUP experiments
"""

#fromngroups=4
#tongroups=4

fromngroups=2
tongroups=10


"""
Experiments

GMF IPA - avg
GMF Expertise
GMF Softmax
MLP IPA
MLP avg - DeepGroup
MLP Expertise
MLP Softmax


Per model

IPA
\-mean

GPA
\- avg
\- Expertise
\- Softmax

"""

from multihot_activations import get_activation_expert, get_activation_softmax


for i, ngrp in enumerate(range(fromngroups,tongroups+1)):
    
    """
        IPA
    """
    # Expand group as individual
    test_data = dataset.get_group_test_as_individuals(ngrp, BATCH)
    
    # Predict all
    y_pred_as_individual = model.predict(test_data)
    y_pred_as_individual = y_pred_as_individual.reshape((-1, ngrp))
    # [[predict_rating1, predict_rating2, ..., predict_ratingn], [predict_rating1, ...] ... ]
    # write_file(outdir, ngrp, modelname+'_indi', y_pred_as_individual, index=False)
    
    # Mean Individuals
    y_pred = np.mean(y_pred_as_individual, axis=1)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir, ngrp, modelname+'_ipa_mean', y_pred, index=False)
    
    """
        GPA
    """
    
    # Dataframe.mean is not used here.
    test_data = dataset.get_group_test(ngrp, BATCH, pd.DataFrame.mean, 1/ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir,ngrp, modelname+"_gpa_mean", y_pred)
    
    
    """
        Expert
    """
    expert_closure = get_activation_expert(
                        dataset.get_num_users(),
                        dataset.get_num_items(),
                        ngrp,
                        dataset.get_rating_count()
                    )
    test_data = dataset.get_group_test(ngrp, BATCH, pd.DataFrame.mean, expert_closure)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir,ngrp, modelname+"_gpa_expert", y_pred)
    
    
    """
        Expert
    """
    softmax_closure = get_activation_softmax(
                        dataset.get_num_users(),
                        dataset.get_num_items(),
                        ngrp,
                        dataset.get_rating_count()
                    )
    test_data = dataset.get_group_test(ngrp, BATCH, pd.DataFrame.mean, softmax_closure)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir,ngrp, modelname+"_gpa_softmax", y_pred)
    
    
    
    """
    test_data = dataset.get_test_for_group_size_with_softmax(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_softmax", y_pred)

    
    
    test_data = dataset.get_test_for_group_size_with_max(ngrp)
    y_pred = model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    write_file(outdir,ngrp, modelname+"_max", y_pred)
    """
    
    
    
    
    
    
    
    
    # Expert Mean
    #group_rating_table = dataset.get_rating_count_table_for_group(ngrp)
    #group_rating_table_row_sum = np.sum(group_rating_table, axis=1)
    #group_multiply_factor = group_rating_table/group_rating_table_row_sum.reshape(-1,1)
    #y_pred = np.sum(biasedmf * group_multiply_factor, axis=1)
    #y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    #write_file(outdir, ngrp, biased_model_name+'_indi_expert', y_pred, index=False)
    
    
    

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
    
    """
    print(group_rating_table)
    print(group_rating_table_row_sum)
    print(group_multiply_factor)
    print(y_pred_as_individual * group_multiply_factor)
    print(y_pred)
    print(np.sum(y_pred_as_individual * group_multiply_factor, axis=1))
    """

    
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
