OUTDIR='results_ml1m'
DATASET = "data_groups.GroupDataML1M"

#OUTDIR='results_ft'
#DATASET = "data_groups.GroupDataFT"

#OUTDIR='results_anime'
#DATASET = "data_groups.GroupDataANIME"


K = 8
BATCH=64
EPOCH=20


import json
import pprint
import os
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
    print("The new directory %s is created!" % OUTDIR)


from data_utils import dynamic_import, init_random, get_seeds
from data_groups import OneHotGenerator

init_random() # Before dataset load, shuffle and split
seeds = get_seeds()

DynamicClass = dynamic_import(DATASET)
dataset = DynamicClass()

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

from ncf_models import get_model_list, get_model, store_model

model_list = get_model_list()

num_users = dataset.get_num_users()
num_items = dataset.get_num_items()

for seed in get_seeds():
    for model_name in model_list:
        print(f"Model {model_name}")
        
        model = get_model(model_name, K, dataset, seed)
        model.summary()
        model.compile(
            loss=tf.keras.losses.MeanAbsoluteError(),
            #optimizer=keras.optimizers.Adam(lr=0.001)
            optimizer=keras.optimizers.Nadam()
        )
        """model.compile(
            #optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )"""
        
        (x_train, x_val, y_train, y_val) = dataset.get_train_val()
        
        train_secuencer = OneHotGenerator(x_train, y_train, num_users, num_items, BATCH)
        val_secuencer = OneHotGenerator(x_val, y_val, num_users, num_items, BATCH)
        
        history = model.fit(
            train_secuencer,
            validation_data=val_secuencer,
            epochs=EPOCH,
            verbose=1
        )
        
        (x_test, y_test) = dataset.get_test()
        test_secuencer = OneHotGenerator(x_test, y_test, num_users, num_items, BATCH)
        
        results = model.evaluate(test_secuencer)
        
        store_model(model, history, results, OUTDIR)
