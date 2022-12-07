from data_utils import init_random
init_random()

import argparse
import sys
import os

"""

python src/train/train.py --outdir {outdir} --model {model} --seed {seed} --k {k} --dataset {args.dataset}"

"""
text='Train a models'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--outdir', type=str, required=True, help="Output directory")
parser.add_argument('--model', type=str, required=True, help="Model")
parser.add_argument('--seed', type=str, required=True, help="seed")
parser.add_argument('--k', type=str, required=True, help="k")
parser.add_argument('--dataset', type=str, required=True, help="Dataset")
args = parser.parse_args()


# Dataset and seed
from src.data.data import code_to_py
from data_utils import dynamic_import, init_random, get_seeds
from data_groups import OneHotGenerator

dsimport = code_to_py(args.dataset)
init_random(args.seed) # Before dataset load, shuffle and split

DynamicClass = dynamic_import(dsimport)
dataset = DynamicClass()


# Outdir
import json
import pprint
import os
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    print("The new directory %s is created!" % args.outdir)

args.outdir+='/'+dataset.get_data_code()
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    print("The new directory %s is created!" % args.outdir)



BATCH=64
EPOCH=20

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

from src.models.models import get_model, store_model


num_users = dataset.get_num_users()
num_items = dataset.get_num_items()


print(f"Model {args.model}")

model = get_model(args.model, args.k, dataset, args.seed)

model.summary()
model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(lr=0.001)
    #optimizer=keras.optimizers.Nadam()
)

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

store_model(model, history, results, args.outdir)
