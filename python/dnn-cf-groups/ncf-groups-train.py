from data_utils import init_random
init_random()

import argparse
import sys
import os


text = 'Program to train a model with groups data. It uses a Dense as an embedding.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--outdir', type=str, required=True, help="Output directory")
parser.add_argument('--model', type=str, required=True, help="Model name, in ncf_models")
parser.add_argument('--seed', type=str, required=True, help="Seed")
parser.add_argument('--k', type=int, required=True, help="Number of factor in each embedding")
parser.add_argument('--dataset', type=str, required=True, help="Dataset")

parser.add_argument('--embacti', type=str, required=False, help="Dataset")

args = parser.parse_args()

if args.embacti:
    embedding_activation = args.embacti
    #embedding_activation = 'relu'
    #embedding_activation = 'tanh'
    #embedding_activation = 'sigmoid'
else:
    embedding_activation = None
    #embedding_activation = 'linear'

BATCH=64
EPOCH=25
steps_per_epoch = None

# NETFLIX
EPOCH=5
steps_per_epoch = 200000

# Seed inizialization
init_random(args.seed)


# Dataset load
from data_utils import dynamic_import
from data_groups import OneHotGenerator

DynamicClass = dynamic_import(args.dataset)
dataset = DynamicClass()
num_users = dataset.get_num_users()
num_items = dataset.get_num_items()
(x_train, x_val, y_train, y_val) = dataset.get_train_val()
train_secuencer = OneHotGenerator(x_train, y_train, num_users, num_items, BATCH)
val_secuencer = OneHotGenerator(x_val, y_val, num_users, num_items, BATCH)


# Output directory creation
outputdir=args.outdir+"/"+dataset.get_data_code()
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    print("The new directory %s is created!" % outputdir)


# Model creation
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

from ncf_models import get_model_list, get_model, store_model


model = get_model(args.model, args.k, dataset, args.seed, embedding_activation)

model.summary()
model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(lr=0.001)
    #optimizer=keras.optimizers.Nadam()
)

history = model.fit(
    train_secuencer,
    validation_data=val_secuencer,
    epochs=EPOCH,
    verbose=1,
    #use_multiprocessing=True,
    #workers=6
    steps_per_epoch=steps_per_epoch,
)

(x_test, y_test) = dataset.get_test()
test_secuencer = OneHotGenerator(x_test, y_test, num_users, num_items, BATCH)

results = model.evaluate(test_secuencer)

store_model(model, history, results, outputdir)