from data_utils import init_random
init_random()
import argparse
import sys
import os
from pathlib import Path
from data_groups import OneHotGenerator
from data_utils import dynamic_import
DynamicClass = dynamic_import("data_groups.GroupDataML1M")
dataset = DynamicClass()


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import csv

#model_path="evodeep_k8_dsml1m_seed1234.h5"
model_path="evodeep_k8_dsml1m_seed1234_embacti_relu.h5"
#model_path="evodeep_k8_dsml1m_seed1234_embacti_sigmoid.h5"
#model_path="mlp_k8_dsml1m_seed1234.h5"
model = keras.models.load_model(model_path)

import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)
from ncf_virtual_users_functions import extract_embeddings_outputs, get_embeddings, get_fromembeddings_model

(userembeddinglayer, itemembeddinglayer, userembeddinglayer2, itemembeddinglayer2) = get_embeddings(model)
(user_embeddings, item_embeddings) = extract_embeddings_outputs(model, userembeddinglayer, itemembeddinglayer, dataset)

"""
user_embeddings[0] -> User EmbeddingOut of -> 1 0 0 0 0 ... 0 0 0 0 0 0 
item_embeddings[0] -> Item EmbeddingOut of -> 0 0 0 0 0 ... 0 1 0 0 0 0  (First one of first item, nuser offset)
"""


virtual_model = get_fromembeddings_model(model, userembeddinglayer, itemembeddinglayer, userembeddinglayer2, itemembeddinglayer2)

"""
test
"""
for nu in range(0,5):
    for ni in range(0,5):
        print("User " + str(nu) + " Item " + str(ni))
        test_input = np.zeros((1, dataset.get_num_users()+dataset.get_num_items()), dtype=np.int32)
        test_input[0][nu]=1
        test_input[0][dataset.get_num_users()+ni]=1

        print(model.predict(test_input))
        print(virtual_model.predict([user_embeddings[nu:nu+1],item_embeddings[ni:ni+1]]))


"""
Output


User 0 Item 0
[[3.0008833]]
[[3.0008833]]
User 0 Item 1
[[3.0024865]]
[[3.0024865]]
User 0 Item 2
[[4.000348]]
[[4.000348]]
User 0 Item 3
[[4.000847]]
[[4.000847]]
User 0 Item 4
[[2.8024366]]
[[2.8024366]]
User 1 Item 0
[[3.9997838]]
[[3.9997838]]
User 1 Item 1
[[4.000439]]
[[4.000439]]
User 1 Item 2
[[4.1892676]]
[[4.1892676]]
User 1 Item 3
[[4.999479]]
[[4.999479]]
User 1 Item 4
[[3.004635]]
[[3.004635]]
User 2 Item 0
[[2.9912896]]
[[2.9912896]]
User 2 Item 1
[[2.9993594]]
[[2.9993594]]
User 2 Item 2
[[3.8732188]]
[[3.8732188]]
User 2 Item 3
[[4.000284]]
[[4.000284]]
User 2 Item 4
[[1.9986773]]
[[1.9986773]]
User 3 Item 0
[[3.0070648]]
[[3.0070648]]
User 3 Item 1
[[3.9996974]]
[[3.9996974]]
User 3 Item 2
[[4.9988766]]
[[4.9988766]]
User 3 Item 3
[[4.9980497]]
[[4.9980497]]
User 3 Item 4
[[2.744315]]
[[2.744315]]
User 4 Item 0
[[2.9994137]]
[[2.9994137]]
User 4 Item 1
[[3.000908]]
[[3.000908]]
User 4 Item 2
[[4.0003796]]
[[4.0003796]]
User 4 Item 3
[[4.0005274]]
[[4.0005274]]
User 4 Item 4
[[1.9977582]]
[[1.9977582]]

"""