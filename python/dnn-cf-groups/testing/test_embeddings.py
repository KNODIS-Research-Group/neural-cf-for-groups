#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:02:40 2021

@author: jorge
"""

import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(
    input_dim=10,
    output_dim=3,
    input_length=1)
)

le = tf.keras.layers.Embedding(
    input_dim=10,
    output_dim=3,
    input_length=1)


input_array = np.random.randint(10, size=(32, 1))

model.compile('rmsprop', 'mse')

model.summary()

output_array = model.predict(input_array)
print(output_array.shape)