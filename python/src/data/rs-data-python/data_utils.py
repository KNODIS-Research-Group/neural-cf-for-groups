import random
import numpy as np
import tensorflow as tf
from numpy.random import seed

def get_seeds():
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    #return [2, 3, 5, 7, 11]
    #return [13, 17, 19, 23, 29]
    #return [31, 37, 41, 43, 47]
    #return [53, 59, 61, 67, 71]
    
    #return [2]
    #return [41, 43, 47, 53, 59, 61, 67, 71]

def init_random(val=37):
    random.seed(int(val))
    seed(int(val))
    if tf.__version__.startswith('1'):
        tf.set_random_seed(int(val))
    else:
        tf.random.set_seed(int(val))


"""
import from string
https://stackoverflow.com/a/547867/932888

from data_utils import dynamic_import
DynamicClass = dynamic_import("data_groups.GroupDataOneHotML")
dataset = DynamicClass()

"""
def dynamic_import(name):
    components = name.split('.')
    module_string = ".".join(components[:-1])
    mod = __import__(module_string)
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod