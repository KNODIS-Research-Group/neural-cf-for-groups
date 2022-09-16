OUTDIR='results_ml1m'
DATASET = "data_groups.GroupDataML1M"

#OUTDIR='results_ft'
#DATASET = "data_groups.GroupDataFT"

#OUTDIR='results_anime'
#DATASET = "data_groups.GroupDataANIME"

BATCH=64

import os
import pathlib
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd

model_path_list = pathlib.Path(OUTDIR).glob('*.h5')
model_list = []

for model_path in model_path_list:
    model_list.append(model_path)


from data_groups import OneHotGenerator
from data_utils import dynamic_import
DynamicClass = dynamic_import(DATASET)
dataset = DynamicClass()

dl_data_summary = pd.DataFrame({
                                    'model': pd.Series(dtype='str'),
                                    'exp': pd.Series(dtype='int'),
                                    'mean': pd.Series(dtype='float'),
                                    'median': pd.Series(dtype='float'),
                                    'std': pd.Series(dtype='float'),
                                    'max': pd.Series(dtype='float'),
                                    'min': pd.Series(dtype='float')
})

import re

regex = r"(.*)_(\d+)\.h5$"
compiled_regexp = re.compile(regex)

for model_path in model_list:
    print (model_path)
    file_name = model_path.parts[-1]
    m = compiled_regexp.match(file_name)
    if m:
        model_name = m.group(1)
        exp = m.group(2)
    else:
        raise Exception('Error in model name')
    
    model = keras.models.load_model(model_path, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.2)})
    (x_test, y_test) = dataset.get_test()
    test_secuencer = OneHotGenerator(x_test, y_test, dataset.get_num_users(), dataset.get_num_items(), BATCH)
    predictions = model.predict(test_secuencer)
    errors = abs(predictions - y_test)
    
    dl_data_summary = dl_data_summary.append({
                                                'model': model_name,
                                                'exp': int(exp),
                                                'mean': np.mean(errors),
                                                'median': np.median(errors),
                                                'std': np.std(errors),
                                                'max': errors.max(),
                                                'min': errors.min()
                                            }, ignore_index=True)
    

dl_data_summary = dl_data_summary.sort_values(by=['model','exp'])
dl_data_summary.to_csv(OUTDIR+'_summary.csv', index=False)