import os
import math

import numpy as np
import pandas as pd
from keras.utils import Sequence

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, ReadTrainMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitValMixin, GetNumsByMaxMixin
from data_onehot import dense_embedding, dense_embedding_for_group, dense_embedding_for_group_with_closure


"""
Exampled in: https://keras.io/api/utils/python_utils/#sequence-class
fit and evaluate with secuencer
"""
class OneHotGenerator(Sequence):

    def __init__(self, x, y, num_users, num_items, batch_size):
        self.x, self.y = x, y
        self.num_users, self.num_items = num_users, num_items
        self.batch_size = batch_size
        self.calculated_len = len(x)

    def __len__(self):
        return math.ceil(self.calculated_len / float(self.batch_size))

    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size

        batch_x = self.x[idx_start:idx_end]
        batch_y = self.y[idx_start:idx_end]

        return np.array(dense_embedding(batch_x, self.num_users, self.num_items)), np.array(batch_y)


class MultiHotGenerator(Sequence):

    def __init__(self,
                    group_size,
                    path_group_file,
                    num_users,
                    num_items,
                    batch_size,
                    y_aggregation_func=pd.DataFrame.mean,
                    multihot_activation=None
                ):
        
        #TODO: REFACTORREVIEW
        names = ["g", "i"]
        for u in range(1, group_size+1):
            names.append("u"+str(u))
            names.append("r"+str(u))
        group_data = pd.read_csv(
            path_group_file,
            header=0,
            names=names
        )
        #TODO: REFACTORREVIEW
        group_data = group_data.drop("g", axis=1)
        
        if multihot_activation == None:
            self.multihot_activation = 1 / group_size
        else:
            self.multihot_activation = multihot_activation 
        
        self.data = group_data
        self.group_size = group_size
        self.num_users, self.num_items = num_users, num_items
        self.batch_size = batch_size
        self.calculated_len = len(self.data)
        self.y_aggregation_func = y_aggregation_func
        if self.y_aggregation_func == pd.DataFrame.mode:
            self.y = group_data.filter(regex='^r').apply(lambda x: x.mode(),axis=1).mean(axis=1,skipna=True)
        else:
            self.y = group_data.filter(regex='^r').apply(self.y_aggregation_func,axis=1)

    def __len__(self):
        return math.ceil(self.calculated_len / float(self.batch_size))

    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size

        batch_data = self.data[idx_start:idx_end]
        
        # Activation
        if isinstance(self.multihot_activation, float):
            batch_x = dense_embedding_for_group(batch_data.to_numpy(), self.num_users, self.num_items, self.group_size, self.multihot_activation)
        else:
            batch_x = dense_embedding_for_group_with_closure(batch_data.to_numpy(), self.num_users, self.num_items, self.group_size, self.multihot_activation)
        
        # Y Aggregation
        #batch_y = batch_data.filter(regex='^r').mean(axis=1)
        #batch_y = batch_data.filter(regex='^r').median(axis=1)
        #All work but 'mode'. Wired behaviour with serie. 
        # I need to mean when several values of mode occurs
        if self.y_aggregation_func == pd.DataFrame.mode:
            batch_y = batch_data.filter(regex='^r').apply(lambda x: x.mode(),axis=1).mean(axis=1,skipna=True) # Mode then Mean... may be several values
        else:   
            batch_y = batch_data.filter(regex='^r').apply(self.y_aggregation_func,axis=1)
        
        return np.array(batch_x), np.array(batch_y)


class OneHotGeneratorAsIndividual(MultiHotGenerator):
    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size

        batch_data = self.data[idx_start:idx_end]
        
        as_individual = []
        y = []
        batch_data = (np.asarray(batch_data))
        #TODO: REFACTORREVIEW (not modified now but is related with above TODO)
        for idx, row in enumerate(batch_data): # i, u1, r1, u2, r2, etc...
            item = row[0] # i
            for u in range(0, self.group_size):
                user = row[1+(u*2)]
                as_individual.append([user, item])
                y.append(row[1+(u*2)+1])

        assert(len(as_individual) == (len(batch_data) * self.group_size))

        as_individual = np.asarray(as_individual, dtype=np.int64)
        as_individual = dense_embedding(as_individual, self.num_users, self.num_items)
        
        batch_x = as_individual
        batch_y = y

        return np.array(batch_x), np.array(batch_y)