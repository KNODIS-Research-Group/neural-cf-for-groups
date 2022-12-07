import os
import math

import numpy as np
import pandas as pd
from keras.utils import Sequence

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, ReadTrainMixin, ExpertInfoMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitValMixin, GetNumsByMaxMixin
from data_onehot import dense_embedding, dense_embedding_for_group
from data_groups import MultiHotGenerator, OneHotGeneratorAsIndividual


DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/data"


class GroupData(ReadTestMixin, ReadTrainMixin, ExpertInfoMixin, ShuffleMixin, SplitValMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):

    def __init__(self, *args, **kwargs):
        super(GroupData, self).__init__(*args, **kwargs)
    
    def get_shape(self):
        return (self.get_num_users() + self.get_num_items(),) # OneHot Dense as Embedding
    
    def get_data_root(self):
        return DATA_ROOT
    
    def __get_generator(self, group_size, batch_size, path, agg_function, activation_function):
        return MultiHotGenerator(
                    group_size,
                    path,
                    self.get_num_users(),
                    self.get_num_items(),
                    batch_size,
                    agg_function,
                    activation_function
                )

    def get_group_test(self, group_size, batch_size, agg_function, activation_function):
        return self.__get_generator(group_size, batch_size, DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+".csv", agg_function, activation_function)
    
    def get_group_test_as_individuals(self, group_size, batch_size):
        return OneHotGeneratorAsIndividual(
                    group_size,
                    DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+".csv",
                    self.get_num_users(),
                    self.get_num_items(),
                    batch_size
                )


class GroupDataML1M(GroupData):
    test_url = "/grupos/ml1m/test-ratings.csv"
    train_url = "/grupos/ml1m/training-ratings.csv"
    code = "ml1m"


class GroupDataFT(GroupData):
    test_url = "/grupos/ft/test-ratings.csv"
    train_url = "/grupos/ft/training-ratings.csv"
    code = "ft"


class GroupDataANIME(GroupData):
    test_url = "/grupos/anime/test-ratings.csv"
    train_url = "/grupos/anime/training-ratings.csv"
    code = "anime"


def code_to_py(code):
    if code == 'ft':
        return 'src.data.data.GroupDataFT'
    if code == 'ml1m':
        return 'src.data.data.GroupDataML1M'
    if code == 'anime':
        return 'src.data.data.GroupDataANIME'
