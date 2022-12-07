import os
import math

import numpy as np
import pandas as pd

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, UniqueIds, ReadTrainMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitTestMixin, SplitValMixin, GetNumsByMaxMixin


DATA_ROOT = os.path.dirname(os.path.realpath(__file__))


class DatasetBX(ReadTrainMixin, ReadTestMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):
    test_url = "/datasets/bx/test-ratings.csv"
    train_url = "/datasets/bx/training-ratings.csv"
    code = "bx"


class DatasetML1M(ReadTrainMixin, ReadTestMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):
    test_url = "/datasets/ml1m/test-ratings.csv"
    train_url = "/datasets/ml1m/training-ratings.csv"
    code = "ml1m"



if __name__ == '__main__':
    test = DatasetML1M()
    print(test.info())