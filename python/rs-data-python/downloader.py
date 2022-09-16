import os
import subprocess
import pandas as pd
import numpy as np
from data_utils import init_random

from zipfile import ZipFile
import urllib.request
DATADIR='datasets'

"""
datasets\
    \ - code
        \ - raw\
            \ - blabla...
        \- test-ratings.csv
        \- train-ratings.csv
"""

def get_data(code, url):
    data_dir=DATADIR+'/'+code
    data_dir_raw=DATADIR+'/'+code+'/raw'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir_raw)
        #urllib.request.urlretrieve(url,f"{data_dir_raw}/{code}.zip")
        #with ZipFile(f"{data_dir_raw}/{code}.zip", 'r') as zip:
        #    zip.extractall(data_dir_raw)
        subprocess.call(["wget", url, "-O", f"{data_dir_raw}/{code}.zip"])
        subprocess.call(["unzip", "-a", "-j", f"{data_dir_raw}/{code}.zip", "-d", data_dir_raw])

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, UniqueIds, ReadTrainMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitTestMixin, SplitValMixin, GetNumsByMaxMixin, SaveDataMixin


# Book-Crossing Dataset
# url: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
class RawBX(UniqueIds, ReadTrainMixin, ShuffleMixin, SplitTestMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, SaveDataMixin, GeneralRS, object):
    train_url = '/'+DATADIR+"/bx/raw/BX-Book-Ratings.csv"
    code = "bx"

    def pandas_wrapper(self, url, field_names):
        return pd.read_csv(
            url,
            header=0,
            names=field_names,
            sep=';',
            warn_bad_lines=True,
            error_bad_lines=False,
            encoding = 'ISO-8859-1'
        )


def process_bx():
    get_data("bx", "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
    raw = RawBX()
    print(raw.info())
    raw.save_train_test(DATADIR+"/bx/")


# Movielens
# https://grouplens.org/datasets/movielens/
class RawMl(UniqueIds, ReadTrainMixin, ShuffleMixin, SplitTestMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, SaveDataMixin, GeneralRS, object):
    def pandas_wrapper(self, url, field_names):
        return pd.read_csv(url, sep="::", engine='python', names=['u','i','r','ts']).drop(['ts'],axis=1)

class RawMl1m(RawMl):
    train_url = '/'+DATADIR+"/ml1m/raw/ratings.dat"
    code = "ml1m"

def process_ml1m():
    get_data("ml1m", "https://files.grouplens.org/datasets/movielens/ml-1m.zip")
    #Seed=1234
    init_random(val=1234)
    raw = RawMl1m()
    print(raw.info())
    raw.save_train_test(DATADIR+"/ml1m/")


if __name__ == '__main__':
    #get_data("bx", "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
    process_ml1m()
    #get_data("ml-1m", "https://files.grouplens.org/datasets/movielens/ml-1m.zip")
