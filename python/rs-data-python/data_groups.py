import os
import math

import numpy as np
import pandas as pd
from keras.utils import Sequence

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, ReadTrainMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitValMixin, GetNumsByMaxMixin
from data_onehot import dense_embedding, dense_embedding_for_group, dense_embedding_for_group_with_expert_info


DATA_ROOT = os.path.dirname(os.path.realpath(__file__))

"""
Mixin to count the number of rating per user in train
- It must be placed right after ReadTrain
"""
class ExpertInfo(object):
   
    def __init__(self, *args, **kwargs):
        self.ratings_count_per_user =   pd.value_counts(
                                            self.x_train[:,0] # Access all rows, get 0 index
                                        ).to_dict()
        assert(len(self.x_train) == (sum(self.ratings_count_per_user.values())))
        super(ExpertInfo, self).__init__(*args, **kwargs)


# TODO: reorganized in Mixin code
class GroupData(ReadTestMixin, ReadTrainMixin, ExpertInfo, ShuffleMixin, SplitValMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):

    def __init__(self, *args, **kwargs):
        super(GroupData, self).__init__(*args, **kwargs)
    
    def get_shape(self):
        return (self.get_num_users() + self.get_num_items(),) # OneHot Embedding
    
    def get_test_one_hot(self):
        return (
            dense_embedding(self.x_test, self.get_num_users(), self.get_num_items()),
            self.y_test
        )
    
    def __get_dataset_for_group_size(self, group_size):
        names = ["i"]
        for u in range(1, group_size+1):
            names.append("u"+str(u))
            names.append("r"+str(u))
        
        group_data = pd.read_csv(
            DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+".csv",
            header=0,
            names=names
        )
        group_data = group_data.fillna(0).to_numpy(dtype=np.int32)
        return group_data

    # Get rating count for group info
    def get_rating_count_table_for_group(self, group_size):
        group_data = self.__get_dataset_for_group_size(group_size)
        rating_count_table = []
        
        for idx, row in enumerate(group_data): # i, u1, r1, u2, r2, etc...
            item = row[0] # i
            rating_count_row = []
            for u in range(0, group_size):
                user = row[1+(u*2)]
                rating_count_row.append(self.ratings_count_per_user[user])
            rating_count_table.append(rating_count_row)
        
        rating_count_table = np.asarray(rating_count_table)
        return rating_count_table

    def get_test_for_group_size(self, group_size, one_hot_activation=None):
        if one_hot_activation == None:
            one_hot_activation = float(1)/group_size
        
        group_data = self.__get_dataset_for_group_size(group_size)
        group_data = dense_embedding_for_group(group_data, self.get_num_users(), self.get_num_items(), group_size, one_hot_activation)
        return group_data

    def get_test_for_group_size_with_expert_info(self, group_size):
        group_data = self.__get_dataset_for_group_size(group_size)
        group_data = dense_embedding_for_group_with_expert_info(group_data, self.get_num_users(), self.get_num_items(), group_size, self.ratings_count_per_user) # TODO: Reorganize this code with the Mixin: ExpertInfo
        return group_data

    def get_test_for_group_size_as_individual(self, group_size):
        group_data = self.__get_dataset_for_group_size(group_size)
        
        as_individual = []

        for idx, row in enumerate(group_data): # i, u1, r1, u2, r2, etc...
            item = row[0] # i
            for u in range(0, group_size):
                user = row[1+(u*2)]
                as_individual.append([user, item])

        assert(len(as_individual) == (len(group_data) * group_size))

        as_individual = np.asarray(as_individual)
        as_individual = dense_embedding(as_individual, self.get_num_users(), self.get_num_items())
        return as_individual
    



class GroupDataML(GroupData):
    test_url = "/grupos/ml/test-ratings.csv"
    train_url = "/grupos/ml/training-ratings.csv"
    code = "ml"


class GroupDataML1M(GroupData):
    test_url = "/grupos/ml1m/test-ratings.csv"
    train_url = "/grupos/ml1m/training-ratings.csv"
    code = "ml1m"


class GroupDataML1MCompleteInfo(GroupData):
    test_url = "/grupos/ml1m-completeinfo/test-ratings.csv"
    train_url = "/grupos/ml1m-completeinfo/training-ratings.csv"
    code = "ml1m-completeinfo"


class GroupDataFT(GroupData):
    test_url = "/grupos/ft/test-ratings.csv"
    train_url = "/grupos/ft/training-ratings.csv"
    code = "ft"


class GroupDataANIME(GroupData):
    test_url = "/grupos/anime/test-ratings.csv"
    train_url = "/grupos/anime/training-ratings.csv"
    code = "anime"


if __name__ == '__main__':
    test = GroupDataML1M()
    print(test.info())
    

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



class OneHotGeneratorGroupSampling(OneHotGenerator):

    def __init__(self, x, y, num_users, num_items, batch_size, nsampling, grp_size):
        super(OneHotGeneratorGroupSampling, self).__init__(x, y, num_users, num_items, batch_size)
        self.nsampling = nsampling

        if batch_size % nsampling != 0:
            raise ValueError('batch_size % nsampling must be 0')

        self.pddata = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
        self.pddata.columns = ['u','i','r']
        self.batch_size = self.batch_size // nsampling
        self.grp_size = grp_size
    
    def __len__(self):
        return self.nsampling * super(OneHotGeneratorGroupSampling).__len__()
    
    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size

        ori_batch = self.pddata[idx_start:idx_end]

        # Sácalo a una función testeable en data one_hot
        #https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        #https://stackoverflow.com/questions/48828484/pandas-reshape-dataframe-every-n-rows-to-columns
        #pddata[pddata['i'] == 0] selcciona items
        """
import pandas as pd
import numpy as np
ratings_ori = pd.DataFrame(np.arange(24).reshape(12,-1),columns=['u','r'])
i = pd.DataFrame(np.zeros(12, dtype=int),columns=['i'])
i[7:12]=np.full((5,1),1)
d = pd.concat([i,ratings_ori], axis=1)
dl = d.groupby('i', as_index=False).agg({'u': list, 'r': list})


#max_ratings = dl['u'].str.len().max() # longitud mayor de la lista


users = pd.DataFrame(dl['u'].tolist(), dtype="Int64")
users.columns = ['user-' + str(x+1) for x in users.columns]
ratings = pd.DataFrame(dl['r'].tolist(), dtype=float)
ratings.columns = ['rating-' + str(x+1) for x in ratings.columns]
dl['item']=dl['i']
data = pd.concat([dl['item'],users,ratings], axis=1)
        """

        return np.array(dense_embedding(batch_x, self.num_users, self.num_items)), np.array(batch_y)