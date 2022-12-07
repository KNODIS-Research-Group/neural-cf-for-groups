import os
import pprint
import numpy as np
import pandas as pd

from sklearn.utils import shuffle

DATA_ROOT = os.path.dirname(os.path.realpath(__file__))

"""
    This function will be used in preprocess data.
    To simplify the possibilities of loading datasets all data MUST be with secuential and 0-index ids for both users and items
    
df = pd.DataFrame({
     'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
     'col2': [2, 1, 1, 8, 1, 4],
     'col3': [0, 1, 9, 4, 2, 3],
     'col4': ['a', 'B', 'c', 'D', 'e', 'F']
})
df

  col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     1     9    c
3  NaN     8     4    D
4    D     1     2    e
5    C     4     3    F

generate_unique_ids(df, "col1")
generate_unique_ids(df, "col2")

5
4

  <_REMOVED_>
  col1  col2  col3 col4  col1  col2
0    A     2     0    a     0   1
1    A     1     1    B     0   0
2    B     1     9    c     1   0
3  NaN     8     4    D     4   3
4    D     1     2    e     3   0
5    C     4     3    F     2   2
"""
def generate_unique_ids(df, col_name):
    df_sorted=df.sort_values(by=[col_name])
    list_ids = df_sorted[col_name].unique().tolist() # [id1,id2,id3,...]
    dict_id_idx = {x: i for i, x in enumerate(list_ids)}  # dics {id1: 0, id2: 1, ...}
    tmp_col=col_name+'_tmp'
    df[tmp_col] = df[col_name]
    df[col_name] = df[tmp_col].map(dict_id_idx)
    df.drop(tmp_col, axis=1, inplace=True)
    return len(list_ids)


class ReadData(object):
    generate_unique_ids = False
    
    def pandas_wrapper(self, url, field_names):
        return pd.read_csv(
            url,
            header=0,
            names=field_names,
        )
    
    def get_data_root(self):
        return DATA_ROOT
    
    def get_data(self, url, field_names, storeto, header, x_info, y_info):
        # TODO: if url doesn't start with '/' remove de DATA_ROOT prefix. It will fetch info from Internet
        pandas_data = self.pandas_wrapper(self.get_data_root()+url, field_names)
        
        if self.generate_unique_ids != False:
            print("Generating Unique IDS")
            generate_unique_ids(pandas_data, "u")
            generate_unique_ids(pandas_data, "i")
        
        x = pandas_data[x_info].to_numpy()
        y = pandas_data[y_info].to_numpy()
        setattr(self, 'x_'+storeto, x)
        setattr(self, 'y_'+storeto, y)


class ReadTestMixin(ReadData):
    test_url     = ""
    test_header  = 0
    test_fields  = ["u", "i", "r"]
    test_x       = ["u", "i"]
    test_y       = ["r"]

    def __init__(self, *args, **kwargs):
        self.get_data(
            self.test_url,
            self.test_fields,
            'test',
            self.test_header,
            self.test_x, self.test_y,
        )
        super(ReadTestMixin, self).__init__(*args, **kwargs)


class ReadTrainMixin(ReadData):
    train_url    = ""
    train_header = 0
    train_fields = ["u", "i", "r"]
    train_x      = ["u", "i"]
    train_y      = ["r"]
    
    def __init__(self, *args, **kwargs):
        self.get_data(
            self.train_url,
            self.train_fields,
            'train',
            self.train_header,
            self.train_x,
            self.train_y,
        )
        super(ReadTrainMixin, self).__init__(*args, **kwargs)


"""
Mixin to count the number of rating per user in TRAIN
- It must be placed right after ReadTrain
"""

class ExpertInfoMixin(object):
   
    def __init__(self, *args, **kwargs):
        self.ratings_count_per_user =   pd.value_counts(
                                            self.x_train[:,0] # Access all rows, get 0 index
                                        ).to_dict()
        max_id = max(self.ratings_count_per_user.keys()) # in FT there is a user in test, not in train
        for id_u in range(max_id):
            if id_u not in self.ratings_count_per_user.keys():
                self.ratings_count_per_user[id_u] = 0
        assert(len(self.x_train) == (sum(self.ratings_count_per_user.values())))
        super(ExpertInfoMixin, self).__init__(*args, **kwargs)

    def get_rating_count(self):
        return self.ratings_count_per_user


"""
    Use after read train and before anything
"""
class UniqueIds(object):
    generate_unique_ids = True
    
    def __init__(self, *args, **kwargs):
        super(UniqueIds, self).__init__(*args, **kwargs)

"""
    Use before SplitValMixin
"""
class ShuffleMixin(object):
    def __init__(self, *args, **kwargs):
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        super(ShuffleMixin, self).__init__(*args, **kwargs)


class SplitTestMixin(object):
    train_test_percentaje = 0.1
    def __init__(self, *args, **kwargs):
        if self.train_test_percentaje != 0.0:
            train_percentage=1-self.train_test_percentaje
            total = len(self.x_train)
            train_indices = int(train_percentage * total)
            x_train, x_test, y_train, y_test = (
                self.x_train[:train_indices],
                self.x_train[train_indices:],
                self.y_train[:train_indices],
                self.y_train[train_indices:],
            )
            self.x_train, self.x_test, self.y_train, self.y_test = (x_train, x_test, y_train, y_test)
        super(SplitTestMixin, self).__init__(*args, **kwargs)


class SplitValMixin(object):
    train_val_percentaje = 0.1
    def __init__(self, *args, **kwargs):
        if self.train_val_percentaje != 0.0:
            train_percentage=1-self.train_val_percentaje
            total = len(self.x_train)
            train_indices = int(train_percentage * total)
            x_train, x_val, y_train, y_val = (
                self.x_train[:train_indices],
                self.x_train[train_indices:],
                self.y_train[:train_indices],
                self.y_train[train_indices:],
            )
            self.x_train, self.x_val, self.y_train, self.y_val = (x_train, x_val, y_train, y_val)
        super(SplitValMixin, self).__init__(*args, **kwargs)


class GetMaxMinRatingMixin(object):
    def __init__(self, *args, **kwargs):
        # Test and Validation could be []
        # Wild RAW datasets...
        self.rating_max = float(max(self.y_train))
        self.rating_min = float(min(self.y_train))
        if hasattr(self, 'y_test') and len(self.y_test) > 0:
            self.rating_max = float(max(self.rating_max, max(self.y_test)))
            self.rating_min = float(min(self.rating_min, min(self.y_test)))
        if hasattr(self, 'y_val') and len(self.y_val) > 0:
            self.rating_max = float(max(self.rating_max, max(self.y_val)))
            self.rating_min = float(min(self.rating_min, min(self.y_val)))
        super(GetMaxMinRatingMixin, self).__init__(*args, **kwargs)


class GetNumsByMaxMixin(object):
    def __init__(self, *args, **kwargs):
        train_num_users = self.x_train[:,0].max() + 1 #Ids start in 0
        train_num_items = self.x_train[:,1].max() + 1 #Ids start in 0
        if hasattr(self, 'x_test') and len(self.x_test) > 0:
            test_num_users = self.x_test[:,0].max() + 1 #Ids start in 0
            test_num_items = self.x_test[:,1].max() + 1 #Ids start in 0
        else:
            test_num_users = 0
            test_num_items = 0
        if hasattr(self, 'x_val') and len(self.x_val) > 0:
            val_num_users = self.x_val[:,0].max() + 1 #Ids start in 0
            val_num_items = self.x_val[:,1].max() + 1 #Ids start in 0
        else:
            val_num_users = 0
            val_num_items = 0
        
        self.num_users = int(max(train_num_users, test_num_users, val_num_users))
        self.num_items = int(max(train_num_items, test_num_items, val_num_items))
        super(GetNumsByMaxMixin, self).__init__(*args, **kwargs)

"""
    Save standar dataset splited
"""
class SaveDataMixin(object):
    train_name='training-ratings.csv'
    test_name='test-ratings.csv'
    
    def __init__(self, *args, **kwargs):
        super(SaveDataMixin, self).__init__(*args, **kwargs)
    
    def save_train_test(self, path):
        pd_train = pd.DataFrame(np.concatenate([self.x_train,self.y_train], axis=1), columns=['u','i','r'])
        pd_train.to_csv(path+self.train_name,index=False)
        if hasattr(self, 'x_test') and len(self.x_test)>0:
            pd_test = pd.DataFrame(np.concatenate([self.x_test,self.y_test], axis=1), columns=['u','i','r'])
            pd_test.to_csv(path+self.test_name,index=False)
        


if __name__ == '__main__':

    from data_general import GeneralRS
    
    class GroupDataGeneralRS(ReadTestMixin, ReadTrainMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):
        test_url = "/grupos/ml/test-ratings.csv"
        train_url = "/grupos/ml/training-ratings.csv"

        def __init__(self, *args, **kwargs):
            super(GroupDataGeneralRS, self).__init__(*args, **kwargs)
    
    test = GroupDataGeneralRS()
    
    print(test.info())
    