import math
from scipy.special import softmax
"""
MOVED To rs-data-python
as closure
"""

"""
  original: [id_item, id_user_1, rating_1, id_user_2, rating_2, ..., id_user_n, rating_n]
  
  NOTE 0-index
  
  rating_count_per_user:
  {
      1: 34,
      2: 17,
      ...
      6000: 54
  }

def dense_embedding_for_group_with_expert_info(original, num_user, num_items, group_size, rating_count_per_user):
    total_features = num_user + num_items    
    n_data = len(original)
    dense_input_as_embedding = np.zeros((n_data, total_features))
    for idx, val in enumerate(original):
        i = int(val[0])
        # item
        dense_input_as_embedding[idx][num_user + i] = 1
        total_rating_in_group = 0
        for u in range(group_size):
            u = int(val[1+(u*2)]) #item offset (+1) , even colums...
            total_rating_in_group = total_rating_in_group + rating_count_per_user[u]
        
        for u in range(group_size):
            u = int(val[1+(u*2)]) #item offset (+1) , even colums...    
            if total_rating_in_group == 0:
                dense_input_as_embedding[idx][u] = 1 / group_size
            else:
                dense_input_as_embedding[idx][u] = rating_count_per_user[u] / total_rating_in_group
    
    return dense_input_as_embedding

## Example

nnu = 3
nni = 6
size_grp = 2
rating_count = {0:1,1:5,2:3}
nnd = np.array([[5, 1, 1.0, 2, 5.0], [1, 2, 3.0, 1, ], [4, 1, 3.0, 2, 2.0], [2, 1, 5.0, 0, 5.0], [0, 2, 3.0, 1, ]])

nned = dense_embedding_for_group_with_expert_info(nnd, nnu, nni, size_grp, rating_count)
print(nned[:,:nnu])
print(nned[:,nnu:])
print(nned)

[[0.         0.625      0.375     ]
 [0.         0.625      0.375     ]
 [0.         0.625      0.375     ]
 [0.16666667 0.83333333 0.        ]
 [0.         0.625      0.375     ]]
[[0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]]
 [[0.         0.625      0.375      0.         0.         0.    0.         0.         1.        ]
  [0.         0.625      0.375      0.         1.         0.    0.         0.         0.        ]
  [0.         0.625      0.375      0.         0.         0.    0.         1.         0.        ]
  [0.16666667 0.83333333 0.         0.         0.         1.    0.         0.         0.        ]
  [0.         0.625      0.375      1.         0.         0.    0.         0.         0.        ]]


def dense_embedding_for_group_with_softmax(original, num_user, num_items, group_size, rating_count_per_user):
    total_features = num_user + num_items    
    n_data = len(original)
    dense_input_as_embedding = np.zeros((n_data, total_features))
    for idx, val in enumerate(original):
        i = int(val[0])
        # item
        dense_input_as_embedding[idx][num_user + i] = 1
        total_rating_in_group = 0
        for u in range(group_size):
            u = int(val[1+(u*2)]) #item offset (+1) , even colums...
            total_rating_in_group = total_rating_in_group + rating_count_per_user[u]
        
        for u in range(group_size):
            u = int(val[1+(u*2)]) #item offset (+1) , even colums...
            if total_rating_in_group == 0:
                dense_input_as_embedding[idx][u] = 1 / group_size
            else:
                dense_input_as_embedding[idx][u] = rating_count_per_user[u] / total_rating_in_group
        
        exp_sum = 0
        for u in range(group_size):
            u = int(val[1+(u*2)]) #item offset (+1) , even colums...
            exp_sum += math.exp(dense_input_as_embedding[idx][u])

        for u in range(group_size):
            u = int(val[1+(u*2)]) #item offset (+1) , even colums...
            dense_input_as_embedding[idx][u] = math.exp(dense_input_as_embedding[idx][u]) / exp_sum

    return dense_input_as_embedding
"""