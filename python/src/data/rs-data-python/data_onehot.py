import numpy as np
import pandas as pd

"""

Function related to generate onehot

"""

"""
  original: [id_user, id_item]
  
  NOTE 0-index
"""
def dense_embedding(original, num_user, num_items):
    total_features = num_user + num_items
    n_data = len(original)
    
    dense_input_as_embedding = np.zeros((n_data, total_features), dtype=np.int32)
    
    for idx, val in enumerate(original):
        u = val[0]
        i = val[1]
        dense_input_as_embedding[idx][u] = 1
        #print(val)
        #print(idx, u, i, num_user)
        dense_input_as_embedding[idx][num_user + i] = 1

    return dense_input_as_embedding

"""
## Example

nnu = 3
nni = 6
nnd = np.array([[2, 5], [2, 3], [1, 4], [1, 2], [1, 0], [0, 0], [0, 3]])

nned = dense_embedding(nnd, nnu, nni)
print(nned[:,:3])
print(nned[:,3:])
print(nned)


[[0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [1. 0. 0.]]
[[0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]]
[[0. 0. 1. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 1. 0. 0.]]
"""

"""
  original: [id_item, id_user_1, rating_1, id_user_2, rating_2, ..., id_user_n, rating_n]
  
  NOTE 0-index
"""
def dense_embedding_for_group(original, num_user, num_items, group_size, activation):
    total_features = num_user + num_items
    one_hot_activation = activation

    n_data = len(original)
    
    dense_input_as_embedding = np.zeros((n_data, total_features))
    for idx, val in enumerate(original):
        i = int(val[0])
        dense_input_as_embedding[idx][num_user + i] = 1
        for u in range(group_size):
            u = int(val[1+(u*2)])
            dense_input_as_embedding[idx][u] = one_hot_activation
    
    return dense_input_as_embedding

"""
## Example

nnu = 3
nni = 6
size_grp = 2
nnd = np.array([[5, 1, 1.0, 2, 5.0], [1, 2, 3.0, 1, ], [4, 1, 3.0, 2, 2.0], [2, 1, 5.0, 0, 5.0], [0, 2, 3.0, 1, ]])

nned = dense_embedding_for_group(nnd, nnu, nni, size_grp)
print(nned[:,:nnu])
print(nned[:,nnu:])
print(nned)


[[0.  0.5 0.5]
 [0.  0.5 0.5]
 [0.  0.5 0.5]
 [0.5 0.5 0. ]
 [0.  0.5 0.5]]
[[0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]]
[[0.  0.5 0.5 0.  0.  0.  0.  0.  1. ]
 [0.  0.5 0.5 0.  1.  0.  0.  0.  0. ]
 [0.  0.5 0.5 0.  0.  0.  0.  1.  0. ]
 [0.5 0.5 0.  0.  0.  1.  0.  0.  0. ]
 [0.  0.5 0.5 1.  0.  0.  0.  0.  0. ]]
"""


"""
    Use in expert and softmax
    https://www.programiz.com/python-programming/closure
"""
def dense_embedding_for_group_with_closure(original, num_user, num_items, group_size, closure):
    return closure(original)


"""
  original: [rating]
  
  NOTE 0-index
"""
def dense_embedding_rating(original, num_val=5):
    total_features = num_val
    n_data = len(original)
    dense_input_as_embedding = np.zeros((n_data, total_features))
    for idx, val in enumerate(original):
        r = int(val - 1)
        if r >= 0:  # -1 no rating
            dense_input_as_embedding[idx][r] = 1

    return dense_input_as_embedding

"""
## Example

nnr = np.array(
    [
        3,
        4,
        3,
        4,
        0,
        0,
        3,
        1,
        5,
    ]
)

nner = dense_embedding_rating(nnr, 5)
nner

array([[0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.]])
"""

#Convertir en porcentajes los votos de entrada

VOTE=0.6
SEMIVOTE=0.2

# NO ONE HOT! previous version too slow
def dense_embedding_rating_as_probability(original, num_val=5):
    total_features = num_val
    n_data = len(original)
    dense_input_as_embedding = np.zeros((n_data, total_features))
    
    for idx, val in enumerate(original):
        i = int(val - 1)
        if i >= 0:  # -1 no rating
            dense_input_as_embedding[idx][i] = VOTE
            if i-1<0:
                dense_input_as_embedding[idx][i] += SEMIVOTE
            else:
                dense_input_as_embedding[idx][i-1] += SEMIVOTE
            if i+1>4:
                dense_input_as_embedding[idx][i] += SEMIVOTE
            else:
                dense_input_as_embedding[idx][i+1] += SEMIVOTE

    return dense_input_as_embedding

""" Example
nnr = array([3, 4, 3, 4, 0, 0, 3, 1, 5])
dense_embedding_rating_as_probability(nnr)

array([[0. , 0.2, 0.6, 0.2, 0. ],
       [0. , 0. , 0.2, 0.6, 0.2],
       [0. , 0.2, 0.6, 0.2, 0. ],
       [0. , 0. , 0.2, 0.6, 0.2],
       [0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. ],
       [0. , 0.2, 0.6, 0.2, 0. ],
       [0.8, 0.2, 0. , 0. , 0. ],
       [0. , 0. , 0. , 0.2, 0.8]])
"""
