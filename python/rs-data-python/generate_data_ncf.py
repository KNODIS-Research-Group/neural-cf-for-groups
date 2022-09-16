import os

dirName = 'ncf-data'
if not os.path.exists(dirName):
    os.makedirs(dirName)

from rs_data import DatasetNCF
dataNCFOriginal = DatasetNCF("../neural_collaborative_filtering/Data/ml-1m")


import pickle

for i in range(3000):
    train_val = dataNCFOriginal.get_train_val()
    pickle.dump(train_val, open(dirName+'/'+str(i)+'.pkl','wb'))