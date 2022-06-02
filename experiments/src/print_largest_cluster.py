import pandas as pd
import os
import re
import numpy as np
import pickle
import joblib
import time

start_time = time.time()

train = pd.read_csv('/home/aarukavishnikov/apodtikhov/data/test.csv')

skip = 0

for size_pattern in range(4, 5):
    current_pattern = [1] * size_pattern

    d = {}

    data = train[np.isnan(train.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values).sum(axis=1) == 0]
    data = data.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values
    idxs = [0]
    for p in current_pattern:
        idxs.append(idxs[-1] + p)
    data = data[:, np.array(idxs)]
    data = data[:, ::-1]
    data = data[data.max(axis=1) != data.min(axis=1)]
    data = (data - data.min(axis=1).reshape(-1, 1)) / (data.max(axis=1).reshape(-1, 1) - data.min(axis=1).reshape(-1, 1))
    data = pd.DataFrame(data).drop_duplicates().values

    # data = data[1 :11]


    from sklearn.neighbors import NearestNeighbors
    from sklearn.neighbors import DistanceMetric
    dist = DistanceMetric.get_metric('dtw')

    X = dist.pairwise(data)
    print(X.shape)
    print(X)
    print("--- %s seconds ---" % (time.time() - start_time))
    raise 1