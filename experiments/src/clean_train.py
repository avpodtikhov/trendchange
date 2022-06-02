import pandas as pd
import os
import re
import numpy as np
import pickle
import joblib
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise_distances
dist = DistanceMetric.get_metric('dtw')

start_time = time.time()

skip = 0
size_pattern = 4

current_pattern = [1] * size_pattern

patterns = pd.read_csv('/home/aarukavishnikov/apodtikhov/src/results2.csv')

test = pd.read_csv('/home/aarukavishnikov/apodtikhov/data/train.csv')

data1 = test[np.isnan(test.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values).sum(axis=1) == 0]
data1 = data1.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values
idxs = [0]
for p in current_pattern:
    idxs.append(idxs[-1] + p)
data1 = data1[:, np.array(idxs)]
data1 = data1[:, ::-1]
data1 = data1[data1.max(axis=1) != data1.min(axis=1)]
data1 = (data1 - data1.min(axis=1).reshape(-1, 1)) / (data1.max(axis=1).reshape(-1, 1) - data1.min(axis=1).reshape(-1, 1))

res = pd.concat([patterns, pd.DataFrame(data1)], axis=1)
res = res[res['TP'] != 0]
res['FN'] = 0
res['round'] = 0
data1 = res.iloc[:, 1:-2].values
print(res.head())

print('Test set is ready')
print("--- %s seconds ---" % (time.time() - start_time))


train = pd.read_csv('/home/aarukavishnikov/apodtikhov/data/test.csv')
for skip in range(1, train.shape[1]):
    if 4 + skip + np.sum(current_pattern) > train.shape[1]:
        break
    data = train[np.isnan(train.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values).sum(axis=1) == 0]
    if data.shape[0] == 0:
        break
    data = data.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values
    idxs = [0]
    for p in current_pattern:
        idxs.append(idxs[-1] + p)
    data = data[:, np.array(idxs)]
    data = data[:, ::-1]
    data = data[data.max(axis=1) != data.min(axis=1)]
    if data.shape[0] == 0:
        break
    data = (data - data.min(axis=1).reshape(-1, 1)) / (data.max(axis=1).reshape(-1, 1) - data.min(axis=1).reshape(-1, 1))
    data = pd.DataFrame(data).drop_duplicates().values

    print('Train set is ready. Round {}'.format(skip))
    print("--- %s seconds ---" % (time.time() - start_time))

    X = dist.pairwise(data)

    print('Train dists were computed')
    print("--- %s seconds ---" % (time.time() - start_time))

    neigh = NearestNeighbors(radius=0.01, metric='precomputed', algorithm='brute')
    neigh.fit(X)

    print('Nearest Neighbours is ready for use')
    print("--- %s seconds ---" % (time.time() - start_time))

    del X

    X = dist.pairwise(data1, data)

    print('Test dists were computed')
    print("--- %s seconds ---" % (time.time() - start_time))

    neighborhoods = neigh.radius_neighbors(X, 0.01, return_distance=False)

    print('Almost everything')
    print("--- %s seconds ---" % (time.time() - start_time))


    length = [len(np.asarray(neighborhoods[i])) for i in range(len(data1))]

    res['FN'] = res['FN'] + np.array(length)
    res['round'] = skip
    res.to_csv('res.csv', index=False)

res.to_csv('res.csv', index=False)