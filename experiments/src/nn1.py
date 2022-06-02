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

train = pd.read_csv('/home/aarukavishnikov/apodtikhov/data/test.csv')

skip = 0
size_pattern = 4

current_pattern = [1] * size_pattern

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

print('Train set is ready')
print("--- %s seconds ---" % (time.time() - start_time))
# data = data[:10000]

X = dist.pairwise(data)

print('Train dists were computed')
print("--- %s seconds ---" % (time.time() - start_time))

neigh = NearestNeighbors(radius=0.01, metric='precomputed', algorithm='brute')
neigh.fit(X)

print('Nearest Neighbours is ready for use')
print("--- %s seconds ---" % (time.time() - start_time))


del X

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

print('Test set is ready')
print("--- %s seconds ---" % (time.time() - start_time))
# data1 = data1[:10000]

X = dist.pairwise(data1, data)

print('Test dists were computed')
print("--- %s seconds ---" % (time.time() - start_time))

neighborhoods = neigh.radius_neighbors(X, 0.01, return_distance=False)

print('Almost everything')
print("--- %s seconds ---" % (time.time() - start_time))


length = [len(np.asarray(neighborhoods[i])) for i in range(len(data1))]

df = pd.DataFrame({'TP': length})
df.to_csv('results2.csv', index=False)

print("--- %s seconds ---" % (time.time() - start_time))