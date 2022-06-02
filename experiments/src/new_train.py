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

results = pd.read_csv('results.csv')
'''
train = pd.read_csv('/home/aarukavishnikov/apodtikhov/data/train.csv')

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

idxs = np.unique(results['index'])

train_idxs_dict = dict(zip(range(len(idxs)), idxs))
data = data[idxs]

print('Train set is ready')
print("--- %s seconds ---" % (time.time() - start_time))

X = dist.pairwise(data)

print('Train dists were computed')
print("--- %s seconds ---" % (time.time() - start_time))

neigh = NearestNeighbors(radius=0.01, metric='precomputed', algorithm='brute')
neigh.fit(X)

print('Nearest Neighbours is ready for use')
print("--- %s seconds ---" % (time.time() - start_time))

del X
'''
test = pd.read_csv('/home/aarukavishnikov/apodtikhov/data/test.csv')

data1 = test[np.isnan(test.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values).sum(axis=1) == 0]
q = data1.iloc[:, 4 + skip : 4 + skip + np.sum(current_pattern) + 1].values
idxs = [0]
for p in current_pattern:
    idxs.append(idxs[-1] + p)
q = q[:, np.array(idxs)]
# data1 = data1[:, ::-1]
idxs = np.where(q.max(axis=1) != q.min(axis=1))[0]
data1 = data1.iloc[idxs]
data1 = data1.reset_index(drop=True)
results = results.reset_index(drop=True)
data1['nn_cluster_index_0'] = results['index']
data1['nn_cluster_dist_0'] = results['distance']
data1 = data1[list(data1.columns[-2:]) + list(data1.columns[:-2])]

print(data1.head())
print(data1.shape)

data1.to_csv('results1.csv', index=False)

raise 1
print('Test set is ready')
print("--- %s seconds ---" % (time.time() - start_time))

X = dist.pairwise(data1, data)

print('Test dists were computed')
print("--- %s seconds ---" % (time.time() - start_time))

neighborhoods = neigh.kneighbors(X, 1, return_distance=True)

print('Almost everything')
print("--- %s seconds ---" % (time.time() - start_time))

distances = np.asarray(neighborhoods[0]).flatten()
indexes = np.asarray(neighborhoods[1]).flatten()

df = pd.DataFrame({'distance': distances, 'index': indexes})
# df.to_csv('results.csv', index=False)

print("--- %s seconds ---" % (time.time() - start_time))