import pandas as pd
import os
import re
import numpy as np
import pickle
import joblib
import time
from dtw_cython.dtw import DTWDistance
from tqdm.auto import tqdm
from utils import *
import matplotlib.pyplot as plt
# Гиперпараметры модели
DELTAS = [1]
SIZE_PATTERNS = range(3, 7)
N_JOBS = 96

dist = DTWDistance()
p = []
for i in range(2, 7):
    patterns = pd.read_csv(f'/home/avpodtikhov/trendchange/results/hun/{i}/1.csv')
    patterns = patterns[patterns['FN'] == 0]
    patterns = patterns.iloc[:, :-2].values
    print(patterns.shape)
    p.append(patterns)

matrixes = []
for el1 in tqdm(p):
    h = None
    for el2 in tqdm(p):
        D = dist.pairwise(el1, el2, N_JOBS)
        if h is None:
            h = D
        else:
            h = np.hstack([h, D])
    matrixes.append(h)
for i in matrixes:
    print(i.shape)

simm = np.vstack([matrixes[0], matrixes[1], matrixes[2], matrixes[3], matrixes[4]])

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.01, min_samples=3, metric='precomputed').fit(simm)
print(np.unique(clustering.labels_, return_counts=True))
patterns_list = []
for i in np.unique(clustering.labels_):
    if i == -1:
        continue
    plt.figure(figsize=(16, 8))
    idxs = np.where(clustering.labels_ == i)[0]
    # print(idxs)
    # raise 1
    for idx in idxs:
        s = 0
        for skip in range(len(p)):
            if idx < s + p[skip].shape[0]:
                patterns_list.append(list(p[skip][idx - s]))
                plt.plot(p[skip][idx - s])
                break
            s = s + p[skip].shape[0]
    plt.savefig(f'imgs/cl_{i}.png')
    plt.close()
pd.DataFrame(patterns_list).to_csv('cluster.csv', index=False)
'''
d = []
for i in tqdm(range(simm.shape[0])):
    md = 100
    for j in tqdm(range(simm.shape[0]), leave=False):
        if i != j:
            md = min(md, simm[i, j])
    d.append(md)
distances = np.sort(d)
plt.figure(figsize=(16, 8))
plt.plot(distances)
plt.savefig('elbow.png')
'''