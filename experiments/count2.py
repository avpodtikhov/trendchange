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
# Гиперпараметры модели
DELTAS = range(1, 11)
SIZE_PATTERNS = range(2, 7)
SCALER = 'max'
N_JOBS = 96
SIZE_TO_EPS = {2: 0.002,
               3: 0.01,
               4: 0.028,
               5: 0.04,
               6: 0.106}

# Создаем инстанс класса расчета расстояний
dist = DTWDistance()

# Подгужаем данные
test = pd.read_csv('source/test_stocks.csv')
patterns_all = pd.read_csv('/home/avpodtikhov/trendchange/results/hun/clean/results2.csv')
# patterns_all = pd.read_csv('/home/avpodtikhov/trendchange/results/hun/results_count_two.csv', index_col=0).iloc[:, :-1]
data_new = {'N_mistakes': [], 'DELTA': [], 'SIZE_PATTERN': [], 'Total': [], 'TP': [], 'FN': [], 'Total1': []}

for n_mistakes in tqdm([10]):
    work_all = []
    count_mistakes = 0
    for DELTA in tqdm(DELTAS, leave=False):
        for SIZE_PATTERN in tqdm(SIZE_PATTERNS, leave=False):
            current_pattern = [1] * SIZE_PATTERN
            # Обрабатываем тестовый датасет
            data, idxs1, idxs2  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=0, return_idxs=True)
            data = scale(data=data, scaler=SCALER)

            patterns = patterns_all[(patterns_all['delta'] == DELTA) & (patterns_all['size'] == SIZE_PATTERN)]
            patterns = patterns[patterns['FN'] <= n_mistakes]
            patterns = patterns.iloc[:, 3 : 3 + SIZE_PATTERN + 1].values

            D = dist.pairwise(patterns, data, N_JOBS)

            work = list(map(lambda x: list(map(lambda y: idxs1[idxs2[y]], list(np.where(x < SIZE_TO_EPS[SIZE_PATTERN])[0]))), D))
            work = [item for sublist in work for item in sublist]
            work_all = work_all + work

    print(data_new)

    data_new['N_mistakes'].append(n_mistakes)
    data_new['Total'].append(len(work_all))
    data_new['TP'].append(len(set(work_all)))
    data_new['FN'].append(0)
    data_new['Total1'].append(0)
    print(data_new)
    print(test.shape[0])
    dss = []
    for skip in tqdm(range(10, 30), leave=False):
        ds = [0] * test.shape[0]
        for DELTA in tqdm(DELTAS, leave=False):
            for SIZE_PATTERN in tqdm(SIZE_PATTERNS, leave=False):
                current_pattern = [1] * SIZE_PATTERN
                # Обрабатываем тестовый датасет
                idxs0 = np.where(np.array(ds) == 0)[0]
                data = test.iloc[idxs0]
                data, idxs1, idxs2  = cut_ts(data=data, pattern=current_pattern, delta=DELTA, skip=skip, return_idxs=True)
                data = scale(data=data, scaler=SCALER)

                patterns = patterns_all[(patterns_all['delta'] == DELTA) & (patterns_all['size'] == SIZE_PATTERN)]
                patterns = patterns[patterns['FN'] <= n_mistakes]
                patterns = patterns.iloc[:, 3 : 3 + SIZE_PATTERN + 1].values

                D = dist.pairwise(patterns, data, N_JOBS)

                work = list(map(lambda x: list(map(lambda y: y, list(np.where(x < SIZE_TO_EPS[SIZE_PATTERN])[0]))), D))
                work = [item for sublist in work for item in sublist]
                for j in work:
                    ds[idxs0[idxs1[idxs2[j]]]] = 1
        dss.append(sum(ds))
        data_new['FN'][-1] += sum(ds)
        data_new['Total1'][-1] += len(ds)
    print(dss)
    print(data_new)

print('----')
print(data_new)