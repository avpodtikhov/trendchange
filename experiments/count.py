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
DELTAS = [1]
SIZE_PATTERNS = [2]
# EPS_RANGE = [0.01]
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
patterns = pd.read_csv('/home/avpodtikhov/trendchange/results/hun/clean2/2/1.csv').iloc[:, :-2].values

data_new = {'N_mistakes': [], 'Total': [], 'TP': [], 'FN': []}

for n_mistakes in tqdm([0]):
    work_all = []
    count_mistakes = 0
    for DELTA in tqdm(DELTAS, leave=False):
        for SIZE_PATTERN in tqdm(SIZE_PATTERNS, leave=False):
            current_pattern = [1] * SIZE_PATTERN
            # Обрабатываем тестовый датасет
            data, idxs1, idxs2  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=0, return_idxs=True)
            data = scale(data=data, scaler=SCALER)

            D = dist.pairwise(patterns, data, N_JOBS)

            work = list(map(lambda x: list(map(lambda y: idxs1[idxs2[y]], list(np.where(x < SIZE_TO_EPS[SIZE_PATTERN])[0]))), D))
            work = [item for sublist in work for item in sublist]
            print(len(set(work)))
            work_all = work_all + work
            eps = SIZE_TO_EPS[SIZE_PATTERN]
            
            for skip in tqdm([2], leave=False):
                # Обработка тестового датасета со сдвигом на skip шагов
                if 3 + skip + DELTA + np.sum(current_pattern) >= test.shape[1]:
                    break
                data, idxs1, idxs2  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=skip, return_idxs=True)
                if data.shape[0] == 0:
                    break
                data = scale(data=data, scaler=SCALER)
                D1 = dist.pairwise(patterns, data, N_JOBS)
                work = list(map(lambda x: list(map(lambda y: idxs1[idxs2[y]], list(np.where(x < SIZE_TO_EPS[SIZE_PATTERN])[0]))), D1))
                work = [item for sublist in work for item in sublist]
                print(len(set(work)))
                count_mistakes += np.sum(D1)
    data_new['N_mistakes'].append(n_mistakes)
    data_new['Total'].append(len(work_all))
    data_new['TP'].append(len(set(work_all)))
    data_new['FN'].append(len(set(work)))
    print(data_new)
print('----')
print(data_new)