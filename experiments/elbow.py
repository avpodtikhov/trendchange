'''
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
DELTAS = [2]
SIZE_PATTERNS = [4]
SCALER = 'minmax'
N_JOBS = 96

# Создаем инстанс класса расчета расстояний
dist = DTWDistance()

# Подгужаем данные
test = pd.read_csv('source/test_stocks.csv')
patterns = pd.read_csv('/home/avpodtikhov/trendchange/new/2_4_minmax.csv')

DELTA = DELTAS[0]
SIZE_PATTERN = SIZE_PATTERNS[0]

current_pattern = [1] * SIZE_PATTERN
# Обрабатываем тестовый датасет

data, idxs1, idxs2  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=0, return_idxs=True)
data = scale(data=data, scaler=SCALER)

D = dist.pairwise(patterns.iloc[:, :-2].values, data, N_JOBS)
work = set()
err = []
for pattern_idx in range(patterns.shape[0]):
    idx_true = np.where(D[pattern_idx] < list(patterns['EPS'])[pattern_idx])[0]
    work = work | set(list(idx_true))
    if len(idx_true) != 0:
        err.append(pattern_idx)
print('TRUE:', len(work))
print(len(work))
print(err)
print(np.array(patterns['EPS'])[err])

false = 0
for skip in range(1, 5):
    data, idxs1, idxs2  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=skip, return_idxs=True)
    data = scale(data=data, scaler=SCALER)

    D = dist.pairwise(patterns.iloc[:, :-2].values, data, N_JOBS)
    work = set()
    err = []
    for pattern_idx in tqdm(range(patterns.shape[0])):
        idx_true = np.where(D[pattern_idx] < list(patterns['EPS'])[pattern_idx])[0]
        work = work | set(list(idx_true))
        if len(idx_true) != 0:
            err.append(pattern_idx)
    false += len(work)
    print(f'FALSE {skip}: {len(work)}')
    print(err)
    print(np.array(patterns['EPS'])[err])
print(false)
'''

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
DELTAS = range(1, 7)
SIZE_PATTERNS = range(3, 6)
SCALER = 'minmax'
N_JOBS = 96

# Создаем инстанс класса расчета расстояний
dist = DTWDistance()

# Подгужаем данные
test = pd.read_csv('source/test_stocks.csv')

work = set()
for DELTA in tqdm(DELTAS):
    for SIZE_PATTERN in tqdm(SIZE_PATTERNS):

        patterns = pd.read_csv(f'/home/avpodtikhov/trendchange/new/{DELTA}_{SIZE_PATTERN}_minmax.csv')

        current_pattern = [1] * SIZE_PATTERN
        # Обрабатываем тестовый датасет

        data, idxs1, idxs2  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=1, return_idxs=True)
        data = scale(data=data, scaler=SCALER)

        D = dist.pairwise(patterns.iloc[:, :-2].values, data, N_JOBS)
        err = []
        for pattern_idx in range(patterns.shape[0]):
            idx_true = np.where(D[pattern_idx] < list(patterns['EPS'])[pattern_idx])[0]
            work = work | set(list(idx_true))
            if len(idx_true) != 0:
                err.append(pattern_idx)
print('TRUE:', len(work))
        # print(len(work))
        # print(err)
        # print(np.array(patterns['EPS'])[err])