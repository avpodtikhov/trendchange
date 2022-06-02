# Попытка заключить каждую последовательность предшествующую смене тренда в шар радиуса эпсилон, где эпсилон - максимальное расстояния при котором нет ошибок на тренирвовочной выборке
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
DELTAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZE_PATTERNS = [3, 4, 5]
SCALER = 'minmax'
N_JOBS = 96

# Создаем инстанс класса расчета расстояний
dist = DTWDistance()

# Подгужаем данные
train = pd.read_csv('/home/avpodtikhov/trendchange/source/val_hp.csv')
test = pd.read_csv('/home/avpodtikhov/trendchange/source/val_hp.csv')

for DELTA in tqdm(DELTAS):
    for SIZE_PATTERN in tqdm(SIZE_PATTERNS, leave=False):
        current_pattern = [1] * SIZE_PATTERN

        # Аналогичным образом обрабатываем тренировочный датасет
        patterns  = cut_ts(data=train, pattern=current_pattern, delta=DELTA, skip=0)
        patterns = scale(data=patterns, scaler=SCALER)

        eps = np.full(patterns.shape[0], np.inf)

        for skip in tqdm(range(0, test.shape[1]), leave=False):
            if skip == DELTA:
                continue
            # Обработка тестового датасета со сдвигом на skip шагов
            if 3 + skip + np.sum(current_pattern) >= test.shape[1]:
                break
            data  = cut_ts(data=test, pattern=current_pattern, delta = 0, skip=skip)
            if data.shape[0] == 0:
                break
            data = scale(data=data, scaler=SCALER)
            # Считаем число ложных срабатываний паттернов
            D = dist.pairwise(patterns, data, N_JOBS)
            new_eps = np.min(D, axis=1)
            eps = np.minimum(eps, new_eps)
            ok = np.where(eps > 1e-4)[0]
            patterns = patterns[ok]
            eps = eps[ok]
            df = pd.DataFrame(patterns)
            df['EPS'] = eps
            df['round'] = skip
            df.to_csv(f'new/{DELTA}_{SIZE_PATTERN}_minmax.csv', index=False)