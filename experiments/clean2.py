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
test = pd.read_csv('source/train_stocks.csv')

for DELTA in tqdm(DELTAS):
    for SIZE_PATTERN in tqdm(SIZE_PATTERNS, leave=False):
        current_pattern = [1] * SIZE_PATTERN
        stop = False
        for eps in [SIZE_TO_EPS[SIZE_PATTERN]]:
            patterns = pd.read_csv(f'/home/avpodtikhov/trendchange/results/hun/clean/{SIZE_PATTERN}/{DELTA}.csv')
            patterns = patterns[patterns['FN'] == 0]
            for skip in tqdm(range(1, test.shape[1]), leave=False):
                if skip == DELTA:
                    continue
                # Обработка тестового датасета со сдвигом на skip шагов
                data  = cut_ts(data=test, pattern=current_pattern + [1], delta=0, skip=skip)
                data = scale(data=data, scaler=SCALER)
                # Считаем число ложных срабатываний паттернов
                D1 = dist.coocurences(patterns.iloc[:, :-2].values, data, eps, N_JOBS)
                # Индексы паттернов сработавших ложно 1 раз
                patterns['FN'] = patterns['FN'] + D1
                patterns = patterns[patterns['FN'] <= 0]

                data  = cut_ts(data=test, pattern=current_pattern[1:], delta=0, skip=skip)
                data = scale(data=data, scaler=SCALER)
                # Считаем число ложных срабатываний паттернов
                D1 = dist.coocurences(patterns.iloc[:, :-2].values, data, eps, N_JOBS)
                # Индексы паттернов сработавших ложно 1 раз
                patterns['FN'] = patterns['FN'] + D1
                patterns = patterns[patterns['FN'] <= 0]

                # Сохрняем промежуточные результаты
                patterns['round'] = patterns['round'] + 1
                path = os.path.join('/home/avpodtikhov/trendchange/results/hun/clean2/', '{}'.format(SIZE_PATTERN), '{}.csv'.format(DELTA))
                os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
                patterns.to_csv(path,  index=False)
                if patterns.shape[0] == 0:
                    stop = True
                    break
            if stop:
                break