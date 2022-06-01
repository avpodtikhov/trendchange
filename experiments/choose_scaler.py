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
# Гиперпараметры модели
DELTAS = [1]
SIZE_PATTERNS = range(2, 7)
EPS_RANGE = np.linspace(0.002, 0.2, 100)

# EPS_RANGE = EPS_RANGE[EPS_RANGE >= 0.026]
GAP = 5
SCALER_RANGE = ['minmax', 'none', 'max']
N_JOBS = 96

# Создаем инстанс класса расчета расстояний
dist = DTWDistance()

# Подгужаем данные
train = pd.read_csv('/home/avpodtikhov/trendchange/source/train_hp.csv')
test = pd.read_csv('/home/avpodtikhov/trendchange/source/val_hp.csv')

for DELTA in DELTAS:
    for SIZE_PATTERN in SIZE_PATTERNS:
        for SCALER in tqdm(SCALER_RANGE):
            current_pattern = [1] * SIZE_PATTERN

            # Обрабатываем тестовый датасет
            data  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=0)
            data = scale(data=data, scaler=SCALER)

            # Аналогичным образом обрабатываем тренировочный датасет
            data1  = cut_ts(data=train, pattern=current_pattern, delta=DELTA, skip=0)
            data1 = scale(data=data1, scaler=SCALER)

            # Расчитываем расстояние между всем сэмплами из тестового и тренировочного датасетов
            D = dist.pairwise(data1, data, N_JOBS)
            stop = False
            cur = 0
            l = 0
            for eps in tqdm(EPS_RANGE, leave=False):
                # Кол-во верных срабатываний с данным EPS для каждого паттерна из тренировочного набора
                length_true = (D <= eps).sum(axis=1)
                # Индексы паттернов, которые сработали на тестовом наборе данных
                idx = np.where(length_true != 0)[0]
                # Исключаем паттерны,  которые ни разу не сработали
                patterns = data1[idx]
                for skip in tqdm(range(20, test.shape[1]), leave=False):
                    # Обработка тестового датасета со сдвигом на skip шагов
                    if 3 + skip + DELTA + np.sum(current_pattern) >= test.shape[1]:
                        break
                    data  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=skip)
                    if data.shape[0] == 0:
                        break
                    data = scale(data=data, scaler=SCALER)
                    # Считаем число ложных срабатываний паттернов
                    D1 = dist.coocurences(patterns, data, eps, N_JOBS)
                    # Индексы паттернов сработавших ложно 1 раз
                    idx_false = np.where(D1 == 0)[0]
                    # Исключаем ложносрабатывающие паттерны
                    patterns = patterns[idx_false]

                    # Сохрняем промежуточные результаты
                    df = pd.DataFrame(patterns)
                    df['round'] = skip
                    path = os.path.join('/home/avpodtikhov/trendchange/results/choose_scaler/detrended/' + SCALER, '{}.csv'.format(eps))
                    df.to_csv(path,  index=False)
                    if (patterns.shape[0] == 0):
                        stop = True
                        break
                if patterns.shape[0] < cur:
                    l += 1
                else:
                    l = 0
                if (l >= GAP) or (patterns.shape[0] < 10):
                    stop = True
                cur = patterns.shape[0]
                print(cur, l)
                if stop:
                    break