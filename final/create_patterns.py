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
SIZE_PATTERNS = [3]
# EPS_RANGE = [0.01]
SCALER = 'max'
N_JOBS = 96

# Создаем инстанс класса расчета расстояний
dist = DTWDistance()

# Подгужаем данные
train = pd.read_csv('/home/avpodtikhov/trendchange/source/train_hp.csv')
test = pd.read_csv('/home/avpodtikhov/trendchange/source/train_hp.csv')
for DELTA in tqdm(DELTAS):
    for SIZE_PATTERN in tqdm(SIZE_PATTERNS, leave=False):
        current_pattern = [1] * SIZE_PATTERN

        # Обрабатываем тестовый датасет
        data  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=0 )
        data = scale(data=data, scaler=SCALER)

        # Аналогичным образом обрабатываем тренировочный датасет
        data1  = cut_ts(data=train, pattern=current_pattern, delta=DELTA, skip=0)
        data1 = scale(data=data1, scaler=SCALER)

        # Расчитываем расстояние между всем сэмплами из тестового и тренировочного датасетов
        # D = dist.pairwise(data1, data, N_JOBS)
        stop = False
        for eps in [0.016]:
            # Кол-во верных срабатываний с данным EPS для каждого паттерна из тренировочного набора
            # length_true = (D < eps).sum(axis=1)
            # Индексы паттернов, которые сработали на тестовом наборе данных
            # idx = np.where(length_true != 0)[0]
            # Исключаем паттерны,  которые ни разу не сработали
            # patterns = data1[idx]
            patterns = data1
            # length_true = length_true[idx]
            length_false = np.array([0] * data1.shape[0])
            for skip in tqdm(range(1, test.shape[1]), leave=False):
                # Обработка тестового датасета со сдвигом на skip шагов
                if 3 + skip + DELTA + np.sum(current_pattern) >= test.shape[1]:
                    break
                data  = cut_ts(data=test, pattern=current_pattern, delta=DELTA, skip=skip)
                if data.shape[0] == 0:
                    break
                data = scale(data=data, scaler=SCALER)
                # Считаем число ложных срабатываний паттернов
                D1 = dist.coocurences(patterns, data, eps, N_JOBS)
                length_false = length_false + D1
                # Индексы паттернов сработавших ложно 1 раз
                idx_false = np.where(length_false == 0)[0]
                # Исключаем ложносрабатывающие паттерны
                patterns = patterns[idx_false]
                # length_true = length_true[idx_false]
                length_false = length_false[idx_false]

                # Сохрняем промежуточные результаты
                df = pd.DataFrame(patterns)
                # df['TP'] = length_true
                df['FN'] = length_false
                df['round'] = skip
                # df.to_csv('test.csv', index=False)
                path = os.path.join('./results/hun_zer1/', '{}'.format(SIZE_PATTERN), '{}_{}.csv'.format(DELTA, eps))
                os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
                df.to_csv(path,  index=False)
                if patterns.shape[0] == 0:
                    stop = True
                    break
            if stop:
                break