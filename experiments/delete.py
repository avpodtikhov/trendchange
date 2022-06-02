# Подсчет пересечений между последовательностями
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

true = pd.read_csv('0_test.csv')
true = true[~true['0'].isna()]
print(true.shape)
print(true.head())
true['len'] = true['0'].apply(lambda x: len(x.split(' ')) - 1)
true = true[true['len'] > 1]
true['intervals'] = true['0'].apply(lambda x: [(int(inv.split('_')[0]), int(inv.split('_')[0]) + int(inv.split('_')[1])) for inv in x.split(' ')[:-1]])
crossess = []
for i, row in tqdm(true.iterrows(), total=true.shape[0]):
    intervals = row['intervals']
    crossess.append([])
    for i1 in range(len(intervals)):
        for i2 in range(i1 + 1, len(intervals)):
            cross = min(intervals[i1][1], intervals[i2][1]) - max(intervals[i1][0], intervals[i2][0])
            if cross < 0:
                crossess[-1].append(abs(cross))
            else:
                crossess[-1].append(0)
true = true.reset_index()
true['crossess'] = crossess
print(true)
true = true[true['crossess'].apply(lambda x: (np.array(x) > 0).sum() > 1)]
print(true)