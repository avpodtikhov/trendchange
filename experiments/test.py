import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import json
from functools import reduce

tqdm.pandas()
true = pd.read_csv('1_test.csv')
true = true[~true['0'].isna()]
print(true.shape)
print(true.head())
true['len'] = true['0'].apply(lambda x: len(x.split(' ')) - 1)

d = json.load(open('occurences.json', 'r'))
idx_train = []
for i, row in tqdm(true.iterrows(), total=true.shape[0]):
    intervals = row['0'].split(' ')[:-1]
    idx_train.append([])
    for inv1 in intervals:
        idx_train[-1].append(d[inv1])
true['idx_train'] = idx_train
true = true[true['len'] > 1]
true['intersection'] = true['idx_train'].apply(lambda x: len(reduce(np.intersect1d, x)))
print(true.sort_values('intersection'))