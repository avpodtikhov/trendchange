import pandas as pd
from tqdm.auto import tqdm
import numpy as np

true = pd.read_csv('100_0_train_clean.csv')
true = true[~true['0'].isna()]
print(true.shape)
print(true.head())
true['len'] = true['0'].apply(lambda x: len(x.split(' ')) - 1)

d = {}
for i, row in tqdm(true.iterrows(), total=true.shape[0]):
    intervals = row['0'].split(' ')[:-1]
    for inv1 in intervals:
        if inv1 not in d:
            d[inv1] = []
        d[inv1].append(i)

import json

json.dump(d, open('occurences100.json', 'w'))