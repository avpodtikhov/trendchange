import numpy as np
import pandas as pd
from scipy import stats

def scale(data, scaler):
    if scaler == 'minmax':
        data = (data - data.min(axis=1).reshape(-1, 1)) / (data.max(axis=1).reshape(-1, 1) - data.min(axis=1).reshape(-1, 1))
    elif scaler == 'max':
        data = data / data.max(axis=1).reshape(-1, 1)
    elif scaler == 'z':
        data = stats.zscore(data, axis=1)
    elif scaler == 'none':
        return data
    else:
        return None
    return data

def cut_ts(data, pattern, delta, skip, return_idxs = False, detrended=False):
    if return_idxs:
        if detrended:
            idxs1 = np.where(np.isnan(data.iloc[:, 3 + skip + delta : 3 + skip + delta + np.sum(pattern) + 2].values).sum(axis=1) == 0)[0]
        else:
            idxs1 = np.where(np.isnan(data.iloc[:, 3 + skip + delta : 3 + skip + delta + np.sum(pattern) + 1].values).sum(axis=1) == 0)[0]

    if detrended:
        data = data[np.isnan(data.iloc[:, 3 + skip + delta : 3 + skip + delta + np.sum(pattern) + 2].values).sum(axis=1) == 0]
        data = data.iloc[:, 3 + skip + delta : 3 + skip + delta + np.sum(pattern) + 2].values
        data = data[:, :-1] - data[:, 1:]
    else:
        data = data[np.isnan(data.iloc[:, 3 + skip + delta : 3 + skip + delta + np.sum(pattern) + 1].values).sum(axis=1) == 0]
        data = data.iloc[:, 3 + skip + delta : 3 + skip + delta + np.sum(pattern) + 1].values
    idxs = [0]
    for p in pattern:
        idxs.append(idxs[-1] + p)
    data = data[:, np.array(idxs)]
    data = data[:, ::-1]
    if return_idxs:
        idxs2 = np.where(data.max(axis=1) != data.min(axis=1))[0]
    data = data[data.max(axis=1) != data.min(axis=1)]
    if return_idxs:
        return data, idxs1, idxs2
    return data
