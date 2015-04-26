#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd


def fake_binary_convolution_matrix():
    """
    Returns a fake binary convolution matrix
    """
    labels = [False, True]
    df = pd.DataFrame([
        [5, 3],
        [2, 7]], columns=labels, index=labels)

    df.index.name = 'Actual'
    df.columns.name = 'Predicted'

    return(df)


def fake_convolution_matrix():
    """
    Returns a fake convolution matrix
    """
    labels = ['N', 'L', 'R', 'A', 'P', 'V']
    df = pd.DataFrame([
        [1971, 19, 1, 8, 0, 1],
        [16, 1940, 2, 23, 9, 10],
        [8, 3, 181, 87, 0, 11],
        [2, 25, 159, 1786, 16, 12],
        [0, 24, 4, 8, 1958, 6],
        [11, 12, 29, 11, 11, 1926] ], columns=labels, index=labels)

    df.index.name = 'Actual'
    df.columns.name = 'Predicted'

    return(df)


df = fake_binary_convolution_matrix()
N_all = df.sum().sum()

print(N_all)

dtype = bool

y_true = np.empty(N_all, dtype=dtype)
y_pred = np.empty(N_all, dtype=dtype)



print(df)
k_true = 0
k_pred = 0
for i, label_true in enumerate(df.index):
    for j, label_pred in enumerate(df.columns):
        N = df.iloc[i,j]
        print(label_true, label_pred, N)
        
        for k in range(k_true, k_true + N):
            y_true[k] = label_true
        k_true = k

        for k in range(k_pred, k_pred + N):
            y_pred[k] = label_pred
        k_pred = k


"""
http://stackoverflow.com/questions/29882747/create-efficiently-fake-truth-predicted-values-from-a-confusion-matrix

In [156]: df.sum(axis=1).values
Out[156]: array([8, 9])
"""
