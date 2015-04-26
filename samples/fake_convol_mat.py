#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd


labels = [False, True]
df = pd.DataFrame([
    [5, 3],
    [2, 7]], columns=labels, index=labels)


labels = ['N', 'L', 'R', 'A', 'P', 'V']
df = pd.DataFrame([
    [1971, 19, 1, 8, 0, 1],
    [16, 1940, 2, 23, 9, 10],
    [8, 3, 181, 87, 0, 11],
    [2, 25, 159, 1786, 16, 12],
    [0, 24, 4, 8, 1958, 6],
    [11, 12, 29, 11, 11, 1926] ], columns=labels, index=labels)

N = df.sum().sum()

print(N)

y_true = np.empty(N)
y_pred = np.empty(N)

for i, label_true in enumerate(df.index):
    for j, label_pred in enumerate(df.columns):
        print((i, j), (label_true, label_pred))