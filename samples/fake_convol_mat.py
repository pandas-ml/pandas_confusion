#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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

def create_arrays_vect(df):
    N_all = df.sum().sum()
    sum_true = df.sum(axis=1).shift(1).fillna(0)
    sum_predicted = df.sum(axis=0).shift(1).fillna(0)

    index = np.arange(N_all)
    s_true = pd.Series(index=index)

    s_true[sum_true] = sum_true.index
    s_true = s_true.fillna(method='ffill')
    y_true = s_true.values

    y_predicted = None  # ToDo

    return y_true, y_predicted

def create_arrays(df):
    """
    Create y_true and y_predicted arrays from confusion matrix

    Original idea from
    http://stackoverflow.com/questions/29882747/create-efficiently-fake-truth-predicted-values-from-a-confusion-matrix
    """

    """
    In [156]: df.sum(axis=1).values
    Out[156]: array([8, 9])
    """

    # Unstack to make tuples of actual,pred,count
    df = df.unstack().reset_index()

    # Pull the value labels and counts
    actual = df['Actual'].values
    predicted = df['Predicted'].values
    totals = df.iloc[:,2].values

    # Use list comprehension to create original arrays
    y_true = [[curr_val]*n for (curr_val, n) in zip(actual, totals)]
    y_predicted = [[curr_val]*n for (curr_val, n) in zip(predicted, totals)]

    # They come nested so flatten them
    y_true = [item for sublist in y_true for item in sublist]
    y_predicted = [item for sublist in y_predicted for item in sublist]

    return y_true, y_predicted

def test_fake_convol_mat():
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

    # Recreate the original confusion matrix and check for equality
    y_t, y_p = create_arrays(df)
    conf_mat = confusion_matrix(y_t,y_p)
    check_labels = np.unique(y_t)

    df_new = pd.DataFrame(conf_mat, columns=check_labels, index=check_labels).loc[labels, labels]
    df_new.index.name = 'Actual'
    df_new.columns.name = 'Predicted'

    print(df)
    print(df_new)

    df_comp = df != df_new

    assert df_comp.sum().sum() == 0

def test_fake_binary_convol_mat():
    labels = [False, True]
    df = pd.DataFrame([
        [5, 3],
        [2, 7]], columns=labels, index=labels)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'

    # Recreate the original confusion matrix and check for equality
    y_t, y_p = create_arrays(df)
    conf_mat = confusion_matrix(y_t,y_p)
    check_labels = np.unique(y_t)

    df_new = pd.DataFrame(conf_mat, columns=check_labels, index=check_labels).loc[labels, labels]
    df_new.index.name = 'Actual'
    df_new.columns.name = 'Predicted'

    print df
    print df_new


def main():
    test_fake_binary_convol_mat()
    #test_fake_convol_mat()


if __name__ == "__main__":
    main()

