#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np
from pandas_confusion import ConfusionMatrix, BinaryConfusionMatrix, Backend, \
    TRUE_NAME_DEFAULT, PREDICTED_NAME_DEFAULT


# =========================================================================


def asserts(confusion_matrix):
    df = confusion_matrix.to_dataframe()
    a = confusion_matrix.to_array()

    assert isinstance(df, pd.DataFrame)
    assert isinstance(a, np.ndarray)

    assert len(df.index) == len(df.columns)
    assert df.index.name == TRUE_NAME_DEFAULT
    assert df.columns.name == PREDICTED_NAME_DEFAULT


# =========================================================================


def test_pandas_confusion_confusion_matrix_strings():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix)

    asserts(confusion_matrix)


def test_pandas_confusion_confusion_matrix_int():
    y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]


    confusion_matrix = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    print("Confusion matrix:\n%s" % confusion_matrix)

    asserts(confusion_matrix)


def test_pandas_confusion_binary_confusion_matrix():
    y_true = [ True,  True, False, False, False,  True, False,  True,  True,
           False,  True, False, False, False, False, False,  True, False,
            True,  True,  True,  True, False, False, False,  True, False,
            True, False, False, False, False,  True,  True, False, False,
           False,  True,  True,  True,  True, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False,  True,  True, False,  True, False,  True,  True,  True,
           False, False,  True, False,  True, False, False,  True, False,
           False, False, False, False, False, False, False,  True, False,
            True,  True,  True,  True, False, False,  True, False,  True,
            True, False,  True, False,  True, False, False,  True,  True,
           False, False,  True,  True, False, False, False, False, False,
           False,  True,  True, False]
    
    y_pred = [False, False, False, False, False,  True, False, False,  True,
           False,  True, False, False, False, False, False, False, False,
            True,  True,  True,  True, False, False, False, False, False,
           False, False, False, False, False,  True, False, False, False,
           False,  True, False, False, False, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False,  True, False, False, False, False, False, False, False,
           False, False,  True, False, False, False, False,  True, False,
           False, False, False, False, False, False, False,  True, False,
           False,  True, False, False, False, False,  True, False,  True,
            True, False, False, False,  True, False, False,  True,  True,
           False, False,  True,  True, False, False, False, False, False,
           False,  True, False, False]

    binary_confusion_matrix = BinaryConfusionMatrix(y_true, y_pred)
    print("Binary confusion matrix:\n%s" % binary_confusion_matrix)

    asserts(binary_confusion_matrix)

"""
def test_pandas_confusion_confusion_matrix_missing_column():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    ##confusion_matrix = ConfusionMatrix(y_true, y_pred)
    confusion_matrix = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix)

    asserts(confusion_matrix)


def test_pandas_confusion_confusion_matrix_missing_row():
    y_true = [2, 0, 2, 2, 0, 0]
    y_pred = [0, 0, 2, 2, 1, 2]
    ##confusion_matrix = ConfusionMatrix(y_true, y_pred)
    confusion_matrix = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix)

    asserts(confusion_matrix)
"""