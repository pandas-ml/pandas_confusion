#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np
from pandas_confusion import ConfusionMatrix, BinaryConfusionMatrix, Backend, \
    TRUE_NAME_DEFAULT, PREDICTED_NAME_DEFAULT
from sklearn.metrics import confusion_matrix


# =========================================================================


def asserts(y_true, y_pred, cm):
    df = cm.to_dataframe()
    a = cm.to_array()

    assert isinstance(df, pd.DataFrame)
    assert isinstance(a, np.ndarray)

    assert len(df.index) == len(df.columns)
    #assert df.index.name == TRUE_NAME_DEFAULT
    #assert df.columns.name == PREDICTED_NAME_DEFAULT

    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), cm.toarray())

# =========================================================================


def test_pandas_confusion_cm_strings():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    cm = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % cm)

    asserts(y_true, y_pred, cm)

def test_pandas_confusion_cm_int():
    y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]


    labels = ["ant", "bird", "cat"]
    cm = ConfusionMatrix(y_true, y_pred, labels=labels)
    print("Confusion matrix:\n%s" % cm)

    asserts(y_true, y_pred, cm)

    assert cm.len() == len(labels)

    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), cm.toarray())


def test_pandas_confusion_binary_cm():
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

    binary_cm = BinaryConfusionMatrix(y_true, y_pred)
    print("Binary confusion matrix:\n%s" % binary_cm)

    asserts(y_true, y_pred, binary_cm)

def test_pandas_confusion_cm_missing_column():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    cm = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % cm)

    asserts(y_true, y_pred, cm)


def test_pandas_confusion_cm_missing_row():
    y_true = [2, 0, 2, 2, 0, 0]
    y_pred = [0, 0, 2, 2, 1, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    cm = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % cm)

    asserts(y_true, y_pred, cm)
