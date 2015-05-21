#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np
from pandas_confusion import ConfusionMatrix, BinaryConfusionMatrix, Backend, \
    TRUE_NAME_DEFAULT, PRED_NAME_DEFAULT
from sklearn.metrics import confusion_matrix


# =========================================================================


def asserts(y_true, y_pred, cm):
    df = cm.to_dataframe()
    a = cm.to_array()

    df_with_sum = cm.to_dataframe(calc_sum=True)

    assert len(y_true) == len(y_pred)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(a, np.ndarray)
    assert isinstance(df_with_sum, pd.DataFrame)

    N = len(df.index)
    assert N == len(df.columns)
    assert cm.len() == len(df.columns)
    
    assert df.index.name == TRUE_NAME_DEFAULT, "%r != %r" % (df.index.name, TRUE_NAME_DEFAULT)
    assert df.columns.name == PRED_NAME_DEFAULT, "%r != %r" % (df.columns.name, PRED_NAME_DEFAULT)

    assert df_with_sum.index.name == TRUE_NAME_DEFAULT, "%r != %r" % (df_with_sum.index.name, TRUE_NAME_DEFAULT)
    assert df_with_sum.columns.name == PRED_NAME_DEFAULT, "%r != %r" % (df_with_sum.columns.name, PRED_NAME_DEFAULT)

    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), cm.toarray())

    assert cm.sum() == len(y_true)

    assert cm.true.name == TRUE_NAME_DEFAULT, "%r != %r" % (cm.true.name, TRUE_NAME_DEFAULT)
    assert cm.pred.name == PRED_NAME_DEFAULT, "%r != %r" % (cm.pred.name, PRED_NAME_DEFAULT)

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

def test_pandas_confusion_cm_empty_column():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    cm = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % cm)

    asserts(y_true, y_pred, cm)


def test_pandas_confusion_cm_empty_row():
    y_true = [2, 0, 2, 2, 0, 0]
    y_pred = [0, 0, 2, 2, 1, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    cm = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % cm)

    asserts(y_true, y_pred, cm)

def test_pandas_confusion_cm_binarize():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    cm = ConfusionMatrix(y_true, y_pred)
    print("Confusion matrix:\n%s" % cm)

    select = ['cat', 'dog']
    print("Binarize with %s" % select)
    binary_cm = cm.binarize(select)

    print("Binary confusion matrix:\n%s" % binary_cm)
    
    assert cm.sum() == binary_cm.sum()

def test_value_counts():
    df = pd.DataFrame({
        'Height': [150, 150, 151, 151, 152, 155, 155, 157, 157, 157, 157, 158, 158, 159, 159, 159, 160, 160, 162, 162, 163, 164, 165, 168, 169, 169, 169, 170, 171, 171, 173, 173, 174, 176, 177, 177, 179, 179, 179, 179, 179, 181, 181, 182, 183, 184, 186, 190, 190],
        'Weight': [54, 55, 55, 47, 58, 53, 59, 60, 56, 55, 62, 56, 55, 55, 64, 61, 59, 59, 63, 66, 64, 62, 66, 66, 72, 65, 75, 71, 70, 70, 75, 65, 79, 78, 83, 75, 84, 78, 74, 75, 74, 90, 80, 81, 90, 81, 91, 87, 100],
        'Size': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL'],
        'SizePred': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'L', 'XL', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'M', 'L', 'L', 'M', 'L', 'M', 'M', 'M']
    })
    cm = ConfusionMatrix(df["Size"], df["SizePred"])
    assert (cm.true - df.Size.value_counts()).sum() == 0
    assert (cm.pred - df.SizePred.value_counts()).sum() == 0
