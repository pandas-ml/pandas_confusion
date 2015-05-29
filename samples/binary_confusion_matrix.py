#!/usr/bin/python
# -*- coding: utf8 -*-

import click

import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import BinaryConfusionMatrix, Backend
from sklearn.metrics import f1_score, classification_report, confusion_matrix

@click.command()
@click.option('--save/--no-save', default=True)
@click.option('--show/--no-show', default=False)
def main(save, show):
    basepath = os.path.dirname(__file__)

    y_true = np.array([True, True, False, False, False, True, False, True, True,
           False, True, False, False, False, False, False, True, False,
            True, True, True, True, False, False, False, True, False,
            True, False, False, False, False, True, True, False, False,
           False, True, True, True, True, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False, True, True, False, True, False, True, True, True,
           False, False, True, False, True, False, False, True, False,
           False, False, False, False, False, False, False, True, False,
            True, True, True, True, False, False, True, False, True,
            True, False, True, False, True, False, False, True, True,
           False, False, True, True, False, False, False, False, False,
           False, True, True, False])
    
    y_pred = np.array([False, False, False, False, False, True, False, False, True,
           False, True, False, False, False, False, False, False, False,
            True, True, True, True, False, False, False, False, False,
           False, False, False, False, False, True, False, False, False,
           False, True, False, False, False, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False, True, False, False, False, False, False, False, False,
           False, False, True, False, False, False, False, True, False,
           False, False, False, False, False, False, False, True, False,
           False, True, False, False, False, False, True, False, True,
            True, False, False, False, True, False, False, True, True,
           False, False, True, True, False, False, False, False, False,
           False, True, False, False])

    #y_true = ~y_true
    #y_pred = ~y_pred

    binary_cm = BinaryConfusionMatrix(y_true, y_pred)
    print("Binary confusion matrix:\n%s" % binary_cm)

    print("")

    attributes = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FDR',
        'FNR', 'ACC', 'F1_score', 'MCC', 'informedness', 'markedness']
    #binary_cm.print_stats(attributes)
    binary_cm.print_stats()
    #stats = binary_cm.stats(attributes)
    #for key, val in stats.items():
    #    print("%s: %f" % (key, val))

    print("sklearn confusion_matrix:\n%s" % confusion_matrix(y_true, y_pred))
    f1score = f1_score(y_true, y_pred)
    print("f1_score: %f" % f1score)

    print("sklearn confusion_matrix_of_rev:\n%s" % confusion_matrix(~y_true, ~y_pred))
    f1score_r = f1_score(~y_true, ~y_pred)
    print("f1_score_of_rev: %f" % f1score_r)

    print(classification_report(y_true, y_pred))
    np.testing.assert_almost_equal(binary_cm.F1_score, f1score)

    binary_cm.plot()
    filename = 'binary_cm.png'
    if save:
        plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    if show:
        plt.show()

    binary_cm.plot(normalized=True)
    filename = 'binary_cm_norm.png'
    if save:
        plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    if show:
        plt.show()

    #import seaborn as sns
    #binary_cm.plot(normalized=True, backend=Backend.Seaborn)
    #sns.plt.show()

    print("FP+TP= %f" % (binary_cm.FP + binary_cm.TP)) # Positive

    print("")
    binary_cm_r = binary_cm.inverse(inplace=False)
    print("Reversed binary confusion matrix:\n%s" % binary_cm_r)
    binary_cm_r.print_stats()
    np.testing.assert_almost_equal(binary_cm_r.F1_score, f1score_r)


    y_true = ["a", "a", "b", "b", "b", "a", "b", "a", "a",
           "b", "a", "b", "b", "b", "b", "b", "a", "b",
            "a", "a", "a", "a", "b", "b", "b", "a", "b",
            "a", "b", "b", "b", "b", "a", "a", "b", "b",
           "b", "a", "a", "a", "a", "b", "b", "b", "b",
            "a", "b", "b", "b", "b", "b", "b", "b", "b",
           "b", "a", "a", "b", "a", "b", "a", "a", "a",
           "b", "b", "a", "b", "a", "b", "b", "a", "b",
           "b", "b", "b", "b", "b", "b", "b", "a", "b",
            "a", "a", "a", "a", "b", "b", "a", "b", "a",
            "a", "b", "a", "b", "a", "b", "b", "a", "a",
           "b", "b", "a", "a", "b", "b", "b", "b", "b",
           "b", "a", "a", "b"]
    
    y_pred = ["b", "b", "b", "b", "b", "a", "b", "b", "a",
           "b", "a", "b", "b", "b", "b", "b", "b", "b",
            "a", "a", "a", "a", "b", "b", "b", "b", "b",
           "b", "b", "b", "b", "b", "a", "b", "b", "b",
           "b", "a", "b", "b", "b", "b", "b", "b", "b",
            "a", "b", "b", "b", "b", "b", "b", "b", "b",
           "b", "a", "b", "b", "b", "b", "b", "b", "b",
           "b", "b", "a", "b", "b", "b", "b", "a", "b",
           "b", "b", "b", "b", "b", "b", "b", "a", "b",
           "b", "a", "b", "b", "b", "b", "a", "b", "a",
            "a", "b", "b", "b", "a", "b", "b", "a", "a",
           "b", "b", "a", "a", "b", "b", "b", "b", "b",
           "b", "a", "b", "b"]
    
    binary_cm = BinaryConfusionMatrix(y_true, y_pred)
    print(binary_cm)
    binary_cm.print_stats()
    print("sklearn confusion_matrix with string as input:\n%s" % confusion_matrix(y_true, y_pred))
    # "b" is considered as "True"
    # "a" is considered as "False"

    #f1score = f1_score(y_true, y_pred)
    np.testing.assert_almost_equal(binary_cm.F1_score, f1score_r)

if __name__ == "__main__":
    main()
