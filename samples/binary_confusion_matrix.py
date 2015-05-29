#!/usr/bin/python
# -*- coding: utf8 -*-

import click

import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import BinaryConfusionMatrix, Backend
#from sklearn.metrics import f1_score, classification_report, confusion_matrix

@click.command()
@click.option('--save/--no-save', default=True)
@click.option('--show/--no-show', default=False)
def main(save, show):
    basepath = os.path.dirname(__file__)

    y_true = [True, True, False, False, False, True, False, True, True,
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
           False, True, True, False]
    
    y_pred = [False, False, False, False, False, True, False, False, True,
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
           False, True, False, False]

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

    #print(confusion_matrix(y_true, y_pred))
    #print("f1_score: %f" % f1_score(y_true, y_pred))
    #print(classification_report(y_true, y_pred))


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

if __name__ == "__main__":
    main()
