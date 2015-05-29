#!/usr/bin/python
# -*- coding: utf8 -*-

import click

import os
#import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import ConfusionMatrix, Backend
from sklearn.metrics import f1_score, classification_report, confusion_matrix

@click.command()
@click.option('--save/--no-save', default=True)
@click.option('--show/--no-show', default=False)
def main(save, show):
    basepath = os.path.dirname(__file__)

    #y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    #y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    #cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    #y_true = [2, 0, 2, 2, 0, 1]
    #y_pred = [0, 0, 2, 2, 0, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    #cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']
    cm = ConfusionMatrix(y_true, y_pred)

    #y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    #y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    #>>> cm(y_true, y_pred, labels=["ant", "bird", "cat"])
    #array([[2, 0, 0],
    #       [0, 0, 1],
    #       [1, 0, 2]])
    #cm = ConfusionMatrix(y_true, y_pred)

    print("Confusion matrix:\n%s" % cm)
    df = cm.to_dataframe()
    print(df)
    print(df.dtypes)

    cm.plot()
    filename = 'cm.png'
    if save:
        plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    if show:
        plt.show()

    cm.plot(normalized=True)
    filename = 'cm_norm.png'
    if save:
        plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    if show:
        plt.show()

    cm.print_stats()
    print(cm.classification_report)

    print("sklearn confusion_matrix:\n%s" % confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    #stat = 'precision'
    #print(cm._avg_stat(stat))
    #print(cm.ACC)

    #import seaborn as sns
    #cm.plot(normalized=True, backend=Backend.Seaborn)
    #sns.plt.show()

    print("Binarize a confusion matrix")
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    cm = ConfusionMatrix(y_true, y_pred)
    print(cm)
    binary_cm = cm.binarize(['ant', 'cat'])
    # A bird is not a "land_animal"
    print(binary_cm)


if __name__ == "__main__":
    main()