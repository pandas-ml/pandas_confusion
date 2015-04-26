#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import ConfusionMatrix, Backend

def main():
    basepath = os.path.dirname(__file__)

    #y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2])
    #y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2])
    y_actu = pd.Series(['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit'])
    y_pred = pd.Series(['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit'])

    confusion_matrix = ConfusionMatrix(y_actu, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix)

    confusion_matrix.plot()
    filename = 'confusion_matrix.png'
    plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    plt.show()

    confusion_matrix.plot(normalized=True)
    filename = 'confusion_matrix_norm.png'
    plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    plt.show()

    #confusion_matrix.plot(normalized=True, backend=Backend.Seaborn)
    #plt.show()

if __name__ == "__main__":
    main()